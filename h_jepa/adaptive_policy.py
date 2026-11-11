"""Adaptive replanning policy for H-JEPA / HWM checkpoints.

The planner commits to ``receding_horizon`` action blocks of ``action_block``
environment steps each (Push-T: 2 blocks x 5 steps = re-plan every 10 steps).
This policy keeps that plan, but re-scores it after the first block using the
world model's own prediction:

    z_hat_{t+k} = F  (z_t, a_t^{(1)})      # k = action_block, one predict call
    z_{t+k}     = E  (o_{t+k})             # same encoder path as planning
    e_t         = ||z_hat - z||^2 / (D * sigma_z^2 + eps)

    xi_t = k   if e_t > tau      (drop the rest of the plan, re-plan now)
    xi_t = 2k  otherwise         (execute the second block as usual)

Step 1 of the rollout plan is ``log_only``: no decision is taken, every check is
appended to a JSONL file so ``tau`` can be calibrated from the empirical
distribution of ``e_t`` instead of being guessed.

Design notes
------------
* The predicted latent is taken from the solver's final rollout
  (``predictions['predicted_embed_0']``), so scoring costs one encoder forward
  per check and nothing else.
* Buffers are per-environment: an environment that triggers an early re-plan
  does not perturb the plan of its neighbours. The solver still plans for the
  whole batch and the fresh plan is only installed into the environments that
  actually asked for it.
* Base-class ``get_action`` was split into ``_prepare_info`` + ``_next_action``
  so the check can reuse the already transformed observation (image transforms
  are not idempotent).
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_worldmodel.policy import WorldModelPolicy


class AdaptiveWorldModelPolicy(WorldModelPolicy):
    """World-model policy that can cut an action block short when the model drifts."""

    def __init__(
        self,
        *,
        adaptive: dict[str, Any] | None = None,
        solver: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(solver=solver, config=config, **kwargs)

        cfg = dict(adaptive or {})
        self.adapt_cfg = cfg
        self.enabled = bool(cfg.get('enabled', False))
        self.log_only = bool(cfg.get('log_only', True))
        self.check_after = int(
            cfg.get('check_after', self.execution_plan_config.action_block)
        )
        tau = cfg.get('tau', None)
        self.tau = None if tau is None else float(tau)
        self.norm = str(cfg.get('norm', 'global'))
        self.eps = float(cfg.get('eps', 1e-8))
        alpha = cfg.get('ema_alpha', None)
        self.ema_alpha = None if alpha is None else float(alpha)

        sigma = cfg.get('sigma_z', None)
        self._sigma2 = (
            None if sigma is None else torch.as_tensor(sigma, dtype=torch.float32)
        )

        self.log_path: Path | None = None
        if cfg.get('log_path', None):
            self.log_path = Path(
                os.path.expandvars(str(cfg['log_path']))
            ).expanduser()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text('')  # fresh file per eval run

        # per-environment state
        self._env_queues: list[deque] = []
        self._since_plan: list[int] = []
        self._anchor: list[torch.Tensor | None] = []
        self._ema: np.ndarray | None = None
        self._check_count = 0
        self._trigger_count = 0

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear per-environment buffers and prediction anchors."""
        super().reset()
        n_envs = int(getattr(self.env, 'num_envs', 1))
        self._env_queues = [deque() for _ in range(n_envs)]
        self._since_plan = [0] * n_envs
        self._anchor = [None] * n_envs
        self._ema = np.zeros(n_envs, dtype=np.float64)

    # ------------------------------------------------------------------ action

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Score the running plan (optional) and return the next action."""
        assert hasattr(self, 'env'), 'Environment not set for the policy'

        info_dict = self._prepare_info(info_dict)
        if self.enabled:
            self._maybe_check(info_dict)
        return self._next_action(info_dict)

    def _next_action(self, info_dict: dict) -> np.ndarray:
        """Pop one action per environment, planning for the ones out of actions."""
        if not self.enabled:
            return super()._next_action(info_dict)

        if any(len(q) == 0 for q in self._env_queues):
            self._plan_and_fill(info_dict)

        action = torch.stack([q.popleft() for q in self._env_queues], dim=0)
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        self._steps_taken += 1
        for env_idx in range(len(self._env_queues)):
            self._since_plan[env_idx] += 1
        return action

    def _plan_and_fill(self, info_dict: dict) -> None:
        """Run the solver once and fill the buffers of the environments that need it."""
        remaining_steps = self._remaining_eval_steps()
        if remaining_steps is not None and remaining_steps <= 0:
            raise RuntimeError('Evaluation step budget exhausted.')

        self._policy_call_count += 1
        print(f'policy call number: {self._policy_call_count}')
        outputs = self.solver(
            info_dict,
            steps_taken=self._steps_taken,
            eval_budget=self.eval_budget,
        )

        actions = outputs['actions']  # (num_envs, horizon, action_dim)
        keep_horizon = min(self.cfg.receding_horizon, actions.shape[1])
        plan = actions[:, :keep_horizon].reshape(
            self.env.num_envs,
            keep_horizon * self.execution_plan_config.action_block,
            -1,
        )
        plan = plan.transpose(0, 1)  # (T, num_envs, action_dim)

        needs_plan = [i for i, q in enumerate(self._env_queues) if len(q) == 0]
        for env_idx in needs_plan:
            self._env_queues[env_idx].extend(plan[:, env_idx])
            self._since_plan[env_idx] = 0

        self._update_anchor(outputs.get('predicted_embed_1'), needs_plan)

    # ------------------------------------------------------------------- score

    def _update_anchor(self, predicted: Any, env_ids: list[int]) -> None:
        """Store the model's own prediction of the latent at the next macro step."""
        if predicted is None:
            return
        if not torch.is_tensor(predicted):
            predicted = torch.as_tensor(predicted)

        predicted = predicted.detach().float().cpu()
        if predicted.dim() == 3:
            # (B, T, D): index 0 is the context frame, index 1 the next macro step
            predicted = predicted[:, 1] if predicted.shape[1] > 1 else predicted[:, -1]
        elif predicted.dim() != 2:
            raise ValueError(
                f'Unexpected predicted_embed_1 shape {tuple(predicted.shape)}'
            )

        for env_idx in env_ids:
            if env_idx < predicted.shape[0]:
                self._anchor[env_idx] = predicted[env_idx]

    def _normalise(self, sq: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
        """Turn squared latent deltas into the scale-free score e_t."""
        sigma = self._sigma2
        if sigma is None:
            return mse
        if sigma.numel() > 1:
            return (sq / (sigma.view(1, -1) + self.eps)).mean(dim=-1)
        return mse / (float(sigma.reshape(-1)[0]) + self.eps)

    def _maybe_check(self, info_dict: dict) -> None:
        """Compare the plan's predicted latent against the freshly encoded one."""
        due = [
            env_idx
            for env_idx in range(len(self._env_queues))
            if self._since_plan[env_idx] == self.check_after
            and len(self._env_queues[env_idx]) > 0
            and self._anchor[env_idx] is not None
        ]
        if not due:
            return

        with torch.no_grad():
            embeddings = self.solver._encode_all_levels(
                info_dict=info_dict, key='pixels'
            )
            z = embeddings['embed_1'][:, -1].detach().float().cpu()  # (B, D)

        # score only the environments that are actually due: an environment can
        # be mid-block after an early re-plan and have no anchor for this step
        z_due = z[torch.as_tensor(due, dtype=torch.long)]
        anchors = torch.stack([self._anchor[i] for i in due])
        sq = (z_due - anchors) ** 2
        mse = sq.mean(dim=-1)  # ||dz||^2 / D
        e = self._normalise(sq, mse)
        z_sq = (z_due ** 2).mean(dim=-1)

        score = e
        if self.ema_alpha is not None:
            e_np = e.numpy().astype(np.float64)
            for pos, env_idx in enumerate(due):
                self._ema[env_idx] = (
                    self.ema_alpha * e_np[pos]
                    + (1.0 - self.ema_alpha) * self._ema[env_idx]
                )
            score = torch.from_numpy(self._ema[due].copy())

        self._check_count += 1
        tau = self.tau
        triggered = []
        for pos, env_idx in enumerate(due):
            hit = bool(tau is not None and score[pos].item() > tau)
            if hit:
                triggered.append(env_idx)

            if self.log_path is not None:
                record = {
                    'check': int(self._check_count),
                    't': int(self._steps_taken),
                    'call': int(self._policy_call_count),
                    'env': int(env_idx),
                    'D': int(z.shape[-1]),
                    'mse': float(mse[pos].item()),
                    'z_sq': float(z_sq[pos].item()),
                    'e': float(e[pos].item()),
                    'score': float(score[pos].item()),
                    'trigger': hit,
                }
                with self.log_path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')

        if triggered and not self.log_only:
            self._trigger_count += len(triggered)
            for env_idx in triggered:
                self._env_queues[env_idx].clear()
                self._since_plan[env_idx] = 0

    # -------------------------------------------------------------- diagnostics

    @property
    def adaptive_stats(self) -> dict[str, Any]:
        """Cheap summary of what the policy observed (for logs / metrics)."""
        return {
            'checks': int(self._check_count),
            'triggers': int(self._trigger_count),
            'policy_calls': int(self._policy_call_count),
            'log_path': None if self.log_path is None else str(self.log_path),
        }
