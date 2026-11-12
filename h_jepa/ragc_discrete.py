"""RAGC-Discrete: bounded DISCRETE action selection (no gradient descent).

The gradient path was not validated (A* directional-derivative agreement on the
held-out anchors was 13/20, p=0.13), so this module deliberately does something
much narrower:

    candidates = { A*, A* +/- eps*u1, A* +/- eps*u2 }
    pick the candidate with the lowest predicted cost
    switch to it ONLY IF  Q(A*) - Q(best) > margin
    and the candidate is legal and does not regress the ORIGINAL model cost

The head is frozen and never differentiated.  `RAGCRefiner` (gradient) is NOT
used.  Every guard failure returns the original A* untouched.

Guarantees:
  * never consumes the planner's torch RNG (candidates use a private NumPy RNG)
  * only acts on the FIRST planning call (`steps_taken == 0`) with [B,2,10]
  * no-op when disabled, so `enabled=false` reproduces the baseline exactly
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ragc_oracle import antithetic_candidates


@dataclass(frozen=True)
class DiscreteConfig:
    epsilon: float = 0.30
    directions: int = 2
    margin: float = 0.05
    max_original_cost_regression: float = 0.10
    apply_steps_taken: int = 0
    seed: int = 0


@dataclass
class SelectionStats:
    n_calls: int = 0
    n_considered: int = 0
    n_switched: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    last_gain: float | None = None
    last_choice: int = 0
    last_exception: str | None = None

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


class RAGCDiscreteSelector:
    """Post-solve discrete candidate chooser for the level-1 plan."""

    def __init__(
        self,
        head: torch.nn.Module,
        config: DiscreteConfig,
        action_mean,
        action_std,
        *,
        device: torch.device | str | None = None,
    ):
        self.head = head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.cfg = config
        if device is not None:
            self.device = torch.device(device)
        else:
            first = next(iter(head.parameters()), None)
            self.device = first.device if first is not None else torch.device("cpu")
        self.mean = torch.as_tensor(action_mean, dtype=torch.float32).reshape(2)
        self.std = torch.as_tensor(action_std, dtype=torch.float32).reshape(2)
        if not bool((self.std > 0).all()):
            raise ValueError("Invalid action normalization std")
        self.stats = SelectionStats()
        self._rng_calls = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _raw_actions(self, tokens: np.ndarray) -> np.ndarray:
        """[H,10] standardized tokens -> [5H,2] environment actions."""
        mean = self.mean.cpu().numpy().reshape(1, 2)
        std = self.std.cpu().numpy().reshape(1, 2)
        return np.asarray(tokens, np.float32).reshape(-1, 2) * std + mean

    def _legal(self, tokens: np.ndarray) -> bool:
        raw = self._raw_actions(tokens)
        return bool(np.isfinite(raw).all() and np.abs(raw).max() <= 1.0 + 1e-6)

    @staticmethod
    def _broadcast_info(solver, level_info: dict, n: int):
        d = solver._expand_info_dict_for_samples(
            level_info, start_idx=0, end_idx=n, num_samples=1,
        )
        d["cost_last_n"] = int(solver._config.cost_last_n)
        d["intermediate_cost_weight"] = float(
            solver._config.intermediate_cost_weight)
        return d

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def select(
        self,
        *,
        solver,
        level_info: dict,
        original_actions: torch.Tensor,
        steps_taken: int = 0,
    ) -> torch.Tensor:
        # A selector must never break a rollout: any unexpected failure is
        # converted into "keep the original plan" and recorded.
        try:
            return self._select(
                solver=solver, level_info=level_info,
                original_actions=original_actions, steps_taken=steps_taken)
        except Exception as exc:  # noqa: BLE001 - deliberate safety net
            self.stats.note(f"exception:{type(exc).__name__}")
            self.stats.last_exception = str(exc)[:200]
            return original_actions

    def _select(
        self,
        *,
        solver,
        level_info: dict,
        original_actions: torch.Tensor,
        steps_taken: int = 0,
    ) -> torch.Tensor:
        cfg = self.cfg
        self.stats.n_calls += 1

        if steps_taken != cfg.apply_steps_taken:
            self.stats.note("not_first_planning_step")
            return original_actions
        if original_actions.ndim != 3 or original_actions.shape[-1] != 10:
            self.stats.note("unsupported_action_shape")
            return original_actions
        if original_actions.shape[1] != 2:
            self.stats.note("unsupported_horizon")
            return original_actions
        if original_actions.shape[0] != 1:
            self.stats.note("unsupported_batch")
            return original_actions

        self.stats.n_considered += 1
        base_np = original_actions[0].detach().cpu().numpy().astype(np.float32)
        if not self._legal(base_np):
            # The plan itself is outside the env box: do not build candidates on
            # top of it (any comparison would be confounded by clipping).
            self.stats.note("original_actions_illegal")
            return original_actions

        # ---- candidates (private RNG; planner RNG untouched) ----------- #
        self._rng_calls += 1
        rng = np.random.default_rng(cfg.seed + self._rng_calls)
        cands = antithetic_candidates(base_np, cfg.epsilon, cfg.directions, rng)

        legal_idx = [i for i, c in enumerate(cands) if self._legal(c)]
        if 0 not in legal_idx:
            self.stats.note("original_actions_illegal")
            return original_actions

        # ---- head scoring (no autograd) -------------------------------- #
        acts = torch.as_tensor(np.stack(cands), dtype=torch.float32,
                               device=self.device).unsqueeze(0)  # [1,C,2,10]
        z, g, upper = self._latent_context(level_info)
        c = acts.shape[1]
        with torch.no_grad():
            q = self.head(
                z.repeat_interleave(c, 0), g.repeat_interleave(c, 0),
                acts[0], upper.repeat_interleave(c, 0),
            ).reshape(c)
        q = q.detach().cpu().numpy()
        if not np.isfinite(q).all():
            self.stats.note("nonfinite_head_output")
            return original_actions

        legal_mask = np.zeros(c, dtype=bool)
        legal_mask[legal_idx] = True
        order = np.argsort(q)
        best = next((int(i) for i in order if i != 0 and legal_mask[i]), None)
        if best is None:
            self.stats.note("no_legal_candidate")
            return original_actions

        gain = float(q[0] - q[best])
        self.stats.last_gain = gain
        if gain <= cfg.margin:
            self.stats.note("margin_not_met")
            return original_actions

        # ---- original-model cost protection ---------------------------- #
        model = solver.model
        info = self._broadcast_info(solver, level_info, 1)
        base_dev = original_actions.detach().to(self.device, dtype=torch.float32)
        old_cost = float(model.get_cost(info.copy(), base_dev.unsqueeze(1)).item())
        cand_t = torch.as_tensor(cands[best], dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        new_cost = float(model.get_cost(info.copy(), cand_t.unsqueeze(1)).item())
        allowed = old_cost + cfg.max_original_cost_regression * max(
            abs(old_cost), 1.0)
        if not np.isfinite(new_cost) or new_cost > allowed:
            self.stats.note("original_cost_regression")
            return original_actions

        self.stats.n_switched += 1
        self.stats.last_choice = best
        self.stats.note("switched")
        return cand_t.detach().to(
            device=original_actions.device, dtype=original_actions.dtype)

    def _latent_context(self, level_info: dict):
        embed = level_info["embed_0"]
        goal = level_info["goal_embed_0"]
        z = embed[0, -1].detach().to(self.device).float().reshape(1, -1)
        g = goal[0, -1].detach().to(self.device).float().reshape(1, -1)
        upper = bool(level_info.get("goal_embed_0_in_upper_space", False))
        up = torch.as_tensor([upper], dtype=torch.bool, device=self.device)
        return z, g, up
