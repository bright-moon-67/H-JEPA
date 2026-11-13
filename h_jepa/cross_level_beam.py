"""CLFB: cross-level feasibility-guided selection for frozen AntMaze L3 H-JEPA.

Reference implementation for branch research/ant-trace-audit-v1 @ 48ef1c2.
The top-level GD solver still optimizes its original 400 candidates. This module
selects K different *existing* L3 candidates using L3 rollout diversity, probes
L2 reachability with a separate action optimizer, and passes the chosen L3 plan
to the original downstream L2/L1 solvers.

No privileged simulation state or true environment reward is used.
This is research reference code; validate shape, RNG/no-op gates on the real model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CLFBConfig:
    enabled: bool = False
    top_m: int = 32
    beam_width: int = 4
    probe_samples: int = 16
    probe_steps: int = 30
    seed: int = 420042
    # Stage-A gate only: run the whole probe pipeline but force the original
    # L3 winner.  Must reproduce the baseline exactly.
    force_original: bool = False

    def __post_init__(self):
        if not (1 <= self.beam_width <= self.top_m):
            raise ValueError("Require 1 <= beam_width <= top_m")
        if self.probe_samples < 1 or self.probe_steps < 1:
            raise ValueError("probe_samples and probe_steps must be positive")


def diverse_beam_indices(pred: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-first over *complete predicted latent trajectories*.

    pred: [B, M, T+1, D], already sorted by increasing L3 cost.
    The original cost winner (index 0) is always the first beam member.
    Ties select the lower-cost candidate due to sorted pool order.
    """
    if pred.ndim != 4:
        raise ValueError(f"Expected [B,M,T,D] predictions, got {tuple(pred.shape)}")
    B, M, T, D = pred.shape
    if not 1 <= k <= M:
        raise ValueError(f"Invalid beam width {k} for M={M}")
    # Exclude context frame 0; it is the same for all alternatives.
    feat = pred[:, :, 1:].to(torch.float32).reshape(B, M, -1)
    feat = F.normalize(feat, p=2, dim=-1, eps=1e-8)
    # Cosine distances; no privileged XY or true success labels.
    dist = (1.0 - torch.bmm(feat, feat.transpose(1, 2))).clamp_min(0)
    chosen = torch.zeros((B, k), dtype=torch.long, device=pred.device)
    taken = torch.zeros((B, M), dtype=torch.bool, device=pred.device)
    taken[:, 0] = True
    row = torch.arange(B, device=pred.device)
    for j in range(1, k):
        to_selected = dist.gather(
            2, chosen[:, :j].unsqueeze(1).expand(-1, M, -1)
        ).amin(dim=2)
        to_selected = to_selected.masked_fill(taken, -float("inf"))
        next_idx = to_selected.argmax(dim=1)
        chosen[:, j] = next_idx
        taken[row, next_idx] = True
    return chosen


def select_ranked(
    pool_indices: torch.Tensor,
    feasibility_costs: torch.Tensor,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One choice per environment; ties/nonfinite probes keep original L3 plan.

    pool_indices: [B,K] positions 0..M-1 in cost-sorted Top-M pool.
    feasibility_costs: [B,K] minimal *original L2 cost* after probe GD.
    Score = 0.5*(L3 pool rank / (M-1)) + 0.5*(L2 beam rank / (K-1)).
    """
    B, K = pool_indices.shape
    if feasibility_costs.shape != (B, K):
        raise ValueError("Incompatible L3/L2 candidate axes")
    if K < 1:
        raise ValueError("Empty beam")
    if K == 1:
        return pool_indices.new_zeros(B), torch.zeros_like(feasibility_costs)
    if top_m < 2:
        raise ValueError("top_m must be >= 2 for beam selection")
    valid_row = torch.isfinite(feasibility_costs).all(dim=1)
    safe_costs = torch.nan_to_num(
        feasibility_costs, nan=float("inf"), posinf=float("inf"),
        neginf=float("inf")
    )
    # Stable sort => deterministic ties.
    sort = torch.argsort(safe_costs, dim=1, stable=True)
    l2_ranks = torch.argsort(sort, dim=1, stable=True).to(torch.float32)
    r3 = pool_indices.to(torch.float32) / (top_m - 1)
    r2 = l2_ranks / (K - 1)
    scores = 0.5 * r3 + 0.5 * r2
    winner = scores.argmin(dim=1)
    base_score = scores[:, 0]
    win_score = scores.gather(1, winner[:, None]).squeeze(1)
    # If the predicted winner does not strictly beat the original, keep it.
    accept = valid_row & (win_score < base_score - 1e-7)
    winner = torch.where(accept, winner, torch.zeros_like(winner))
    return winner, scores


class CrossLevelBeam:
    def __init__(self, config: CLFBConfig):
        self.cfg = config
        self.last_stats: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def _top_m_rollouts(
        self, l3_solver: Any, l3_info: dict, actions_cpu: torch.Tensor
    ) -> torch.Tensor:
        """Roll out the *original optimized* Top-M L3 actions together."""
        B, M, H, D = actions_cpu.shape
        expanded = l3_solver._expand_info_dict_for_samples(
            l3_info, start_idx=0, end_idx=B, num_samples=M
        )
        predictions = l3_solver.model.rollout(
            expanded, actions_cpu.to(l3_solver.device)
        )["predicted_embed_0"]
        if predictions.ndim != 4 or predictions.shape[:2] != (B, M):
            raise ValueError(
                "L3 rollout must return [B,M,T+1,D], got "
                f"{tuple(predictions.shape)}"
            )
        return predictions.detach().cpu()

    @staticmethod
    def _build_l2_info(
        l2_base_info: dict,
        l2_observation: torch.Tensor,
        l2_cfg: Any,
        upper_num_subgoals: int,
        l3_goals: torch.Tensor,
        l3_actions: torch.Tensor,
        true_l3_goal: torch.Tensor,
        upper_anchored: bool = True,
    ) -> dict:
        """Same goal / action / anchoring semantics as HierarchicalSolver."""
        B = l3_goals.shape[0]
        first_pred_idx = 1 if l3_goals.shape[1] > 1 else 0
        first_action_idx = max(0, first_pred_idx - 1)
        nsub = int(upper_num_subgoals)
        includes_last = first_pred_idx + nsub >= l3_goals.shape[1]
        goal = l3_goals[:, first_pred_idx:first_pred_idx+nsub].clone()
        target_action = l3_actions[
            :, first_action_idx:first_action_idx+nsub
        ].clone()
        if goal.shape[1] != nsub or target_action.shape[1] != nsub:
            raise ValueError("Insufficient L3 subgoal/target-action frames")
        if goal.shape[0] != B:
            raise ValueError("Batch size mismatch")
        if includes_last and upper_anchored:
            goal[:, -1] = true_l3_goal.squeeze(1).to(goal.device)
        info = dict(l2_base_info)
        info["embed_0"] = l2_observation
        info["action_cost_weight"] = float(l2_cfg.action_cost_weight)
        info["goal_embed_0"] = goal
        info["goal_action"] = target_action
        info["goal_embed_0_in_upper_space"] = True
        return info

    def _probe_min_cost(
        self,
        l2_solver: Any,
        l2_info: dict,
        horizon: int,
        call_idx: int,
    ) -> torch.Tensor:
        """Min reachable L2 cost after cheap isolated gradient optimization.

        Shares the frozen World Model, but never calls live_l2.solve():
          - live L2 .init, .torch_gen, .solve_records stay unchanged;
          - torch.autograd.grad(..., action) does not accumulate parameter grads;
          - a local torch.Generator avoids changing default/global RNG state.
        The result is a model-based *feasibility proxy*, NOT a success probability.
        """
        B = int(l2_info["embed_0"].shape[0])
        device = torch.device(l2_solver.device)
        S = self.cfg.probe_samples
        D = int(l2_solver.action_dim)
        micro = min(B, int(l2_solver.batch_size or B))
        out: list[torch.Tensor] = []
        stats = l2_solver._get_action_prior_stats()
        if stats is None:
            mean = torch.zeros(D, device=device)
            std = torch.ones(D, device=device)
        else:
            mean, std = stats
            mean = mean.detach().to(device)
            std = std.detach().to(device)
        std = std * float(l2_solver.var_scale)
        bounds = l2_solver._get_action_clip_bounds()
        cost_last_n = max(1, int(l2_solver._config.cost_last_n))
        inter_weight = float(l2_solver._config.intermediate_cost_weight)

        for start in range(0, B, micro):
            stop = min(start + micro, B)
            bs = stop - start
            expanded = l2_solver._expand_info_dict_for_samples(
                l2_info, start_idx=start, end_idx=stop, num_samples=S
            )
            gen = torch.Generator(device=device).manual_seed(
                int(self.cfg.seed) + int(call_idx) * 10007 + start * 97
            )
            # Keep common random numbers across all beam alternatives:
            # each beam probe resets the same generator for comparable scores.
            init = mean.reshape(1, 1, 1, D).expand(
                bs, S, horizon, D
            ).clone()
            if S > 1:
                init[:, 1:] += torch.randn(
                    (bs, S - 1, horizon, D),
                    device=device,
                    generator=gen,
                    dtype=init.dtype,
                ) * std.reshape(1, 1, 1, D)
            action = init.detach().requires_grad_(True)
            optimizer = l2_solver.optimizer_cls(
                [action], **l2_solver.optimizer_kwargs
            )
            with torch.enable_grad():
                for _ in range(self.cfg.probe_steps):
                    info = dict(expanded)
                    info["cost_last_n"] = cost_last_n
                    info["intermediate_cost_weight"] = inter_weight
                    costs = l2_solver.model.get_cost(info, action)
                    if costs.shape != (bs, S):
                        raise ValueError(f"L2 probe cost shape: {costs.shape}")
                    grad = torch.autograd.grad(costs.sum(), action)[0]
                    optimizer.zero_grad(set_to_none=True)
                    action.grad = grad
                    optimizer.step()
                    if bounds is not None:
                        lo, hi = bounds
                        with torch.no_grad():
                            action.clamp_(
                                min=lo.view(1, 1, 1, D),
                                max=hi.view(1, 1, 1, D),
                            )
            with torch.no_grad():
                info = dict(expanded)
                info["cost_last_n"] = cost_last_n
                info["intermediate_cost_weight"] = inter_weight
                final = l2_solver.model.get_cost(info, action)
                out.append(final.min(dim=1).values.detach().cpu())
        return torch.cat(out, dim=0)

    def select(
        self,
        *,
        level3_solver: Any,
        level2_solver: Any,
        level3_info: dict,
        level2_base_info: dict,
        level2_observation: torch.Tensor,
        level2_cfg: Any,
        level3_num_subgoals: int,
        level3_goal: torch.Tensor,
        level2_horizon: int,
        call_idx: int,
        result: dict,
        steps_taken: int = 0,
    ) -> dict:
        """Replace winning L3 action AND predicted latent (never only one).

        Disabled or width 1 => identical original result object, no extra work.
        Requires level3.solve(return_candidate_pool=True) when enabled.
        """
        if not self.cfg.enabled or self.cfg.beam_width == 1:
            return result
        if level2_horizon <= 1:
            return result
        pool = result.get("candidate_pool")
        if pool is None:
            raise RuntimeError("L3 solver did not return candidate_pool")
        raw_actions = pool["actions"]  # [B,M,H,D] on CPU
        raw_costs = pool["costs"]  # [B,M], ascending
        if not torch.isfinite(raw_costs).all():
            raise ValueError("Nonfinite L3 candidate costs")
        B, M, H3, D3 = raw_actions.shape
        if M < self.cfg.beam_width:
            raise ValueError("Not enough optimized L3 candidates")
        if not torch.equal(raw_actions[:, 0], result["actions"]):
            raise AssertionError("Top-1 candidate must equal original L3 winner")
        t0 = time.perf_counter()
        all_pred = self._top_m_rollouts(
            level3_solver, level3_info, raw_actions
        )
        indices = diverse_beam_indices(all_pred, self.cfg.beam_width)
        b_index = torch.arange(B)[:, None]
        beam_actions = raw_actions[b_index, indices]  # B,K,H3,D3
        beam_pred = all_pred[b_index, indices]  # B,K,H3+1,D
        # Audit beam collapse: different action IDs can predict almost
        # identical trajectories. Log it; do not silently invent diversity.
        flat = beam_pred[:, :, 1:].float().reshape(
            B, self.cfg.beam_width, -1
        )
        norm = F.normalize(flat, p=2, dim=-1, eps=1e-8)
        pair_dist = (1.0 - torch.bmm(
            norm, norm.transpose(1, 2)
        )).clamp_min(0)
        diag = torch.eye(
            self.cfg.beam_width, dtype=torch.bool
        ).unsqueeze(0)
        min_dist = pair_dist.masked_fill(diag, float("inf")).amin(
            dim=(1, 2)
        )
        l2_probe_costs = []
        for j in range(self.cfg.beam_width):
            l2_info = self._build_l2_info(
                l2_base_info=level2_base_info,
                l2_observation=level2_observation,
                l2_cfg=level2_cfg,
                upper_num_subgoals=level3_num_subgoals,
                l3_goals=beam_pred[:, j],
                l3_actions=beam_actions[:, j],
                true_l3_goal=level3_goal,
            )
            probe = self._probe_min_cost(
                level2_solver, l2_info,
                horizon=level2_horizon,
                call_idx=call_idx,
            )
            l2_probe_costs.append(probe)
        l2_costs = torch.stack(l2_probe_costs, dim=1)  # [B,K]
        pos = indices  # indices are positions within cost-sorted Top-M
        winner, scores = select_ranked(pos, l2_costs, top_m=M)
        if self.cfg.force_original:
            # Stage-A gate: probes still ran, but the original winner is kept.
            winner = torch.zeros_like(winner)

        # Keep original predictions byte-for-byte for non-switched envs.
        actions = result["actions"].clone()
        predicted = result["predictions"]["predicted_embed_0"].clone()
        for b, j in enumerate(winner.tolist()):
            if j != 0:
                actions[b] = beam_actions[b, j]
                predicted[b] = beam_pred[b, j]
        out = dict(result)
        out["actions"] = actions
        out["predictions"] = {
            **result["predictions"],
            "predicted_embed_0": predicted,
        }
        elapsed = time.perf_counter() - t0
        self.last_stats = [
            {
                "call_idx": int(call_idx),
                "steps_taken": int(steps_taken),
                "env_idx": int(b),
                "forced_original": bool(self.cfg.force_original),
                "selected_beam_index": int(winner[b]),
                "selected_pool_index": int(indices[b, winner[b]]),
                "switched": bool(winner[b] != 0),
                "l3_costs": raw_costs[b, indices[b]].tolist(),
                "l2_probe_costs": l2_costs[b].tolist(),
                "min_beam_latent_distance": float(min_dist[b]),
                "unique_action_plans": int(
                    torch.unique(
                        beam_actions[b].reshape(
                            self.cfg.beam_width, -1
                        ), dim=0
                    ).shape[0]
                ),
                "scores": scores[b].tolist(),
                "probe_and_select_wall_s": elapsed,
            }
            for b in range(B)
        ]
        self.records.extend(self.last_stats)
        return out
