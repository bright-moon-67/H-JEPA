"""RAGC: post-solve, trust-region correction of frozen H-JEPA level-1 actions.

This module is self-contained, requires only PyTorch, and DOES NOT change
the pretrained H-JEPA model. The adapter is wired in HierarchicalSolver.solve
after original level-1 GradientSolver.solve returns.

Action tokens: [B,H,10] standardized training actions (5 env steps x 2D).
Start/goal: [B,192] L1 and goal-level embeddings, kept distinct.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RAGCConfig:
    state_dim: int = 192
    goal_dim: int = 192
    token_dim: int = 10
    max_horizon: int = 2
    hidden_dim: int = 256
    refinement_steps: int = 3
    step_size: float = 0.10
    advantage_weight: float = 0.30
    trust_weight: float = 0.10
    trust_radius: float = 0.40
    max_original_cost_regression: float = 0.10
    min_advantage_improvement: float = 0.005
    apply_when_horizon_one: bool = False  # V0: no H1 continuation labels


class RealizedAdvantageHead(nn.Module):
    """Lower output means better expected task outcome.

    The goal can reside in the upper-level latent space. We therefore
    CONCATENATE the state and goal; we never subtract embeddings across levels.
    The final two features are goal-is-upper and horizon length.
    """
    def __init__(self, cfg: RAGCConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.state_dim + cfg.goal_dim + cfg.max_horizon * cfg.token_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        z0: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
        goal_is_upper: torch.Tensor | bool,
    ) -> torch.Tensor:
        if z0.ndim != 2 or z0.shape[-1] != self.cfg.state_dim:
            raise ValueError(f"z0 expected [B,{self.cfg.state_dim}], got {z0.shape}")
        b = z0.shape[0]
        if goal.shape != (b, self.cfg.goal_dim):
            raise ValueError(f"goal expected [{b},{self.cfg.goal_dim}], got {goal.shape}")
        if actions.ndim != 3 or actions.shape[0] != b or actions.shape[2] != self.cfg.token_dim:
            raise ValueError(f"actions expected [B,H,{self.cfg.token_dim}], got {actions.shape}")
        h = actions.shape[1]
        if not 1 <= h <= self.cfg.max_horizon:
            raise ValueError(f"need 1 <= H <= {self.cfg.max_horizon}; got {h}")
        a = F.pad(actions, (0, 0, 0, self.cfg.max_horizon - h))
        if isinstance(goal_is_upper, bool):
            upper = torch.full((b, 1), float(goal_is_upper), device=z0.device, dtype=z0.dtype)
        else:
            upper = torch.as_tensor(goal_is_upper, device=z0.device, dtype=z0.dtype).reshape(b, 1)
        h_feature = torch.full((b, 1), h / self.cfg.max_horizon, dtype=z0.dtype, device=z0.device)
        x = torch.cat(
            [z0, goal, a.flatten(1), upper, h_feature],
            dim=1,
        )
        return self.net(x).squeeze(-1)


def paired_realized_advantage_loss(
    head: RealizedAdvantageHead,
    z0: torch.Tensor,
    goal: torch.Tensor,
    plus_actions: torch.Tensor,
    minus_actions: torch.Tensor,
    plus_cost: torch.Tensor,
    minus_cost: torch.Tensor,
    goal_is_upper: torch.Tensor,
    *,
    value_weight: float = 0.25,
    rank_weight: float = 0.10,
    rank_temperature: float = 0.20,
    epsilon: torch.Tensor | None = None,
    gradient_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise PAIRWISE REAL TASK COST differences, not latent MSE.

    plus_cost/minus_cost must come from actual simulator outcomes (preferably
    candidate + frozen-baseline continuation), NEVER from model-predicted cost.
    """
    qp = head(z0, goal, plus_actions, goal_is_upper)
    qm = head(z0, goal, minus_actions, goal_is_upper)
    pc, mc = plus_cost.reshape(-1).float(), minus_cost.reshape(-1).float()
    target_diff = pc - mc
    predicted_diff = qp - qm
    difference_loss = F.smooth_l1_loss(predicted_diff, target_diff)
    value_loss = 0.5 * (F.smooth_l1_loss(qp, pc) + F.smooth_l1_loss(qm, mc))
    # plus is better if real plus_cost < real minus_cost.
    decisive = target_diff.abs() > 1e-4
    if bool(decisive.any()):
        correct_direction = target_diff[decisive].sign() * predicted_diff[decisive]
        rank_loss = F.softplus(-correct_direction / rank_temperature).mean()
    else:
        rank_loss = difference_loss * 0.0
    # Symmetric probes A +/- eps*u give a REAL directional finite difference.
    # Only use it when eps and exact symmetry were verified in the data layer.
    if epsilon is not None:
        eps = torch.as_tensor(epsilon, device=qp.device, dtype=qp.dtype).reshape(-1)
        if bool((eps < 1e-4).any()):
            raise ValueError("epsilon must be positive for finite-difference supervision")
        gradient_loss = F.smooth_l1_loss(
            predicted_diff / (2.0 * eps), target_diff / (2.0 * eps)
        )
    else:
        gradient_loss = difference_loss * 0.0
    loss = (
        difference_loss
        + value_weight * value_loss
        + rank_weight * rank_loss
        + gradient_weight * gradient_loss
    )
    pair_acc = (
        ((predicted_diff[decisive] * target_diff[decisive]) > 0).float().mean()
        if bool(decisive.any()) else torch.tensor(float("nan"), device=z0.device)
    )
    return loss, {
        "pair_loss": difference_loss.detach(),
        "value_loss": value_loss.detach(),
        "rank_loss": rank_loss.detach(),
        "gradient_loss": gradient_loss.detach(),
        "pair_accuracy": pair_acc.detach(),
    }


def latent_context_from_level_info(info: dict, device: torch.device):
    """Unexpanded Level-1 solver info: embed_0 [B,T,D], goal_embed_0 [B,G,D]."""
    z0 = info["embed_0"].to(device).detach()
    goal = info["goal_embed_0"].to(device).detach()
    if z0.ndim != 3 or goal.ndim != 3:
        raise ValueError(f"Expected unsampled [B,T,D] info, got z0={z0.shape}, goal={goal.shape}")
    if z0.shape[0] != goal.shape[0]:
        raise ValueError("state/goal batch mismatch")
    return z0[:, -1, :], goal[:, -1, :], bool(info.get("goal_embed_0_in_upper_space", False))


class RAGCRefiner:
    """A bounded, optional POST-SOLVE action correction.

    No existing GradientSolver, LevelCostModel, or upper hierarchy is replaced.
    If disabled / weight=0 / no certified benefit, actions are unchanged.

    The score head is frozen during evaluation; autograd differentiates
    through actions, NOT through training of the head.
    """
    def __init__(
        self, head: RealizedAdvantageHead, config: RAGCConfig,
        action_mean: torch.Tensor, action_std: torch.Tensor,
    ):
        self.head = head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.cfg = config
        self.mean = torch.as_tensor(action_mean, dtype=torch.float32).reshape(2)
        self.std = torch.as_tensor(action_std, dtype=torch.float32).reshape(2)
        if not bool((self.std > 0).all()):
            raise ValueError("Invalid action normalization std")
        self.last_stats: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self.cfg.advantage_weight > 0 and self.cfg.refinement_steps > 0

    @staticmethod
    def _broadcast_info(solver, level_info: dict, n: int):
        d = solver._expand_info_dict_for_samples(
            level_info, start_idx=0, end_idx=n, num_samples=1,
        )
        d["cost_last_n"] = int(solver._config.cost_last_n)
        d["intermediate_cost_weight"] = float(solver._config.intermediate_cost_weight)
        return d

    def _limit(self, candidate: torch.Tensor, base: torch.Tensor):
        # First trust-ball projection in standardized action-token coordinates.
        radius = float(self.cfg.trust_radius)
        candidate = torch.max(torch.min(candidate, base + radius), base - radius)
        # Then enforce actual env-action box; do not silently use an unbounded action.
        mean = self.mean.to(device=candidate.device, dtype=candidate.dtype)
        std = self.std.to(device=candidate.device, dtype=candidate.dtype)
        low = (-1.0 - mean) / std
        high = (1.0 - mean) / std
        x = candidate.reshape(*candidate.shape[:-1], 5, 2)
        x = torch.max(torch.min(x, high), low)
        return x.reshape_as(candidate)

    @staticmethod
    def _scoring_features(level_info, base, device):
        z, g, upper = latent_context_from_level_info(level_info, device)
        if z.shape[0] != base.shape[0]:
            raise ValueError("batch size mismatch")
        return z, g, upper

    def refine(self, *, solver, level_info: dict, original_actions: torch.Tensor) -> torch.Tensor:
        """Return CPU [B,H,10], exactly as original GD solver returns.

        solver is the repo's already CONFIGURED Level-1 GradientSolver.
        The unchanged original cost remains authoritative for safety gating.
        """
        if not self.enabled:
            self.last_stats = {"enabled": False}
            return original_actions
        if original_actions.ndim != 3 or original_actions.shape[-1] != 10:
            raise ValueError(f"Expected original [B,H,10], got {original_actions.shape}")
        if original_actions.shape[1] == 1 and not self.cfg.apply_when_horizon_one:
            self.last_stats = {"enabled": False, "reason": "horizon_one_untrained"}
            return original_actions
        model = solver.model  # Existing LevelCostModel, preserving action_cost_weight=0.2
        device = next(self.head.parameters()).device
        base = original_actions.detach().to(device=device, dtype=torch.float32)
        b = base.shape[0]
        z, goal, upper = self._scoring_features(level_info, base, device)
        score_info = self._broadcast_info(solver, level_info, b)

        def orig_cost(actions):
            costs = model.get_cost(score_info.copy(), actions.unsqueeze(1))
            if costs.shape != (b, 1):
                raise AssertionError(f"Expected original cost [B,1], got {costs.shape}")
            return costs[:, 0]

        with torch.no_grad():
            # The original policy may emit actions outside the declared Box.
            # Our refiner must not silently replace such a baseline with a
            # clipped one and then attribute the change to RAGC.
            mean = self.mean.to(device=base.device, dtype=base.dtype)
            std = self.std.to(device=base.device, dtype=base.dtype)
            base_raw = base.reshape(b, -1, 5, 2) * std + mean
            valid_original = (
                (base_raw >= -1.0) & (base_raw <= 1.0)
            ).reshape(b, -1).all(dim=1)
            original_cost = orig_cost(base).detach()
            original_advantage_cost = self.head(z, goal, base, upper).detach()
        cost_scale = original_cost.abs().clamp_min(1.0)
        best = base.detach()
        with torch.enable_grad():
            current = base.detach()
            for _ in range(self.cfg.refinement_steps):
                current = current.detach().requires_grad_(True)
                j = orig_cost(current)
                q = self.head(z, goal, current, upper)
                d = (current - base).square().mean(dim=(1, 2))
                objective = (
                    (j - original_cost) / cost_scale
                    + self.cfg.advantage_weight * (q - original_advantage_cost)
                    + self.cfg.trust_weight * d
                )
                grad = torch.autograd.grad(objective.sum(), current)[0]
                if not bool(torch.isfinite(grad).all()):
                    self.last_stats = {"enabled": True, "nonfinite_gradient": True}
                    return original_actions
                norm = grad.flatten(1).norm(dim=-1).clamp_min(1e-8).reshape(b, 1, 1)
                with torch.no_grad():
                    current = self._limit(current - self.cfg.step_size * grad / norm, base)
            best = current.detach()
        with torch.no_grad():
            new_cost = orig_cost(best)
            new_advantage_cost = self.head(z, goal, best, upper)
            predicted_gain = original_advantage_cost - new_advantage_cost
            ok = (
                valid_original
                & torch.isfinite(new_cost)
                & torch.isfinite(new_advantage_cost)
                & (predicted_gain >= self.cfg.min_advantage_improvement)
                & (
                    new_cost <= original_cost
                    + self.cfg.max_original_cost_regression * cost_scale
                )
            )
            selected = torch.where(ok[:, None, None], best, base)
        self.last_stats = {
            "enabled": True,
            "accepted": int(ok.sum().item()),
            "original_actions_inside_box": int(valid_original.sum().item()),
            "batch": b,
            "mean_predicted_gain": float(predicted_gain.mean().item()),
            "mean_original_cost_change": float((new_cost - original_cost).mean().item()),
        }
        return selected.detach().cpu()
