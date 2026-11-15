"""HiDe-inspired task/control cost separation for frozen H-JEPA AntMaze L3.

V1 is NOT a reproduction of original HiDe's MVProp or joint actor-critic.
- Frozen JEPA dynamics & full latent subgoal remain unchanged.
- Train a light XY readout from *live-domain* train state L3 embeddings.
- Add a differentiable navigation cost only to active L3 and L2-to-L3 plans.
- L1 and original JEPA criterion/action cost remain unchanged.

Strict contracts are intentional: a silent shape broadcast would invalidate
the experiment. The navigation head sees NO privileged test-time XY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import nn


class NavigationProjector(nn.Module):
    """L3 [*,D] -> inferred world-XY [*,2] in AntMaze coordinates.

    Optimization occurs in TRAIN-normalized XY coordinates inside the MLP.
    Output is in physical XY units to allow interpretable offline audits.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_dim: int = 128,
        xy_mean: torch.Tensor | None = None,
        xy_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1:
            raise ValueError("Invalid projector dimensions")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        mean = torch.zeros(2, dtype=torch.float32) if xy_mean is None else torch.as_tensor(xy_mean, dtype=torch.float32)
        std = torch.ones(2, dtype=torch.float32) if xy_std is None else torch.as_tensor(xy_std, dtype=torch.float32)
        if mean.shape != (2,) or std.shape != (2,) or bool(torch.any(std <= 0)):
            raise ValueError("XY mean/std must be vectors of length 2, std positive")
        self.register_buffer("xy_mean", mean.clone())
        self.register_buffer("xy_std", std.clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Projector expected D={self.latent_dim}, got {z.shape[-1]}")
        normalized_xy = self.net(z.float())
        return normalized_xy * self.xy_std + self.xy_mean


@dataclass(frozen=True)
class CostWeights:
    alpha_l3: float = 0.25
    alpha_l2: float = 0.25
    scale_l3: float = 1.0
    scale_l2: float = 1.0

    def validate(self) -> None:
        for name, value in vars(self).items():
            valid = (0 <= value <= 1.0) if name.startswith("alpha") else (0 < value <= 1e6)
            if not valid:
                raise ValueError(f"Invalid {name}={value}")


def navigation_match_cost(
    projector: NavigationProjector,
    predicted_goal_space: torch.Tensor,
    target_goal_space: torch.Tensor,
    *,
    cost_last_n: int,
) -> torch.Tensor:
    """Exact suffix matching of H-JEPA criterion's target sequence.

    predicted_goal_space: [B,N,T,D] AFTER original LevelCostModel projection
    target_goal_space:    [B,N,G,D] AFTER original GD sample expansion

    Return [B,N] physical XY squared cost. No detach of predicted states:
    autograd must propagate through frozen projector AND original predictor.
    """
    if predicted_goal_space.ndim != 4 or target_goal_space.ndim != 4:
        raise ValueError("Expected [batch,samples,time,latent] on both sides")
    if (predicted_goal_space.shape[:2] != target_goal_space.shape[:2]
            or predicted_goal_space.shape[-1] != target_goal_space.shape[-1]
            or predicted_goal_space.shape[-1] != projector.latent_dim):
        raise ValueError(
            f"Pred/goal dimensions mismatch: {tuple(predicted_goal_space.shape)} vs "
            f"{tuple(target_goal_space.shape)}"
        )
    if cost_last_n < 1 or predicted_goal_space.shape[2] < 1 or target_goal_space.shape[2] < 1:
        raise ValueError("Invalid horizon/cost_last_n")
    k = min(int(cost_last_n), predicted_goal_space.shape[2], target_goal_space.shape[2])
    # Goal encoded from a separate observation or an upper-level plan, and
    # is not optimized during the lower-level GD solve.
    predicted_xy = projector(predicted_goal_space[:, :, -k:, :])
    with torch.no_grad():
        target_xy = projector(target_goal_space[:, :, -k:, :].detach())
    cost = (predicted_xy - target_xy).square().sum(dim=(-1, -2))
    if cost.shape != predicted_goal_space.shape[:2]:
        raise AssertionError("Wrong navigation cost reduction")
    return cost


class NavigationCostAdapter(nn.Module):
    """Add supervised task-space navigation penalty to original JEPA cost.

    Intended call site: AFTER LevelCostModel.criterion + action cost. This
    module is instantiated ONLY on enabled runs. It is not called at level 1.
    """

    def __init__(
        self,
        projector: NavigationProjector,
        *,
        level: int,
        alpha: float,
        scale: float,
    ):
        super().__init__()
        if level not in (2, 3):
            raise ValueError("Only L3 and L2->L3 can use the navigation cost")
        if not (0 < alpha <= 1.0 and 0 < scale <= 1e6):
            raise ValueError("Invalid frozen alpha/scale")
        self.projector = projector
        self.level = int(level)
        self.alpha = float(alpha)
        self.scale = float(scale)

    def forward(self, base_cost: torch.Tensor,
                predicted_goal_space: torch.Tensor,
                info_dict: dict[str, Any]) -> torch.Tensor:
        if not info_dict.get("_hide_nav_active", False):
            return base_cost  # truly no-op; no extra computations
        if base_cost.ndim != 2:
            raise ValueError("GD expects base_cost [B,N]")
        goal_level = int(info_dict.get("goal_embed_0_level", self.level + 1))
        if goal_level != 3:
            raise ValueError(f"HiDe active at goal level={goal_level}, expected L3")
        if not bool(info_dict.get("goal_embed_0_in_upper_space", self.level < 3)) and self.level == 2:
            raise ValueError("L2 navigation term requires L3 projected predictions")
        if self.level == 3 and bool(info_dict.get("goal_embed_0_in_upper_space", False)):
            raise ValueError("L3 navigation must use L3 goal space")
        goal = info_dict["goal_embed_0"]
        nav_cost = navigation_match_cost(
            self.projector, predicted_goal_space, goal,
            cost_last_n=int(info_dict.get("cost_last_n", 1))
        )
        if nav_cost.shape != base_cost.shape:
            raise ValueError("Navigation loss shape does not match original cost")
        return base_cost + (self.alpha * self.scale) * nav_cost


def prepare_frozen_projector(head: NavigationProjector, device: torch.device | str) -> NavigationProjector:
    """Freeze PARAMETERS but retain gradient to inputs through the head."""
    head.to(device)
    head.eval()
    head.requires_grad_(False)
    return head
