"""Pairwise, antisymmetric action-response model.

Input:
    z0:    [B, 192]      frozen H-JEPA Level-1 latent
    acts_a, acts_b: [B, 5, 2] raw Push-T environment actions
Output:
    r_ab:  [B, 192]      predicted E(o_A_end)-E(o_B_end)
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponseConfig:
    state_dim: int = 192
    action_dim: int = 2
    horizon: int = 5
    hidden_dim: int = 384


class ActionResponseNet(nn.Module):
    """R(z,A,B) = phi(z,A) - phi(z,B).

    The subtraction enforces:
      R(z,A,A) == 0
      R(z,A,B) == -R(z,B,A)
    by construction, independently of training.
    """

    def __init__(self, cfg: ResponseConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.state_dim + cfg.horizon * cfg.action_dim
        self.phi = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )

    def potential(self, z0: torch.Tensor, actions: torch.Tensor):
        if z0.ndim != 2 or z0.shape[-1] != self.cfg.state_dim:
            raise ValueError(
                f"z0 should be [B,{self.cfg.state_dim}], got {tuple(z0.shape)}"
            )
        required = (z0.shape[0], self.cfg.horizon, self.cfg.action_dim)
        if tuple(actions.shape) != required:
            raise ValueError(f"actions should be {required}, got {tuple(actions.shape)}")
        x = torch.cat([z0, actions.reshape(z0.shape[0], -1)], dim=-1)
        return self.phi(x)

    def forward(self, z0, acts_a, acts_b):
        return self.potential(z0, acts_a) - self.potential(z0, acts_b)


def response_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target mismatch: {pred.shape} vs {target.shape}")
    return F.mse_loss(pred.float(), target.float())
