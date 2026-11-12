"""CPU only; does NOT import Gym, H-JEPA, or Pymunk."""
import numpy as np
import pytest
import torch

from ragc import (
    RAGCConfig, RealizedAdvantageHead, RAGCRefiner,
    paired_realized_advantage_loss,
)
from ragc_oracle import antithetic_candidates, raw_actions_from_tokens, terminal_task_cost


def create_batch(b=4, h=2):
    z = torch.randn(b, 192)
    goal = torch.randn(b, 192)
    a = torch.randn(b, h, 10)
    return z, goal, a


def test_head_h1_h2_shapes_and_backward():
    torch.manual_seed(1)
    cfg = RAGCConfig(hidden_dim=32)
    head = RealizedAdvantageHead(cfg)
    for h in (1, 2):
        z, goal, a = create_batch(h=h)
        a.requires_grad_(True)
        q = head(z, goal, a, True)
        assert q.shape == (4,)
        grad = torch.autograd.grad(q.sum(), a)[0]
        assert grad.shape == a.shape
        assert torch.isfinite(grad).all()


def test_pairwise_direction_loss_is_trainable():
    torch.manual_seed(3)
    cfg = RAGCConfig(hidden_dim=32)
    head = RealizedAdvantageHead(cfg)
    opt = torch.optim.Adam(head.parameters(), lr=0.005)
    z, g, a = create_batch(b=8)
    minus = a - 0.4
    target_plus = torch.zeros(8)
    target_minus = torch.ones(8)
    with torch.no_grad():
        before, _ = paired_realized_advantage_loss(
            head, z, g, a, minus, target_plus, target_minus, torch.ones(8)
        )
    for _ in range(100):
        loss, _ = paired_realized_advantage_loss(
            head, z, g, a, minus, target_plus, target_minus, torch.ones(8)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after, _ = paired_realized_advantage_loss(
            head, z, g, a, minus, target_plus, target_minus, torch.ones(8)
        )
    assert after.item() < before.item() * 0.30


def test_antithetic_pair_and_action_conversion():
    rng = np.random.default_rng(4)
    base = np.zeros((2, 10), np.float32)
    choices = antithetic_candidates(base, eps=0.2, directions=3, rng=rng)
    assert len(choices) == 7
    for i in (1, 3, 5):
        np.testing.assert_allclose(choices[i] + choices[i+1], base * 2, atol=1e-6)
    raw = raw_actions_from_tokens(base, mean=[0.0,0.0], std=[0.2,0.2])
    assert raw.shape == (10,2)


def test_task_angle_wrap_and_agent_included():
    g = np.zeros(7)
    g[4] = np.deg2rad(179.0)
    s = g.copy()
    s[4] = np.deg2rad(-179.0)
    assert terminal_task_cost(s, g, True) < 0.02
    s[0] = 80  # Agent far away despite perfect block pose
    assert terminal_task_cost(s, g, False) > 1.0


class ToySolver:
    def __init__(self):
        self.model = ToyCost()

    class Plan:
        cost_last_n = 1
        intermediate_cost_weight = 1.0
    _config = Plan()
    @staticmethod
    def _expand_info_dict_for_samples(info, start_idx, end_idx, num_samples):
        result = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                result[k] = v[start_idx:end_idx].unsqueeze(1).expand(
                    end_idx-start_idx, num_samples, *v.shape[1:]
                )
            else:
                result[k] = v
        return result


class ToyCost:
    def get_cost(self, info, actions):
        # global minimum at token=0; shape [B,S]
        return actions.square().mean(dim=(2,3))


class ConstantQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(()))
    def forward(self, z, goal, actions, upper):
        return self.a * torch.zeros(actions.shape[0], device=actions.device)


def test_disabled_refiner_is_exact_noop():
    cfg = RAGCConfig(refinement_steps=0)
    r = RAGCRefiner(ConstantQ(), cfg, torch.zeros(2), torch.ones(2))
    a = torch.randn(2,2,10)
    out = r.refine(solver=ToySolver(), level_info={}, original_actions=a)
    assert out is a


def test_trust_projection_and_fallback_for_constant_q():
    cfg = RAGCConfig(hidden_dim=32, refinement_steps=3)
    head = RealizedAdvantageHead(cfg)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
    r = RAGCRefiner(head, cfg, torch.zeros(2), torch.ones(2))
    a = torch.zeros(2,2,10)
    info = {
        "embed_0": torch.randn(2,1,192),
        "goal_embed_0": torch.randn(2,1,192),
        "goal_embed_0_in_upper_space": True,
    }
    out = r.refine(solver=ToySolver(), level_info=info, original_actions=a)
    torch.testing.assert_close(out, a, rtol=0, atol=0)
    assert r.last_stats["accepted"] == 0


def test_wrong_horizon_fails_fast():
    model = RealizedAdvantageHead(RAGCConfig())
    with pytest.raises(ValueError):
        model(torch.zeros(1,192), torch.zeros(1,192), torch.zeros(1,3,10), False)


class LinearActionQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, z0, goal, actions, goal_is_upper):
        return -self.scale * actions.mean(dim=(1, 2))


def test_positive_refinement_with_real_gradient_direction():
    cfg = RAGCConfig(
        refinement_steps=4, step_size=0.18,
        advantage_weight=1.0, min_advantage_improvement=0.001,
        max_original_cost_regression=0.30,
    )
    refiner = RAGCRefiner(LinearActionQ(), cfg, torch.zeros(2), torch.ones(2))
    original = torch.zeros(2, 2, 10)
    info = {
        'embed_0': torch.randn(2,1,192),
        'goal_embed_0': torch.randn(2,1,192),
        'goal_embed_0_in_upper_space': True,
    }
    corrected = refiner.refine(solver=ToySolver(), level_info=info, original_actions=original)
    assert corrected.shape == original.shape
    assert bool((corrected > original).any())
    assert float((corrected - original).abs().max()) <= cfg.trust_radius + 1e-5
    assert refiner.last_stats['accepted'] > 0
