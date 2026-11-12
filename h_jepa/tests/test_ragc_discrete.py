"""Tests for RAGC-Discrete (no simulator, no GPU).

Covered:
  1. default no-op (non-first step / unsupported shape / disabled)
  2. margin guard keeps A*
  3. a genuine switch returns one of the four LEGAL candidates
  4. illegal candidates are never selected
  5. original-model cost regression keeps A*
  6. the planner's torch RNG is NOT consumed (candidates use a private RNG)
  7. HierarchicalSolver exposes an opt-in selector hook, defaulting to None
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ragc_discrete import DiscreteConfig, RAGCDiscreteSelector


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class LinearHead(torch.nn.Module):
    """q = -(w * actions).sum() -> lower is better for large positive actions."""

    def __init__(self, w=1.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(w)), requires_grad=False)

    def forward(self, z0, goal, actions, goal_is_upper):
        return -(actions * self.w).sum(dim=(1, 2))


class FakeCostModel:
    def __init__(self, base=0.0, penalty=0.0, original=None):
        self.base, self.penalty, self.original = base, penalty, original

    def get_cost(self, info, actions):
        # actions: [B,1,2,10]; penalise anything that differs from `original`
        b = actions.shape[0]
        if self.original is None:
            return torch.full((b, 1), self.base)
        diff = (actions[:, 0] - self.original).abs().max().item()
        return torch.full((b, 1), self.base + self.penalty * (diff > 1e-9))


class FakeSolver:
    def __init__(self, cost_model):
        self.model = cost_model
        self._config = SimpleNamespace(cost_last_n=1, intermediate_cost_weight=1.0)

    def _expand_info_dict_for_samples(self, level_info, start_idx=0,
                                      end_idx=1, num_samples=1):
        return dict(level_info)


def make_level_info(batch=1, dim=192):
    return {
        "embed_0": torch.zeros(batch, 2, dim),
        "goal_embed_0": torch.zeros(batch, 2, dim),
        "goal_embed_0_in_upper_space": False,
    }


def make_selector(head=None, cfg=None, cost_model=None):
    head = head or LinearHead()
    cfg = cfg or DiscreteConfig(epsilon=0.3, directions=2, margin=0.05, seed=7)
    sel = RAGCDiscreteSelector(head, cfg, np.array([0.0, 0.0]),
                               np.array([1.0, 1.0]))
    solver = FakeSolver(cost_model or FakeCostModel())
    return sel, solver


# --------------------------------------------------------------------------- #
# 1. no-op paths
# --------------------------------------------------------------------------- #
def test_noop_on_non_first_step():
    sel, solver = make_selector()
    a = torch.randn(1, 2, 10)
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=10)
    assert torch.equal(out, a)
    assert sel.stats.reasons.get("not_first_planning_step") == 1


def test_noop_on_unsupported_shape():
    sel, solver = make_selector()
    a = torch.randn(1, 2, 5)
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert torch.equal(out, a)
    assert sel.stats.n_switched == 0


# --------------------------------------------------------------------------- #
# 2. margin guard
# --------------------------------------------------------------------------- #
def test_margin_guard_keeps_original():
    # head returns a CONSTANT -> gain == 0 -> below margin
    class ConstHead(torch.nn.Module):
        def forward(self, z0, goal, actions, goal_is_upper):
            return torch.zeros(actions.shape[0])

    sel, solver = make_selector(head=ConstHead())
    a = torch.zeros(1, 2, 10)
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert torch.equal(out, a)
    assert sel.stats.reasons.get("margin_not_met") == 1
    assert sel.stats.n_switched == 0


# --------------------------------------------------------------------------- #
# 3/4. genuine switch + legality
# --------------------------------------------------------------------------- #
def test_switch_returns_legal_candidate_matching_argmin():
    sel, solver = make_selector()
    a = torch.zeros(1, 2, 10)
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert sel.stats.n_switched == 1
    assert out.shape == a.shape

    # returned action is one of the generated candidates
    rng = np.random.default_rng(sel.cfg.seed + 1)
    from ragc_oracle import antithetic_candidates
    cands = np.stack(antithetic_candidates(
        a[0].numpy(), sel.cfg.epsilon, sel.cfg.directions, rng))
    assert np.allclose(out[0].numpy(), cands[sel.stats.last_choice], atol=1e-6)

    # and it stays inside the environment action box
    raw = (out[0].numpy().reshape(-1, 2) * sel.std.numpy()
           + sel.mean.numpy())
    assert np.abs(raw).max() <= 1.0 + 1e-6


def test_illegal_candidates_are_never_selected():
    # huge std -> every perturbed token maps to |raw action| > 1
    sel = RAGCDiscreteSelector(LinearHead(), DiscreteConfig(seed=3),
                               np.zeros(2), np.full(2, 1e4))
    solver = FakeSolver(FakeCostModel())
    a = torch.zeros(1, 2, 10)
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert torch.equal(out, a)
    assert sel.stats.n_switched == 0


def test_illegal_original_action_is_not_modified():
    sel, solver = make_selector()
    a = torch.full((1, 2, 10), 5.0)  # raw = 5.0 * std -> outside [-1,1]
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert torch.equal(out, a)
    assert sel.stats.reasons.get("original_actions_illegal") == 1


# --------------------------------------------------------------------------- #
# 5. original-model cost protection
# --------------------------------------------------------------------------- #
def test_cost_regression_keeps_original():
    a = torch.zeros(1, 2, 10)
    sel, _ = make_selector(cost_model=FakeCostModel(base=1.0, penalty=100.0,
                                                    original=a[0]))
    solver = FakeSolver(FakeCostModel(base=1.0, penalty=100.0, original=a[0]))
    out = sel.select(solver=solver, level_info=make_level_info(),
                     original_actions=a, steps_taken=0)
    assert torch.equal(out, a)
    assert sel.stats.reasons.get("original_cost_regression") == 1
    assert sel.stats.n_switched == 0


# --------------------------------------------------------------------------- #
# 6. planner RNG isolation
# --------------------------------------------------------------------------- #
def test_planner_torch_rng_untouched():
    sel, solver = make_selector()
    a = torch.zeros(1, 2, 10)
    state = torch.random.get_rng_state().clone()
    sel.select(solver=solver, level_info=make_level_info(),
               original_actions=a, steps_taken=0)
    assert torch.equal(torch.random.get_rng_state(), state)


# --------------------------------------------------------------------------- #
# 7. solver hook
# --------------------------------------------------------------------------- #
def test_hierarchical_solver_selector_hook_defaults_to_none():
    hs = pytest.importorskip("hierarchical_solver")
    s = hs.HierarchicalSolver(level_solvers={1: object()}, level_models=[object()])
    assert s.ragc_selector is None
    sentinel = object()
    s.set_ragc_selector(sentinel)
    assert s.ragc_selector is sentinel
