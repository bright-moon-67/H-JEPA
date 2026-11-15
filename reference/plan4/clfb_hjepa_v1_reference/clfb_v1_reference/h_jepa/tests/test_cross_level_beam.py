"""CPU tests for CLFB research reference (no real checkpoint required)."""
from types import SimpleNamespace
import torch
import pytest

from cross_level_beam import (
    CLFBConfig,
    CrossLevelBeam,
    diverse_beam_indices,
    select_ranked,
)


def test_disabled_and_beam_one_are_identity():
    reference = {"actions": torch.zeros(1, 2, 4)}
    assert CrossLevelBeam(CLFBConfig(enabled=False)).select(
        level3_solver=None, level2_solver=None, level3_info={},
        level2_base_info={}, level2_observation=None, level2_cfg=None,
        level3_num_subgoals=1, level3_goal=None, level2_horizon=2,
        call_idx=0, result=reference
    ) is reference
    assert CrossLevelBeam(CLFBConfig(enabled=True, beam_width=1)).select(
        level3_solver=None, level2_solver=None, level3_info={},
        level2_base_info={}, level2_observation=None, level2_cfg=None,
        level3_num_subgoals=1, level3_goal=None, level2_horizon=2,
        call_idx=0, result=reference
    ) is reference


def test_diversity_keeps_best_and_chooses_distinct_route():
    pred = torch.tensor(
        [[[[0., 0.], [1., 0.]],
          [[0., 0.], [1., 0.]],
          [[0., 0.], [-1., 0.]],
          [[0., 0.], [0., 1.]]]]
    )
    chosen = diverse_beam_indices(pred, 3)
    assert chosen.tolist() == [[0, 2, 3]]


def test_l2_rank_can_improve_cross_level_score():
    pool_pos = torch.tensor([[0, 1, 3, 8]])
    costs = torch.tensor([[10., 2., 3., 4.]])
    winner, scores = select_ranked(pool_pos, costs, top_m=32)
    assert winner.tolist() == [1]
    assert scores[0, 1] < scores[0, 0]


def test_tied_costs_keep_original():
    pool_pos = torch.tensor([[0, 1, 3, 8]])
    winner, _ = select_ranked(
        pool_pos, torch.tensor([[1., 1., 1., 1.]]), top_m=32
    )
    assert winner.tolist() == [0]


def test_invalid_probe_keeps_original():
    pool_pos = torch.tensor([[0, 1, 2, 3]])
    winner, _ = select_ranked(
        pool_pos,
        torch.tensor([[1., 0., float('nan'), 2.]]),
        top_m=32,
    )
    assert winner.tolist() == [0]


def test_l2_subgoals_have_exact_original_time_alignment():
    l2cfg = SimpleNamespace(action_cost_weight=0.1)
    l3_preds = torch.arange(5.0).view(1, 5, 1)
    l3_actions = torch.arange(4.0).view(1, 4, 1)
    l2info = CrossLevelBeam._build_l2_info(
        l2_base_info={"pixels": "input pixels"},
        l2_observation=torch.zeros(1, 1, 2),
        l2_cfg=l2cfg, upper_num_subgoals=1,
        l3_goals=l3_preds, l3_actions=l3_actions,
        true_l3_goal=torch.tensor([[[99.]]]),
    )
    assert float(l2info["goal_embed_0"][0, 0, 0]) == 1.
    assert float(l2info["goal_action"][0, 0, 0]) == 0.
    assert l2info["goal_embed_0_in_upper_space"] is True
    assert l2info["action_cost_weight"] == 0.1
    # When next subgoal is actually the final future state, match real goal.
    l2info_last = CrossLevelBeam._build_l2_info(
        l2_base_info={}, l2_observation=torch.zeros(1, 1, 2),
        l2_cfg=l2cfg, upper_num_subgoals=1,
        l3_goals=l3_preds[:, :2], l3_actions=l3_actions[:, :1],
        true_l3_goal=torch.tensor([[[99.]]]),
    )
    assert float(l2info_last["goal_embed_0"][0, 0, 0]) == 99.


class QuadraticCost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def get_cost(self, info, action):
        target = info["goal_embed_0"].squeeze(-2).unsqueeze(-2)
        return ((action - target) ** 2).mean(dim=(-1, -2)) * self.weight


class FakeL2:
    def __init__(self):
        self.device = "cpu"
        self.action_dim = 2
        self.batch_size = 1
        self.var_scale = 1.0
        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": 0.1}
        self._config = SimpleNamespace(
            cost_last_n=1, intermediate_cost_weight=1.0
        )
        self.model = QuadraticCost()
        self.torch_gen = torch.Generator(device="cpu").manual_seed(7777)
        self.init = torch.zeros(2, 3)
        self.solve_records = []

    def _get_action_prior_stats(self):
        return None

    def _get_action_clip_bounds(self):
        return None

    def _expand_info_dict_for_samples(
        self, info_dict, start_idx, end_idx, num_samples
    ):
        result = {}
        for key, value in info_dict.items():
            if isinstance(value, torch.Tensor):
                v = value[start_idx:end_idx]
                result[key] = v.unsqueeze(1).expand(
                    end_idx - start_idx, num_samples, *v.shape[1:]
                )
            else:
                result[key] = value
        return result


def test_probe_does_not_modify_l2_solver_or_global_rng():
    solver = FakeL2()
    cfg = CLFBConfig(enabled=True, probe_samples=4, probe_steps=7)
    beam = CrossLevelBeam(cfg)
    info = {
        "embed_0": torch.zeros(2, 1, 2),
        "goal_embed_0": torch.full((2, 1, 2), 0.7),
    }
    global_state = torch.random.get_rng_state().clone()
    live_state = solver.torch_gen.get_state().clone()
    old_init = solver.init.clone()
    out1 = beam._probe_min_cost(solver, info, horizon=2, call_idx=0)
    out2 = beam._probe_min_cost(solver, info, horizon=2, call_idx=0)
    assert out1.shape == (2,)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(live_state, solver.torch_gen.get_state())
    assert torch.equal(old_init, solver.init)
    assert solver.solve_records == []
    assert solver.model.weight.grad is None


def test_select_changes_actions_and_latent_together():
    beam = CrossLevelBeam(CLFBConfig(enabled=True, top_m=4, beam_width=3))
    pool_actions = torch.tensor(
        [[[[0., 0.]], [[1., 0.]], [[-1., 0.]], [[0., 1.]]]]
    )
    # For beam order [0,2,1 or 3], j=1 will be selected.
    pool_preds = torch.tensor(
        [[[[0., 0.], [0., 0.]],
          [[0., 0.], [1., 0.]],
          [[0., 0.], [-1., 0.]],
          [[0., 0.], [0., 1.]]]]
    )
    pool = {
        "actions": pool_actions,
        "costs": torch.tensor([[0., 1., 2., 3.]]),
    }
    original = {
        "actions": pool_actions[:, 0].clone(),
        "predictions": {"predicted_embed_0": pool_preds[:, 0].clone()},
        "candidate_pool": pool,
    }
    beam._top_m_rollouts = lambda *args: pool_preds
    calls = iter((torch.tensor([9.]), torch.tensor([0.]), torch.tensor([8.])))
    beam._probe_min_cost = lambda *args, **kwargs: next(calls)
    result = beam.select(
        level3_solver=None, level2_solver=None, level3_info={},
        level2_base_info={}, level2_observation=torch.zeros(1, 1, 2),
        level2_cfg=SimpleNamespace(action_cost_weight=0.0),
        level3_num_subgoals=1, level3_goal=torch.zeros(1, 1, 2),
        level2_horizon=2, call_idx=0, result=original,
    )
    # Whatever the farthest-first index, selected plan is that candidate.
    chosen_pool_index = beam.last_stats[0]["selected_pool_index"]
    assert chosen_pool_index != 0
    torch.testing.assert_close(
        result["actions"], pool_actions[:, chosen_pool_index]
    )
    torch.testing.assert_close(
        result["predictions"]["predicted_embed_0"],
        pool_preds[:, chosen_pool_index],
    )
    # Input dict remains untouched.
    assert torch.equal(original["actions"], pool_actions[:, 0])
