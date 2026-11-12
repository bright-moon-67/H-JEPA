"""Feedback-N executor tests (no model, no simulator).

The point of `execution_receding_horizon` is to decouple
  * how long a plan is EXECUTED   (policy buffer)
from
  * what problem the solver PLANS  (receding_horizon -> horizon schedules)
so that "more frequent real feedback" can be tested without changing the
planning problem.

Checks:
  1. None == explicit value -> strict no-op
  2. execution=1 -> a new plan is requested every 5 env steps (action_block=5)
  3. execution=2 -> every 10 env steps (original behaviour)
  4. the solver still receives the ORIGINAL absolute `steps_taken`
  5. the first 5 executed actions are identical between 1 and 2
  6. env-action dimensions and the action inverse-transform are unchanged
  7. invalid horizons are rejected
"""
import numpy as np
import pytest
import torch

from stable_worldmodel.policy import PlanConfig, WorldModelPolicy


class CountingSolver:
    """Returns a distinct-but-deterministic plan per call."""

    def __init__(self, action_block=5, action_dim=2):
        self.calls: list[int] = []
        self.budgets: list[int | None] = []
        self.action_block = action_block
        self.action_dim = action_dim

    def configure(self, **kwargs):  # pragma: no cover - not used here
        self.configured = kwargs

    def __call__(self, info_dict, steps_taken=None, eval_budget=None):
        self.calls.append(int(steps_taken))
        self.budgets.append(eval_budget)
        # 2 tokens x (action_block * action_dim) = [1, 2, 10]
        k = len(self.calls)
        tok = torch.full((1, 2, self.action_block * self.action_dim),
                         float(k), dtype=torch.float32)
        return {"actions": tok}


class FakeEnv:
    def __init__(self):
        self.num_envs = 1
        self.action_space = type("AS", (), {
            "shape": (1, 2),
        })()


class ScaleProcess:
    """Mimics the action normalizer: transform * s, inverse_transform / s."""

    def __init__(self, s=10.0):
        self.s = s
        self.inverse_calls = 0

    def transform(self, x):
        return x / self.s

    def inverse_transform(self, x):
        self.inverse_calls += 1
        return x * self.s


def make_policy(execution=None, process=None):
    cfg = PlanConfig(horizon=2, receding_horizon=2, action_block=5, action_dim=2)
    solver = CountingSolver()
    policy = WorldModelPolicy(
        solver=solver,
        config=cfg,
        process=process or {},
        transform={},
        eval_budget=90,
        execution_receding_horizon=execution,
    )
    policy.env = FakeEnv()
    policy.reset()
    return policy, solver


def info():
    return {
        "pixels": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
        "goal": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
    }


def run(policy, n_steps):
    return np.stack([policy.get_action(info()) for _ in range(n_steps)])


# --------------------------------------------------------------------------- #
# 1. no-op
# --------------------------------------------------------------------------- #
def test_none_and_explicit_two_are_identical():
    p_none, s_none = make_policy(execution=None)
    p_two, s_two = make_policy(execution=2)
    a = run(p_none, 25)
    b = run(p_two, 25)
    assert s_none.calls == s_two.calls == [0, 10, 20]
    assert np.array_equal(a, b)
    assert p_none.flatten_receding_horizon == p_two.flatten_receding_horizon == 10


# --------------------------------------------------------------------------- #
# 2/3. replanning cadence
# --------------------------------------------------------------------------- #
def test_execution_one_replans_every_five_steps():
    policy, solver = make_policy(execution=1)
    run(policy, 25)
    assert solver.calls == [0, 5, 10, 15, 20]
    assert policy.flatten_receding_horizon == 5


def test_execution_two_replans_every_ten_steps():
    policy, solver = make_policy(execution=2)
    run(policy, 25)
    assert solver.calls == [0, 10, 20]


def test_buffer_maxlen_matches_execution_horizon():
    p1, _ = make_policy(execution=1)
    p2, _ = make_policy(execution=None)
    assert p1._action_buffer.maxlen == 5
    assert p2._action_buffer.maxlen == 10


# --------------------------------------------------------------------------- #
# 4. solver still sees the real absolute step count
# --------------------------------------------------------------------------- #
def test_solver_receives_real_steps_taken_and_budget():
    policy, solver = make_policy(execution=1)
    run(policy, 16)
    assert solver.calls == [0, 5, 10, 15]
    assert set(solver.budgets) == {90}


# --------------------------------------------------------------------------- #
# 5. first plan is shared
# --------------------------------------------------------------------------- #
def test_first_five_actions_are_identical():
    p1, _ = make_policy(execution=1)
    p2, _ = make_policy(execution=2)
    a1 = run(p1, 5)
    a2 = run(p2, 5)
    assert np.array_equal(a1, a2)
    # both are exactly the first token of plan #1 (value 1.0)
    assert np.allclose(a1, 1.0)


def test_second_token_is_not_executed_at_half_horizon_by_default():
    """At step 5 the original policy still reuses token 2 (value 1.0), while
    Feedback-5 asks for a NEW plan (value 2.0)."""
    p1, _ = make_policy(execution=1)
    p2, _ = make_policy(execution=2)
    run(p1, 5)
    run(p2, 5)
    a1_next = p1.get_action(info())  # step 5
    a2_next = p2.get_action(info())  # step 5
    assert np.allclose(a1_next, 2.0)   # fresh plan
    assert np.allclose(a2_next, 1.0)   # still the buffered second token


# --------------------------------------------------------------------------- #
# 6. dims + inverse transform
# --------------------------------------------------------------------------- #
def test_action_shape_and_inverse_transform_unchanged():
    proc = ScaleProcess(10.0)
    p1, _ = make_policy(execution=1, process={"action": proc})
    p2, _ = make_policy(execution=2, process={"action": proc})
    a1 = p1.get_action(info())
    a2 = p2.get_action(info())
    assert a1.shape == a2.shape == (1, 2)
    assert proc.inverse_calls == 2
    assert np.allclose(a1, 10.0) and np.allclose(a2, 10.0)


# --------------------------------------------------------------------------- #
# 7. validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -1, 3, 5])
def test_invalid_execution_horizon_rejected(bad):
    cfg = PlanConfig(horizon=2, receding_horizon=2, action_block=5, action_dim=2)
    with pytest.raises(ValueError):
        WorldModelPolicy(solver=CountingSolver(), config=cfg,
                         process={}, transform={},
                         execution_receding_horizon=bad)
