"""Unit tests for the opt-in AntMaze trace logger.

These tests use fake World / Policy / Solver objects: no MuJoCo, no model,
no checkpoint.  They verify the three invariants that matter for the audit:

1. enabling the tracer does not change the action stream;
2. it consumes no random numbers;
3. task ids are mapped per env and per-task files are disjoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antmaze_trace import AntMazeTrace  # noqa: E402


class FakeCostModel(torch.nn.Module):
    def __init__(self, n_envs, num_samples=4):
        super().__init__()
        self.n_envs = n_envs
        self.num_samples = num_samples
        self.calls = []

    def get_cost(self, info_dict, action_candidates):
        self.calls.append(int(action_candidates.shape[0]))
        base = action_candidates.float().reshape(action_candidates.shape[0], -1).mean(1)
        cost = base[:, None] + torch.arange(
            self.num_samples, dtype=torch.float32
        )[None, :]
        return cost.requires_grad_(True)


class FakeSolver:
    def __init__(self, n_envs, batch_size=1):
        self.n_envs = n_envs
        self.level = 1
        self.horizon = 2
        self.num_samples = 4
        self.n_steps = 3
        self.batch_size = batch_size
        self.solve_records = []
        self.model = FakeCostModel(n_envs, self.num_samples)
        self.solve_count = 0

    def solve(self, info_dict, planning_horizon, steps_taken, eval_budget):
        self.solve_count += 1
        actions = torch.full(
            (self.n_envs, int(planning_horizon), 2),
            float(steps_taken),
            dtype=torch.float32,
        )
        # Simulate GradientSolver: batch loop, n_steps + one final cost call.
        n_iters = []
        for start in range(0, self.n_envs, self.batch_size):
            end = min(start + self.batch_size, self.n_envs)
            batch = actions[start:end]
            for _ in range(self.n_steps):
                self.model.get_cost(info_dict, batch)
            self.model.get_cost(info_dict, batch)
            n_iters.extend([self.n_steps] * (end - start))
        self.solve_records.append(
            {
                "level": self.level,
                "horizon": int(planning_horizon),
                "num_samples": self.num_samples,
                "n_iters": n_iters,
                "wall_clock": 0.01,
                "wall_clock_per_env": [0.01] * self.n_envs,
            }
        )
        pred = actions.mean(dim=-1, keepdim=True).expand(-1, -1, 3).contiguous()
        return {
            "actions": actions,
            "predictions": {"predicted_embed_0": pred},
        }


class FakePolicy:
    def __init__(self, n_envs, action_dim=2):
        self.n_envs = n_envs
        self.action_dim = action_dim
        self.calls = 0
        self.solver = FakeSolver(n_envs)

    def get_action(self, infos):
        self.calls += 1
        return np.full((self.n_envs, self.action_dim), float(self.calls), np.float32)


class FakeWorld:
    def __init__(self, policy, n_envs=2, steps=6):
        self.policy = policy
        self.num_envs = n_envs
        self.steps = steps
        self.t = 0
        self.terminateds = np.zeros(n_envs, dtype=bool)
        self.truncateds = np.zeros(n_envs, dtype=bool)
        self.infos = {
            "xy": np.zeros((n_envs, 2), np.float32),
            "goal_xy": np.ones((n_envs, 2), np.float32),
            "qpos": np.zeros((n_envs, 5), np.float32),
            "qvel": np.zeros((n_envs, 5), np.float32),
            "success": np.zeros(n_envs, np.float32),
        }

    def step(self):
        action = self.policy.get_action(self.infos)
        self.t += 1
        self.infos = {k: v.copy() for k, v in self.infos.items()}
        for i in range(self.num_envs):
            self.infos["xy"][i] = [self.t, self.t + i]
        self.infos["success"][-1] = float(self.t >= 3)
        self.terminateds = self.infos["success"] > 0
        return action


def _run(make_trace=None, n_envs=2, steps=6, task_ids=(10, 11)):
    policy = FakePolicy(n_envs)
    world = FakeWorld(policy, n_envs, steps)
    tracer = None
    if make_trace is not None:
        tracer = make_trace()
        tracer.begin()
        tracer.attach(world, policy)
    actions = []
    for _ in range(steps):
        policy.solver.solve(
            info_dict={"goal_embed_0": torch.ones(n_envs, 1, 3)},
            planning_horizon=2,
            steps_taken=policy.calls * 2,
            eval_budget=100,
        )
        actions.append(world.step().copy())
    if tracer is not None:
        tracer.finish()
    return actions, policy, world


def test_trace_off_is_noop(tmp_path):
    from antmaze_trace import install_trace

    class Cfg(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    cfg = type("Obj", (), {})()
    cfg.eval = Cfg({})
    cfg.output = {"dir": "w0"}
    cfg.seed = 1
    cfg.hierarchical_plan_config = type(
        "P", (), {"receding_horizon": 2}
    )()
    cfg.hierarchical_plan_config.level1 = type("L1", (), {"action_block": 5})()
    policy = FakePolicy(2)
    world = FakeWorld(policy)
    before_step = world.step
    assert install_trace(cfg=cfg, world=world, policy=policy, results_dir=tmp_path) is None
    assert world.step == before_step
    assert not list(tmp_path.rglob("*.jsonl"))


def test_actions_rng_and_records(tmp_path):
    control, control_policy, _ = _run()

    np.random.seed(1234)
    torch.manual_seed(1234)
    expected_rng = (np.random.rand(), torch.rand(3))

    np.random.seed(1234)
    torch.manual_seed(1234)
    traced, traced_policy, _ = _run(
        lambda: AntMazeTrace(
            tmp_path / "trace",
            seed=7,
            task_ids=[10, 11],
            replanning_interval=2,
        )
    )
    got_rng = (np.random.rand(), torch.rand(3))

    # 1) identical action stream
    for a, b in zip(control, traced):
        np.testing.assert_array_equal(a, b)
    assert traced_policy.solver.solve_count == control_policy.solver.solve_count

    # 2) no random numbers consumed
    assert expected_rng[0] == got_rng[0]
    assert torch.equal(expected_rng[1], got_rng[1])

    # 3) step records: one initial + one per env step, per env
    lines = (tmp_path / "trace" / "steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 * (6 + 1)
    records = [json.loads(line) for line in lines]
    assert {(r["env"], r["task"]) for r in records} == {(0, 10), (1, 11)}
    assert records[0]["t"] == 0 and records[0]["action"] is None
    assert records[1]["t"] == 0 and records[1]["action"] is None

    # 4) planning records carry per-level solver metadata
    plan = [
        json.loads(line)
        for line in (tmp_path / "trace" / "planning.jsonl").read_text().splitlines()
    ]
    assert len(plan) == 6
    assert plan[0]["level"] == 1
    assert plan[0]["n_iters"] == [3, 3]
    assert plan[0]["actions"][0][0] == [0.0, 0.0]
    assert plan[1]["call_idx"] == 1
    assert plan[0]["cost_batches"][0]["n_iters"] == 3
    assert len(plan[0]["cost_batches"][0]["cost_curve"]) == 4

    # 5) per-task files are disjoint and tagged
    for task, env in ((10, 0), (11, 1)):
        payload = json.loads(
            (tmp_path / "trace" / "tasks" / f"task_{task:03d}.json").read_text()
        )
        assert payload["task"] == task
        assert {s["env"] for s in payload["steps"]} == {env}
        assert all(p["level"] == 1 for p in payload["planning"])

    done = json.loads((tmp_path / "trace" / "trace_done.json").read_text())
    assert done["n_step_records"] == 14
    assert done["n_planning_records"] == 6
    assert done["error"] is None


def test_multi_level_and_missing_task_ids(tmp_path):
    policy = FakePolicy(2)
    solver2 = FakeSolver(2)
    solver2.level = 2
    solver2.horizon = 4
    solver2.model = FakeCostModel(2)

    class HierPolicy(FakePolicy):
        def __init__(self, n_envs):
            FakePolicy.__init__(self, n_envs)
            level1 = self.solver
            self.solver = type("Hier", (), {})()
            self.solver.level_solvers = {1: level1, 2: solver2}

    policy = HierPolicy(2)
    world = FakeWorld(policy)

    tracer = AntMazeTrace(tmp_path / "multi", task_ids=None)
    tracer.begin()
    tracer.attach(world, policy)

    for level, solver in ((1, policy.solver.level_solvers[1]), (2, solver2)):
        solver.solve(
            info_dict={"goal_embed_0": torch.ones(2, 1, 3)},
            planning_horizon=2,
            steps_taken=0,
            eval_budget=100,
        )
    world.step()
    tracer.finish()

    plan = [
        json.loads(line)
        for line in (tmp_path / "multi" / "planning.jsonl").read_text().splitlines()
    ]
    assert [p["level"] for p in plan] == [1, 2]
    records = [
        json.loads(line)
        for line in (tmp_path / "multi" / "steps.jsonl").read_text().splitlines()
    ]
    assert records[0]["task"] == 0 and records[1]["task"] == 1
