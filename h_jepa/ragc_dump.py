"""Opt-in instrumentation for official eval action-trace dumping.

Enable by setting RAGC_DUMP=/path/to/dump.jsonl before running eval.py.
Records one JSONL line per level-1 GradientSolver.solve call plus a task_start
marker per evaluate_from_dataset call. Does not alter planner behavior.
"""
import hashlib
import json
import os

import stable_worldmodel as swm
from stable_worldmodel.solver.gd import GradientSolver

_state = {"task": -1, "installed": False}


def _hash_tensor(t):
    import torch
    if not torch.is_tensor(t):
        return None
    return hashlib.sha1(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def install():
    if _state["installed"]:
        return
    path = os.environ.get("RAGC_DUMP")
    if not path:
        return

    orig_solve = GradientSolver.solve

    def solve(self, *args, **kwargs):
        result = orig_solve(self, *args, **kwargs)
        if int(getattr(self, "level", 1)) == 1:
            info = kwargs.get("info_dict")
            _append(path, {
                "kind": "solve",
                "task": _state["task"],
                "steps_taken": kwargs.get("steps_taken"),
                "planning_horizon": kwargs.get("planning_horizon"),
                "pixels_sha1": _hash_tensor(info.get("pixels")) if info else None,
                "goal_sha1": _hash_tensor(info.get("goal")) if info else None,
                "actions": result["actions"].detach().cpu().numpy().tolist(),
            })
        return result

    GradientSolver.solve = solve

    orig_eval = swm.World.evaluate_from_dataset

    def evaluate(self, *args, **kwargs):
        _state["task"] += 1
        _append(path, {"kind": "task_start", "task": _state["task"]})
        return orig_eval(self, *args, **kwargs)

    swm.World.evaluate_from_dataset = evaluate
    _state["installed"] = True
