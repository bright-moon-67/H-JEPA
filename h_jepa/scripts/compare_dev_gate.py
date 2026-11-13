#!/usr/bin/env python3
"""PHASE 1 full-budget gate: trace off vs on and initial-state reproduction."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h_jepa"))

OFF, ON_A, ON_B, DEV_PACK, OUT = sys.argv[1:6]


def metrics(p):
    return yaml.safe_load((Path(p) / "metrics.yaml").read_text())


def planning(p):
    f = Path(p) / "planning_compute.json"
    return json.loads(f.read_text()) if f.exists() else None


def steps_sha(p):
    h = hashlib.sha256()
    for line in (Path(p) / "trace" / "steps.jsonl").read_bytes().splitlines():
        rec = json.loads(line)
        h.update(json.dumps(rec, sort_keys=True).encode())
    return h.hexdigest()


def planning_sig(p):
    out = []
    for line in (Path(p) / "trace" / "planning.jsonl").read_text().splitlines():
        r = json.loads(line)
        out.append(
            {
                "level": r["level"],
                "call_idx": r["call_idx"],
                "steps_taken": r["steps_taken"],
                "horizon": r["horizon"],
                "n_iters": r["n_iters"],
                "actions": (
                    None
                    if r["actions"] is None
                    else [
                        [[round(float(x), 6) for x in step] for step in env]
                        for env in r["actions"]
                    ]
                ),
                "cost": (
                    None
                    if r["cost_batches"] is None
                    else [
                        [round(float(x), 6) for x in c["cost_curve"]]
                        for c in r["cost_batches"]
                    ]
                ),
            }
        )
    return out


def step0_hashes(p):
    first = {}
    for line in (Path(p) / "trace" / "steps.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["t"] == 0:
            first[str(r["task"])] = r["hashes"]
    return first


def nan_eq(a, b):
    return len(a) == len(b) and all(
        (x == y) or (isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y))
        for x, y in zip(a, b)
    )


res = {"checks": {}, "passed": True}


def check(name, ok, detail=None):
    res["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        res["passed"] = False


m_off, m_a, m_b = metrics(OFF), metrics(ON_A), metrics(ON_B)
check("off_vs_on_success", m_off["episode_successes"] == m_a["episode_successes"],
      {"off": m_off["episode_successes"], "on": m_a["episode_successes"]})
check("off_vs_on_steps", nan_eq(m_off.get("steps_to_success") or [], m_a.get("steps_to_success") or []))
check("off_vs_on_seeds", m_off.get("seeds") == m_a.get("seeds"), {"seeds": m_off.get("seeds")})

p_off, p_a = planning(OFF), planning(ON_A)
check("off_vs_on_planning_records", p_off is not None and p_a is not None and len(p_off) == len(p_a),
      {"off": None if p_off is None else len(p_off), "on": None if p_a is None else len(p_a)})
if p_off and p_a and len(p_off) == len(p_a):
    same = all(
        (x.get("level"), x.get("horizon"), x.get("num_samples"), x.get("n_iters"))
        == (y.get("level"), y.get("horizon"), y.get("num_samples"), y.get("n_iters"))
        for x, y in zip(p_off, p_a)
    )
    check("off_vs_on_planning_n_iters", same)

check("on_a_vs_on_b_success", m_a["episode_successes"] == m_b["episode_successes"])
check("on_a_vs_on_b_steps", nan_eq(m_a.get("steps_to_success") or [], m_b.get("steps_to_success") or []))
sha_a, sha_b = steps_sha(ON_A), steps_sha(ON_B)
check("on_a_vs_on_b_step_sha", sha_a == sha_b, {"a": sha_a, "b": sha_b})
check("on_a_vs_on_b_planning", planning_sig(ON_A) == planning_sig(ON_B))

meta_a = json.loads((Path(ON_A) / "trace" / "meta.json").read_text())
meta_b = json.loads((Path(ON_B) / "trace" / "meta.json").read_text())
for key in ("env_seed", "env_seed_by_task", "checkpoint_sha256", "planner_seeds",
            "batch_size_by_level", "env_to_batch", "git_commit"):
    check(f"meta_{key}", meta_a.get(key) == meta_b.get(key),
          {"a": meta_a.get(key), "b": meta_b.get(key)})

# Independent reproduction of the settled initial state for each task.
try:
    from functools import partial

    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401
    from stable_worldmodel.world import _find_env_method, _make_env
    from stable_worldmodel.wrapper import MegaWrapper, SyncWorld, VariationWrapper

    dev = torch.load(DEV_PACK, map_location="cpu", weights_only=False)
    hashes = step0_hashes(ON_A)
    kw = dict(
        maze_type="medium",
        dataset_type="stitch",
        ob_type="pixels",
        width=64,
        height=64,
        camera_name="back",
        terminate_at_goal=True,
        render_mode="rgb_array",
    )
    ok_all, detail = True, {}
    for i, ep in enumerate(dev["data"]):
        task = int(dev["episodes_idx"][i])
        if str(task) not in hashes:
            continue
        start_xy = np.asarray(ep["xy"][0], dtype=np.float64)
        seed = meta_a["env_seed_by_task"][str(task)]
        wrappers = [
            partial(MegaWrapper, image_shape=(64, 64), history_size=1, frame_skip=1)
        ]
        env_fn = partial(_make_env, "swm/OGBAntMaze-v0", 700, wrappers, **kw)
        vec = VariationWrapper(SyncWorld([env_fn]))
        vec.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED
        vec.unwrapped.envs[0].action_space.seed(seed)
        vec.reset(seed=[seed], options=[{}])
        settle = _find_env_method(vec.unwrapped.envs[0], "settled_info_from_xy")
        info = settle(start_xy, render_pixels=True)
        qpos = np.ascontiguousarray(np.asarray(info["qpos"]))
        qvel = np.ascontiguousarray(np.asarray(info["qvel"]))
        h_qpos = hashlib.sha256(qpos.tobytes()).hexdigest()
        h_qvel = hashlib.sha256(qvel.tobytes()).hexdigest()
        vec.close()
        got = hashes[str(task)]
        same = got["qpos"] == h_qpos and got["qvel"] == h_qvel
        ok_all = ok_all and same
        detail[str(task)] = {
            "offline_qpos": h_qpos,
            "trace_qpos": got["qpos"],
            "offline_qvel": h_qvel,
            "trace_qvel": got["qvel"],
            "match": same,
        }
    check("initial_state_reproduced", ok_all, detail)
except Exception as exc:  # pragma: no cover
    check("initial_state_reproduced", False, f"{type(exc).__name__}: {exc}")
    check("initial_state_reproduced", ok_all, detail)
except Exception as exc:  # pragma: no cover
    check("initial_state_reproduced", False, f"{type(exc).__name__}: {exc}")

Path(OUT).write_text(json.dumps(res, indent=1))
for name, info in res["checks"].items():
    print(f"[{'OK' if info['ok'] else 'FAIL'}] {name}")
print(f"PHASE1 GATE {'PASSED' if res['passed'] else 'FAILED'}")
raise SystemExit(0 if res["passed"] else 1)
