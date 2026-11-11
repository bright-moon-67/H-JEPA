"""Anchor-centric counterfactual collector (plan2 P1).

One physical anchor state -> K real action branches, each executed ONCE,
producing C(K,2) counterfactual pairs. Train/val/test are split by ANCHOR
(disjoint anchor seeds). Writes <split>.h5 in the anchor layout used by
action_response_data.py.

    python -u scripts/collect_action_response_multi.py \
        --out-dir /tmp/pusht_cf --branches 4 \
        --train-anchors 2000 --val-anchors 300 --test-anchors 300 --workers 12

Determinism: both/all branches replay the SAME seed and SAME roll-in prefix
before diverging; the collector asserts identical 7-D anchor state and
identical anchor pixels. The 7-D observation is NOT a full Pymunk snapshot,
so `_set_state` is never used.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401 -- registers swm/PushT-v1

from action_response_data import save_anchor_h5

MODES = ("free", "approach", "push")
# Raw env-action deltas around the guided base action.
BRANCH_DIRECTIONS = (
    (0.0, 0.0),
    (0.4, 0.0),
    (-0.4, 0.0),
    (0.0, 0.4),
    (0.0, -0.4),
    (0.6, 0.0),
)


def make_env():
    return gym.make(
        "swm/PushT-v1",
        render_mode="rgb_array",
        resolution=224,
        with_target=False,
        relative=True,
    )


def steer_action(state, rng, *, mode):
    if mode == "free":
        return rng.uniform(-0.75, 0.75, size=2).astype(np.float32)
    agent_xy = np.asarray(state[:2], dtype=np.float64)
    block_xy = np.asarray(state[2:4], dtype=np.float64)
    direction = (block_xy - agent_xy) / 100.0
    noise = rng.normal(0.0, 0.08 if mode == "push" else 0.15, size=2)
    return np.clip(direction + noise, -0.85, 0.85).astype(np.float32)


def run_branch(env, seed, prefix, branch_actions):
    obs, _ = env.reset(seed=int(seed))
    for a in prefix:
        obs, _, _, _, _ = env.step(a)

    start_state = np.asarray(obs["state"], dtype=np.float32).copy()
    start_rgb = np.asarray(env.render(), dtype=np.uint8).copy()

    contacts = 0
    for a in branch_actions:
        obs, _, _, _, info = env.step(a)
        contacts += int(info.get("n_contacts", 0))

    return {
        "start_state": start_state,
        "start_rgb": start_rgb,
        "end_state": np.asarray(obs["state"], dtype=np.float32).copy(),
        "end_rgb": np.asarray(env.render(), dtype=np.uint8).copy(),
        "contacts": contacts,
    }


def make_action_branches(base_action, rng, n_branches, horizon=5):
    base = np.repeat(np.asarray(base_action, dtype=np.float32)[None], horizon, axis=0)
    if n_branches > len(BRANCH_DIRECTIONS):
        raise ValueError(f"n_branches must be <= {len(BRANCH_DIRECTIONS)}")
    branches = []
    for delta in BRANCH_DIRECTIONS[:n_branches]:
        noise = rng.normal(0.0, 0.04, size=(horizon, 2)).astype(np.float32)
        branch = np.clip(base + np.asarray(delta, np.float32) + noise, -1.0, 1.0)
        branches.append(branch.astype(np.float32))
    return branches


def collect_anchor(env, anchor_seed, rng, mode, n_branches):
    obs, _ = env.reset(seed=anchor_seed)

    if mode == "free":
        n_prefix = int(rng.integers(0, 4))
    elif mode == "approach":
        n_prefix = int(rng.integers(4, 9))
    else:
        n_prefix = int(rng.integers(9, 16))

    prefix = []
    for _ in range(n_prefix):
        a = steer_action(obs["state"], rng, mode=mode)
        prefix.append(a)
        obs, _, _, _, _ = env.step(a)
    prefix = np.asarray(prefix, dtype=np.float32).reshape(-1, 2)

    base = steer_action(obs["state"], rng, mode=mode)
    actions = make_action_branches(base, rng, n_branches)
    outcomes = [run_branch(env, anchor_seed, prefix, a) for a in actions]

    for k, result in enumerate(outcomes):
        np.testing.assert_allclose(
            result["start_state"], outcomes[0]["start_state"],
            rtol=0, atol=1e-5,
            err_msg=f"anchor state differs, branch={k}, seed={anchor_seed}",
        )
        if not np.array_equal(result["start_rgb"], outcomes[0]["start_rgb"]):
            raise AssertionError(f"anchor pixels differ, branch={k}, seed={anchor_seed}")

    return {
        "anchor_seed": anchor_seed,
        "mode": mode,
        "rgb0": outcomes[0]["start_rgb"],
        "end_rgb": np.stack([o["end_rgb"] for o in outcomes]),
        "actions": np.stack(actions),
        "state0": outcomes[0]["start_state"],
        "end_state": np.stack([o["end_state"] for o in outcomes]),
        "contacts": np.asarray([o["contacts"] for o in outcomes], dtype=np.int32),
        "prefix_len": len(prefix),
    }


_WORKER_ENV = None


def _worker_init():
    global _WORKER_ENV
    _WORKER_ENV = make_env()


def _worker_task(job):
    anchor_seed, mode, n_branches = job
    rng = np.random.default_rng(anchor_seed)
    return collect_anchor(_WORKER_ENV, anchor_seed, rng, mode, n_branches)


def collect_split(n_anchors, seed, n_branches, workers):
    jobs = [
        (int(seed) * 100_000 + i, MODES[i % len(MODES)], n_branches)
        for i in range(n_anchors)
    ]
    rows = []
    if workers <= 1:
        _worker_init()
        for k, job in enumerate(jobs):
            rows.append(_worker_task(job))
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"  collected {k+1}/{len(jobs)}", flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_worker_init) as pool:
            for k, row in enumerate(
                pool.imap_unordered(_worker_task, jobs, chunksize=8), 1
            ):
                rows.append(row)
                if k % 100 == 0 or k == len(jobs):
                    print(f"  collected {k}/{len(jobs)}", flush=True)

    rows.sort(key=lambda r: r["anchor_seed"])
    return {
        "anchor_seed": np.asarray([r["anchor_seed"] for r in rows], dtype=np.int64),
        "modes": np.asarray([r["mode"] for r in rows]),
        "rgb0": np.stack([r["rgb0"] for r in rows]),
        "end_rgb": np.stack([r["end_rgb"] for r in rows]),
        "actions": np.stack([r["actions"] for r in rows]),
        "state0": np.stack([r["state0"] for r in rows]),
        "end_state": np.stack([r["end_state"] for r in rows]),
        "contacts": np.stack([r["contacts"] for r in rows]),
        "prefix_len": np.asarray([r["prefix_len"] for r in rows], dtype=np.int32),
    }


def report_coverage(name, data):
    n, k = data["contacts"].shape
    contact_any = (data["contacts"] > 0).any(axis=1)
    triu = np.triu(np.ones((k, k), dtype=bool), k=1)
    pair_cov = (
        (data["contacts"][:, :, None] + data["contacts"][:, None, :]) > 0
    )[:, triu]
    print(f"[{name}] anchors={n}, branches={k}, pairs={n * k * (k - 1) // 2}")
    print(f"[{name}] contact anchors={contact_any.mean():.1%}, "
          f"contact pairs={pair_cov.mean():.1%}")
    for mode in MODES:
        m = data["modes"] == mode
        if m.any():
            print(f"[{name}]   {mode:>8s}: n={m.sum():4d}, "
                  f"contact anchors={contact_any[m].mean():.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--train-anchors", type=int, default=2000)
    parser.add_argument("--val-anchors", type=int, default=300)
    parser.add_argument("--test-anchors", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = (
        ("train", args.train_anchors, args.seed),
        ("val", args.val_anchors, args.seed + 1000),
        ("test", args.test_anchors, args.seed + 2000),
    )
    seen = set()
    for name, n_anchors, seed in splits:
        if n_anchors < 1:
            continue
        print(f"collecting {name}: {n_anchors} anchors x {args.branches} branches",
              flush=True)
        data = collect_split(n_anchors, seed, args.branches, args.workers)
        seeds = set(int(s) for s in data["anchor_seed"])
        assert not (seeds & seen), f"anchor seeds collide between splits: {name}"
        seen |= seeds
        report_coverage(name, data)
        path = out / f"{name}.h5"
        save_anchor_h5(path, data)
        print(f"saved {path} ({path.stat().st_size / 1024**2:.0f} MiB)", flush=True)

    print("anchor split disjointness verified", flush=True)


if __name__ == "__main__":
    main()
