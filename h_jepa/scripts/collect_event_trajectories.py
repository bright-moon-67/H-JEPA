"""Collect continuous Push-T trajectories for the P3 event prototype.

Each episode is a scripted rollout with three phases (free move -> approach ->
push), so contact starts/ends occur naturally. Per-step RGB / action / state /
contact counts are stored for later signature analysis. mode/phase is NOT used
as an event label — event labels come from real `n_contacts` transitions.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/collect_event_trajectories.py \
        --out /tmp/pusht_events/events.h5 --episodes 25 --steps 60 --workers 6
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel.envs  # noqa: F401

MODES = ("free", "approach", "push")


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


def collect_episode(env, seed, n_steps):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    rgbs, actions, states, contacts = [], [], [], []
    for t in range(n_steps):
        if t < n_steps // 4:
            mode = "free"
        elif t < 3 * n_steps // 4:
            mode = "approach"
        else:
            mode = "push"
        a = steer_action(obs["state"], rng, mode=mode)

        rgb = np.asarray(env.render(), dtype=np.uint8)
        obs, _, _, _, info = env.step(a)

        rgbs.append(rgb)
        actions.append(a)
        states.append(np.asarray(obs["state"], dtype=np.float32))
        contacts.append(int(info.get("n_contacts", 0)))

    return {
        "seed": seed,
        "rgb": np.stack(rgbs),
        "actions": np.stack(actions).astype(np.float32),
        "states": np.stack(states).astype(np.float32),
        "contacts": np.asarray(contacts, dtype=np.int32),
    }


_E = None


def _init():
    global _E
    _E = make_env()


def _task(args):
    seed, n_steps = args
    return collect_episode(_E, seed, n_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    jobs = [(args.seed + i, args.steps) for i in range(args.episodes)]
    if args.workers <= 1:
        _init()
        rows = [_task(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers, initializer=_init) as pool:
            rows = pool.map(_task, jobs)

    rows.sort(key=lambda r: r["seed"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("rgb", data=np.stack([r["rgb"] for r in rows]),
                         compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=np.stack([r["actions"] for r in rows]),
                         compression="gzip", compression_opts=4)
        f.create_dataset("states", data=np.stack([r["states"] for r in rows]),
                         compression="gzip", compression_opts=4)
        f.create_dataset("contacts", data=np.stack([r["contacts"] for r in rows]))
        f.create_dataset("seeds", data=np.asarray([r["seed"] for r in rows]))

    c = np.stack([r["contacts"] for r in rows])
    transitions = (np.diff((c > 0).astype(np.int8), axis=1) != 0)
    print(f"saved {out} ({out.stat().st_size / 1024**2:.1f} MiB): "
          f"{len(rows)} episodes x {args.steps} steps")
    print(f"contact steps: {float((c > 0).mean()):.1%}; "
          f"transition steps: {int(transitions.sum())} "
          f"({transitions.sum() / len(rows):.2f} per episode)")


if __name__ == "__main__":
    main()
