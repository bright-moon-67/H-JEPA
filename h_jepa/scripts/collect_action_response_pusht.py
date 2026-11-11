"""Generate real paired counterfactual Push-T rollouts.

Crucial: do NOT approximate a full Pymunk snapshot by repeatedly calling
_set_state(state7). Restore identical hidden dynamics by replaying the SAME
seed and SAME roll-in action history before branching.

Run from h_jepa/:
    python -u scripts/collect_action_response_pusht.py \
        --out /tmp/pusht_cf_train.npz --pairs 64 --seed 42
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401 -- registers swm/PushT-v1


def make_env():
    # Signature verified against stable_worldmodel/envs/pusht/env.py.
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
    # Approximate agent-to-block guidance. This is a DATA COLLECTION
    # heuristic, not a task-specific feature used by ActionResponseNet.
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


def collect(n_pairs, seed, fixed_probes=False):
    rng = np.random.default_rng(seed)
    env = make_env()
    rows = []
    modes = ("free", "approach", "push")
    try:
        for i in range(n_pairs):
            # Different seed per anchor; SAME seed reused by both branches.
            anchor_seed = int(seed) * 100_000 + i
            mode = modes[i % len(modes)]
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

            if fixed_probes:
                # The exact SAME candidate action pair across anchor states.
                # Used to inspect whether response varies with state/mode.
                acts_a = np.tile(np.array([0.60, 0.00], np.float32), (5, 1))
                acts_b = np.tile(np.array([0.60, 0.35], np.float32), (5, 1))
            else:
                base = steer_action(obs["state"], rng, mode=mode)
                acts_a = np.clip(
                    base[None] + rng.normal(0, 0.10, (5, 2)), -1, 1
                ).astype(np.float32)
                acts_b = np.clip(
                    acts_a + rng.normal(0, 0.28, (5, 2)), -1, 1
                ).astype(np.float32)

            A = run_branch(env, anchor_seed, prefix, acts_a)
            B = run_branch(env, anchor_seed, prefix, acts_b)

            # A fundamental causal-data invariant.
            np.testing.assert_allclose(
                A["start_state"], B["start_state"], rtol=0, atol=1e-5,
                err_msg=f"anchor state differs, pair={i}"
            )
            if not np.array_equal(A["start_rgb"], B["start_rgb"]):
                raise AssertionError(f"anchor pixels differ, pair={i}")
            if float(np.max(np.abs(acts_a - acts_b))) < 1e-7:
                raise AssertionError(f"identical actions, pair={i}")

            rows.append({
                "rgb0": A["start_rgb"],
                "rgb_a": A["end_rgb"],
                "rgb_b": B["end_rgb"],
                "acts_a": acts_a,
                "acts_b": acts_b,
                "state0": A["start_state"],
                "state_a": A["end_state"],
                "state_b": B["end_state"],
                "contacts_a": A["contacts"],
                "contacts_b": B["contacts"],
                "mode": mode,
                "anchor_seed": anchor_seed,
            })
            if (i + 1) % 16 == 0 or i + 1 == n_pairs:
                print(f"collected {i+1}/{n_pairs}", flush=True)
    finally:
        env.close()

    data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    assert data["rgb0"].shape == (n_pairs, 224, 224, 3)
    assert data["acts_a"].shape == (n_pairs, 5, 2)
    assert data["rgb0"].dtype == np.uint8
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-probes", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1:
        raise ValueError("--pairs must be >= 1")
    data = collect(args.pairs, args.seed, args.fixed_probes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    count = np.count_nonzero((data["contacts_a"] + data["contacts_b"]) > 0)
    print(f"saved {out}; contact-pair coverage: {count}/{args.pairs}", flush=True)


if __name__ == "__main__":
    main()
