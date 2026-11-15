#!/usr/bin/env python3
"""TRAIN-ONLY LIVE all-level export for the V1.1 topology waypoints.

The frozen B1 export (`export_hide_live_train.py`) stores L3 and L2 only. The
topology adapter replaces the goal at ALL THREE levels with one LIVE-rendered
training representative, so every level must be exported and every level must
prove state/goal encoder parity (AGENT_TASK III.3: parity "at each level, not
just level 3").

Same environment protocol as B1 - restore the recorded qpos/qvel into the real
`MegaWrapper -> SyncWorld -> VariationWrapper` chain, re-render with `get_ob`,
then `WorldModelPolicy._prepare_info -> HierarchicalSolver._encode_all_levels`.
HDF5 pixels are never used.

  python scripts/export_topology_states.py \
    --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
    --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
    --output artifacts/hide/live_train_states_v11.npz \
    --max-episodes 150 --every 10 --batch-size 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
from omegaconf import OmegaConf
from stable_worldmodel.world import _find_env_method

from data import NORMALIZER_ARTIFACT_FILENAME
from planning_eval import (
    _build_eval_process_from_policy_normalizer, _build_policy,
    img_transform, load_model,
)

LEVELS = (1, 2, 3)
PARITY_TOL = 1e-4


def sha256_file(file: Path) -> str:
    hasher = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rows_from_train(episodes, offsets, *, max_episodes: int, every: int):
    if every < 1 or max_episodes < 12:
        raise ValueError("Must sample from >=12 episodes and every>=1")
    rows, episode, steps = [], [], []
    for i, length in enumerate(episodes[:max_episodes]):
        for t in range(0, int(length), every):
            rows.append(int(offsets[i]) + t)
            episode.append(i)
            steps.append(t)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(episode, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--max-episodes", type=int, default=150)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=420042)
    args = ap.parse_args()
    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse non-training HDF5")
    ckpt = args.checkpoint.resolve()
    norm = ckpt.parent / NORMALIZER_ARTIFACT_FILENAME
    if not norm.is_file():
        raise FileNotFoundError(norm)

    cfg = OmegaConf.load(args.config)
    cfg.policy = str(ckpt)
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    policy = _build_policy(
        cfg, process, {"pixels": img_transform(cfg), "goal": img_transform(cfg)},
        model=model,
    )
    if len(policy.solver.level_models) != 3:
        raise ValueError("Expected a 3-level world model")

    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    cfg.world.num_envs = args.batch_size
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    world = swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),) * 2)
    parts = {f"embed_{lv}": [] for lv in LEVELS}
    goal_parts = {f"embed_{lv}": [] for lv in LEVELS}
    xy_parts = []
    parity = {f"embed_{lv}": 0.0 for lv in LEVELS}
    try:
        world.set_policy(policy)
        policy.reset()
        seeds = [args.seed + i for i in range(args.batch_size)]
        for i, env in enumerate(world.envs.unwrapped.envs):
            env.action_space.seed(int(seeds[i]))
        world.reset(seed=seeds, options=[{} for _ in range(args.batch_size)])
        with h5py.File(args.dataset, "r") as h:
            rows, episode, steps = rows_from_train(
                h["ep_len"][:], h["ep_offset"][:],
                max_episodes=args.max_episodes, every=args.every,
            )
            for start in range(0, len(rows), args.batch_size):
                current = rows[start:start + args.batch_size]
                qpos_src = np.asarray(h["qpos"][current], dtype=np.float64)
                qvel_src = np.asarray(h["qvel"][current], dtype=np.float64)
                count = len(current)
                qpos = np.concatenate(
                    (qpos_src, np.repeat(qpos_src[-1:], args.batch_size - count, 0)),
                    axis=0,
                )
                qvel = np.concatenate(
                    (qvel_src, np.repeat(qvel_src[-1:], args.batch_size - count, 0)),
                    axis=0,
                )
                pixels, proprio, xy = [], [], []
                for i, env in enumerate(world.envs.unwrapped.envs):
                    set_state = _find_env_method(env, "set_state")
                    get_ob = _find_env_method(env, "get_ob")
                    get_xy = _find_env_method(env, "get_xy")
                    if any(f is None for f in (set_state, get_ob, get_xy)):
                        raise RuntimeError("Missing World wrapper state/render method")
                    set_state(qpos=qpos[i], qvel=qvel[i])
                    img = np.asarray(get_ob())
                    if img.ndim != 3 or img.shape[-1] != 3:
                        raise ValueError(f"Expected live HWC RGB, got {img.shape}")
                    pixels.append(img)
                    xy.append(np.asarray(get_xy(), dtype=np.float32))
                    proprio.append(
                        np.concatenate((qpos[i, 2:], qvel[i])).astype(np.float32)
                    )
                raw = {
                    "pixels": np.asarray(pixels)[:, None],
                    "proprio": np.asarray(proprio)[:, None],
                    "goal": np.asarray(pixels)[:, None].copy(),
                    "goal_proprio": np.asarray(proprio)[:, None].copy(),
                }
                import torch

                with torch.inference_mode():
                    prepared = policy._prepare_info(raw)
                    state = policy.solver._encode_all_levels(prepared, key="pixels")
                    goal = policy.solver._encode_all_levels(prepared, key="goal")
                for lv in LEVELS:
                    key = f"embed_{lv}"
                    delta = float((state[key] - goal[key]).abs().amax().item())
                    parity[key] = max(parity[key], delta)
                    if state[key].shape[:2] != (args.batch_size, 1):
                        raise ValueError(f"{key} shape {tuple(state[key].shape)}")
                    parts[key].append(state[key][:count, 0].float().cpu().numpy())
                    goal_parts[key].append(goal[key][:count, 0].float().cpu().numpy())
                xy_parts.append(np.asarray(xy, dtype=np.float32)[:count, :2])
                if start % (args.batch_size * 20) == 0:
                    done = min(start + args.batch_size, len(rows))
                    print(f"Encoded {done}/{len(rows)} live TRAIN states", flush=True)
    finally:
        world.close()

    bad = {k: v for k, v in parity.items() if v > PARITY_TOL}
    if bad:
        raise RuntimeError(
            f"NO-GO: state vs goal encoder parity failed at {bad}; "
            "waypoint goal embeddings would be inconsistent across levels"
        )
    arrays = {}
    for lv in LEVELS:
        arrays[f"embed_{lv}"] = np.concatenate(parts[f"embed_{lv}"])
        arrays[f"goal_embed_{lv}"] = np.concatenate(goal_parts[f"embed_{lv}"])
    xy = np.concatenate(xy_parts)
    for k, v in arrays.items():
        if not np.isfinite(v).all():
            raise ValueError(f"Non-finite {k}")
    if not len(xy) == len(episode) == len(rows):
        raise ValueError("Axis mismatch")

    meta = {
        "source_split": "train",
        "dataset_name": args.dataset.stem,
        "source_data_sha256": sha256_file(args.dataset),
        "render_domain": "live_eval_pipeline",
        "encoder_contract": (
            "WorldModelPolicy._prepare_info -> HierarchicalSolver._encode_all_levels"
        ),
        "checkpoint_sha256": sha256_file(ckpt),
        "normalizer_sha256": sha256_file(norm),
        "num_episodes": int(args.max_episodes),
        "frame_stride": args.every,
        "num_states": int(len(xy)),
        "levels": list(LEVELS),
        "all_level_goal_parity_passed": True,
        "parity_max_abs": parity,
        "no_dev_or_official_tasks": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xy=xy,
        episode=episode,
        source_row=rows,
        source_step=steps,
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
        **arrays,
    )
    print(
        json.dumps(
            {
                "num_states": meta["num_states"],
                "parity_max_abs": {k: round(v, 9) for k, v in parity.items()},
                "shapes": {k: list(v.shape) for k, v in arrays.items()},
            },
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
