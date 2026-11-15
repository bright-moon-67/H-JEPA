#!/usr/bin/env python3
"""Export dev-task START/GOAL L3 embeddings using the EXACT live eval pipeline.

The 12-task development pack stores placeholder ZERO pixels, so they must never
be encoded.  This script rebuilds the real ``xy_neutral`` world state with the
project's deterministic protocol (MegaWrapper -> SyncWorld -> VariationWrapper,
per-task ``action_space`` seeding, ``settled_info_from_xy`` settle steps, then
the eval callables ``set_state`` and ``set_navigation_goal``) and encodes the
resulting observations with ``WorldModelPolicy._prepare_info`` followed by
``HierarchicalSolver._encode_all_levels`` - the same code path the planner uses.

XY is never fed to the planner; it only defines the offline replay state.

Usage from H-JEPA/h_jepa:
  PYTHONPATH=. python scripts/export_pahrl_dev_queries.py \
    --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
    --dev-tasks audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
    --output artifacts/pahrl/queries_dev.npz \
    --env-seeds 1000 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from planning_eval import (
    _build_eval_process_from_policy_normalizer,
    _build_policy,
    img_transform,
    load_model,
)

from stable_worldmodel.world import _find_env_method


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _proprio_from_info(info: dict) -> np.ndarray:
    """Match World._match_proprio_dim: qpos[2:] + qvel (27 dims for AntMaze)."""
    qpos = np.asarray(info["qpos"])
    qvel = np.asarray(info["qvel"])
    return np.concatenate([qpos[..., 2:], qvel], axis=-1).astype(np.float32)


def build_world(cfg, num_envs: int):
    """Mirror run_planning_eval's world construction exactly."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    cfg.world.num_envs = int(num_envs)
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    image_shape = (int(cfg.eval.img_size),) * 2
    return cfg, swm.World(**world_cfg, image_shape=image_shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dev-tasks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    p.add_argument("--env-seeds", type=int, nargs="+", default=[1000, 2000])
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.policy = str(args.checkpoint.resolve())
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    policy = _build_policy(cfg, process, transform, model=model)
    solver = policy.solver
    if len(solver.level_models) != 3:
        raise ValueError("Only H-JEPA L3 is supported")

    payload = torch.load(
        args.dev_tasks, map_location="cpu", weights_only=False
    )
    data = payload["data"]
    task_ids = [int(x) for x in payload["episodes_idx"]]
    n_tasks = len(task_ids)

    start_z_all, goal_z_all = [], []
    task_col, seed_col = [], []
    start_hash_rows, goal_hash_rows = [], []

    for env_seed in args.env_seeds:
        cfg_resolved, world = build_world(cfg, n_tasks)
        try:
            policy.reset()
            world.set_policy(policy)
            seeds = [int(env_seed) + gid for gid in task_ids]
            for i, env in enumerate(world.envs.unwrapped.envs):
                env.action_space.seed(int(seeds[i]))
            world.reset(seed=seeds, options=[{} for _ in range(n_tasks)])

            start_pixels, goal_pixels = [], []
            start_proprio, goal_proprio = [], []
            for i, env in enumerate(world.envs.unwrapped.envs):
                start_xy = np.asarray(data[i]["xy"][0], dtype=np.float64)
                goal_xy = np.asarray(data[i]["goal_xy"][-1], dtype=np.float64)
                settle = _find_env_method(env, "settled_info_from_xy")
                if settle is None:
                    raise RuntimeError("Environment lacks settled_info_from_xy")
                start_info = settle(start_xy, render_pixels=True)
                goal_info = settle(goal_xy, render_pixels=True)

                set_state = _find_env_method(env, "set_state")
                if set_state is None:
                    raise RuntimeError("Environment lacks set_state callable")
                set_state(
                    qpos=np.asarray(start_info["qpos"]).copy(),
                    qvel=np.asarray(start_info["qvel"]).copy(),
                )
                set_nav = _find_env_method(env, "set_navigation_goal")
                if set_nav is None:
                    raise RuntimeError("Environment lacks set_navigation_goal")
                set_nav(goal_xy=goal_xy.copy())

                start_pixels.append(np.asarray(start_info["pixels"]))
                goal_pixels.append(np.asarray(goal_info["pixels"]))
                start_proprio.append(_proprio_from_info(start_info))
                goal_proprio.append(_proprio_from_info(goal_info))
                start_hash_rows.append({
                    "task": task_ids[i],
                    "env_seed": int(env_seed),
                    "seed": int(seeds[i]),
                    "xy": sha256_array(start_info["xy"]),
                    "qpos": sha256_array(np.asarray(start_info["qpos"])),
                    "qvel": sha256_array(np.asarray(start_info["qvel"])),
                    "pixels": sha256_array(np.asarray(start_info["pixels"])),
                    "goal_pixels": sha256_array(np.asarray(goal_info["pixels"])),
                })

            raw = {
                "pixels": np.stack(start_pixels)[:, None],
                "proprio": np.stack(start_proprio)[:, None],
            }
            raw_goal = {
                "goal": np.stack(goal_pixels)[:, None],
                "goal_proprio": np.stack(goal_proprio)[:, None],
            }
            with torch.inference_mode():
                start_z = solver._encode_all_levels(
                    policy._prepare_info(raw), key="pixels"
                )["embed_3"][:, 0]
                goal_z = solver._encode_all_levels(
                    policy._prepare_info(raw_goal), key="goal"
                )["embed_3"][:, 0]
            start_z_all.append(start_z.float().cpu().numpy())
            goal_z_all.append(goal_z.float().cpu().numpy())
            task_col.extend(task_ids)
            seed_col.extend([int(env_seed)] * n_tasks)
            goal_hash_rows.extend(
                [
                    {
                        "task": task_ids[i],
                        "env_seed": int(env_seed),
                        "seed": int(seeds[i]),
                        "goal_xy": sha256_array(
                            np.asarray(data[i]["goal_xy"][-1])
                        ),
                    }
                    for i in range(n_tasks)
                ]
            )
            print(f"encoded {n_tasks} queries for env_seed={env_seed}")
        finally:
            world.close()

    start_z = np.concatenate(start_z_all, axis=0)
    goal_z = np.concatenate(goal_z_all, axis=0)
    if not np.isfinite(start_z).all() or not np.isfinite(goal_z).all():
        raise ValueError("Non-finite query encodings")

    normalizer_file = args.checkpoint.parent / "normalizer.pt"
    if not normalizer_file.is_file():
        raise FileNotFoundError(f"Missing normalizer: {normalizer_file}")
    meta = {
        "format": "pahrl_hjepa_dev_queries_v1",
        "source_split": "dev_tasks",
        "dev_task_pack": str(args.dev_tasks),
        "dev_task_sha256": sha256_file(args.dev_tasks),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "normalizer_sha256": sha256_file(normalizer_file),
        "checkpoint_path": str(args.checkpoint),
        "env_seeds": [int(x) for x in args.env_seeds],
        "task_ids": task_ids,
        "protocol": (
            "MegaWrapper->SyncWorld->VariationWrapper; action_space.seed("
            "env_seed+task); reset(seed); settled_info_from_xy(start/goal, "
            "render_pixels=True); set_state(qpos,qvel); set_navigation_goal"
        ),
        "encoder_contract": (
            "WorldModelPolicy._prepare_info -> "
            "HierarchicalSolver._encode_all_levels"
        ),
        "eval_config": str(args.config),
        "embed_dim": int(start_z.shape[1]),
        "start_hashes": start_hash_rows,
        "goal_hashes": goal_hash_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        start_z=start_z,
        goal_z=goal_z,
        task=torch.tensor(task_col).numpy(),
        env_seed=torch.tensor(seed_col).numpy(),
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    print("Saved", args.output, start_z.shape, goal_z.shape)


if __name__ == "__main__":
    main()
