#!/usr/bin/env python3
"""Stage-2 gate: live re-render of a stored TRAIN state must match the graph.

For a matched stored training state, restoring ``qpos``/``qvel`` in the real
wrapper chain and rendering through the live pipeline must reproduce the stored
observation and the exported L3 embedding to numerical tolerance.

Run from H-JEPA/h_jepa with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from pahrl_local_reach import (
    build_eval_stack,
    build_world,
    proprio_from_qpos_qvel,
    sha256_array,
)
from stable_worldmodel.world import _find_env_method


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nodes", type=int, nargs="+", default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    args = p.parse_args()

    with np.load(args.states, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        source_row = np.asarray(data["source_row"], dtype=np.int64)
        episode = np.asarray(data["episode"])
        steps = np.asarray(data["steps"])
    nodes = args.nodes
    if nodes is None:
        idx = np.linspace(0, len(z) - 1, args.num_samples).astype(int)
        nodes = [int(i) for i in idx]

    cfg, policy = build_eval_stack(args.checkpoint, args.config)
    world = build_world(cfg, len(nodes))
    rows = [{} for _ in nodes]
    try:
        policy.reset()
        world.set_policy(policy)
        seeds = [95000 + i for i in range(len(nodes))]
        for i, env in enumerate(world.envs.unwrapped.envs):
            env.action_space.seed(seeds[i])
        world.reset(seed=seeds, options=[{} for _ in range(len(nodes))])

        stored_pixels, live_pixels, live_proprio = [], [], []
        with h5py.File(args.dataset, "r") as handle:
            for i, node in enumerate(nodes):
                row = int(source_row[node])
                qpos = np.asarray(handle["qpos"][row], dtype=np.float64)
                qvel = np.asarray(handle["qvel"][row], dtype=np.float64)
                stored_pixels.append(np.asarray(handle["pixels"][row]))
                env = world.envs.unwrapped.envs[i]
                set_state = _find_env_method(env, "set_state")
                set_state(qpos=qpos, qvel=qvel)
                get_ob = _find_env_method(env, "get_ob")
                live_pixels.append(np.asarray(get_ob()))
                live_proprio.append(proprio_from_qpos_qvel(qpos, qvel))
                rows[i] = {
                    "node": int(node),
                    "episode": int(episode[node]),
                    "step": int(steps[node]),
                    "source_row": row,
                    "stored_pixels_sha256": sha256_array(stored_pixels[-1]),
                    "live_pixels_sha256": sha256_array(live_pixels[-1]),
                }

        raw = {
            "pixels": np.stack(live_pixels)[:, None],
            "proprio": np.stack(live_proprio)[:, None],
        }
        with torch.inference_mode():
            live_z = policy.solver._encode_all_levels(
                policy._prepare_info(raw), key="pixels"
            )["embed_3"][:, 0].float().cpu().numpy()
        live_unit = live_z / np.maximum(
            np.linalg.norm(live_z, axis=1, keepdims=True), 1e-8
        )
        stored_unit = z[np.asarray(nodes)] / np.maximum(
            np.linalg.norm(z[np.asarray(nodes)], axis=1, keepdims=True), 1e-8
        )
        cos = 1.0 - np.sum(live_unit * stored_unit, axis=1)
        for i, row in enumerate(rows):
            row["pixel_exact_match"] = bool(
                row["stored_pixels_sha256"] == row["live_pixels_sha256"]
            )
            row["cosine_distance_live_vs_graph"] = float(cos[i])
            row["max_abs_diff"] = float(
                np.max(np.abs(live_z[i] - z[nodes[i]]))
            )

        report = {
            "checkpoint": str(args.checkpoint),
            "states": str(args.states),
            "dataset": str(args.dataset),
            "num_nodes": len(nodes),
            "pixel_exact_match_count": int(
                sum(r["pixel_exact_match"] for r in rows)
            ),
            "max_cosine_distance": float(np.max(cos)),
            "max_abs_diff": float(max(r["max_abs_diff"] for r in rows)),
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    finally:
        world.close()


if __name__ == "__main__":
    main()
