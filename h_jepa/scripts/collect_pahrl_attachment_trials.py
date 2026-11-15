#!/usr/bin/env python3
"""Collect held-out TRAIN-only local reachability trials for attachment calibration.

The graph is built from TRAIN episodes ``0..max_episodes-1``.  This script
samples queries from LATER TRAIN episodes (never validation or official test
data), attaches each query to nearby graph nodes by latent cosine distance,
and then drives the frozen H-JEPA controller to the memorized node state.

XY is used only to *select hard experiments* (same-region vs cross-wall) and
to measure physical success; it is never given to the planner.

Run from H-JEPA/h_jepa with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

from pahrl_local_reach import (
    LocalReachRunner,
    build_eval_stack,
    build_world,
    proprio_from_qpos_qvel,
)
from pahrl_memory import GraphMemory


def _owner_attr(env, name):
    owner = env
    while owner is not None and not hasattr(owner, name):
        nxt = getattr(owner, "env", None)
        if nxt is owner:
            break
        owner = nxt
    return getattr(owner, name, None)


def bfs_grid(maze_map: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Grid BFS distances over free cells (0 = free, 1 = wall)."""
    dist = np.full(maze_map.shape, -1, dtype=np.int32)
    if maze_map[start] != 0:
        return dist
    dist[start] = 0
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < maze_map.shape[0] and 0 <= nj < maze_map.shape[1]):
                continue
            if maze_map[ni, nj] != 0 or dist[ni, nj] >= 0:
                continue
            dist[ni, nj] = dist[i, j] + 1
            queue.append((ni, nj))
    return dist


def encode_queries(cfg, policy, pixels, proprios) -> np.ndarray:
    raw = {
        "pixels": np.asarray(pixels)[:, None],
        "proprio": np.asarray(proprios)[:, None],
    }
    with torch.inference_mode():
        z = policy.solver._encode_all_levels(
            policy._prepare_info(raw), key="pixels"
        )["embed_3"][:, 0]
    return z.float().cpu().numpy()


def live_query_embeddings(cfg, policy, qpos, qvel, base_seed: int) -> np.ndarray:
    """Encode queries exactly as the ONLINE controller would: restore the
    physical state in the live wrapper chain, render, then encode."""
    from stable_worldmodel.world import _find_env_method

    world = build_world(cfg, len(qpos))
    try:
        policy.reset()
        world.set_policy(policy)
        seeds = [base_seed + i for i in range(len(qpos))]
        for i, env in enumerate(world.envs.unwrapped.envs):
            env.action_space.seed(seeds[i])
        world.reset(seed=seeds, options=[{} for _ in range(len(qpos))])
        pixels, proprios = [], []
        for i, env in enumerate(world.envs.unwrapped.envs):
            set_state = _find_env_method(env, "set_state")
            set_state(qpos=qpos[i], qvel=qvel[i])
            get_ob = _find_env_method(env, "get_ob")
            pixels.append(np.asarray(get_ob()))
            proprios.append(proprio_from_qpos_qvel(qpos[i], qvel[i]))
        return encode_queries(cfg, policy, pixels, proprios)
    finally:
        world.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rollout-log", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    p.add_argument("--held-out-episodes", type=int, nargs="+", default=list(range(300, 308)))
    p.add_argument("--episode-step", type=int, default=300)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--base-seed", type=int, default=93000)
    args = p.parse_args()

    graph = GraphMemory.load(args.graph)
    if graph.metadata.get("source_split") != "train":
        raise ValueError("Attachment calibration must be TRAIN-only")
    with np.load(args.states, allow_pickle=False) as data:
        graph_episode = np.asarray(data["episode"])
        graph_steps = np.asarray(data["steps"])
        graph_source_row = np.asarray(data["source_row"], dtype=np.int64)
        graph_xy = np.asarray(data["xy_diagnostic_only"], dtype=np.float32)
    held = set(int(e) for e in args.held_out_episodes)
    if held & set(int(e) for e in np.unique(graph_episode)):
        raise ValueError("Held-out queries overlap graph episodes")

    cfg, policy = build_eval_stack(args.checkpoint, args.config)

    with h5py.File(args.dataset, "r") as handle:
        lengths = np.asarray(handle["ep_len"][:])
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        rows = []
        for episode in args.held_out_episodes:
            if args.episode_step >= int(lengths[episode]):
                raise ValueError(
                    f"Episode {episode} shorter than step {args.episode_step}"
                )
            rows.append(int(offsets[episode]) + int(args.episode_step))
        rows_arr = np.asarray(rows, dtype=np.int64)
        pixels = np.asarray(handle["pixels"][rows_arr])
        qpos = np.asarray(handle["qpos"][rows_arr], dtype=np.float64)
        qvel = np.asarray(handle["qvel"][rows_arr], dtype=np.float64)
        xy = np.asarray(handle["xy"][rows_arr], dtype=np.float64)
    proprios = np.stack(
        [proprio_from_qpos_qvel(qpos[i], qvel[i]) for i in range(len(rows))]
    )
    # The ONLINE controller decides attachments from a live render of the
    # current state, so the query embedding must come from the same path.
    query_z = live_query_embeddings(cfg, policy, qpos, qvel, args.base_seed)
    print("live query embeddings:", query_z.shape)

    # Maze geometry is used ONLY to pick hard offline experiments.
    probe_world = build_world(cfg, 1)
    try:
        probe = probe_world.envs.unwrapped.envs[0]
        maze_map = np.asarray(_owner_attr(probe, "maze_map"))
        xy_to_ij = _owner_attr(probe, "xy_to_ij")
        goal_tol = float(_owner_attr(probe, "_goal_tol") or 1.0)
    finally:
        probe_world.close()

    z_norm = graph.z_norm
    node_cells = np.stack(
        [np.asarray(xy_to_ij(graph_xy[idx])) for idx in range(len(graph_xy))]
    )

    trials: list[dict] = []
    trial_meta: list[dict] = []
    for qi, episode in enumerate(args.held_out_episodes):
        d = np.maximum(0.0, 1.0 - z_norm @ (query_z[qi] / np.linalg.norm(query_z[qi])))
        order = np.argsort(d, kind="stable")[: args.top_k]
        qcell = tuple(int(v) for v in np.asarray(xy_to_ij(xy[qi])))
        dist_grid = bfs_grid(maze_map, qcell)
        node_bfs = dist_grid[node_cells[:, 0], node_cells[:, 1]]
        node_xy_dist = np.linalg.norm(graph_xy - xy[qi], axis=1)

        same_region, cross_wall = [], []
        for idx in order:
            cell_dist = int(node_bfs[int(idx)])
            if cell_dist < 0:
                continue
            record = {
                "node": int(idx),
                "cosine_distance": float(d[int(idx)]),
                "xy_distance": float(node_xy_dist[int(idx)]),
                "cell_bfs_distance": cell_dist,
                "node_episode": int(graph_episode[int(idx)]),
                "node_step": int(graph_steps[int(idx)]),
            }
            if cell_dist <= 1:
                same_region.append(record)

        # Hard negative: the MOST tempting latent neighbour that is physically
        # close but separated by walls (fixed selection rule, offline only).
        cross_mask = (node_bfs >= 3) & (node_xy_dist <= 8.0)
        if np.any(cross_mask):
            masked = np.where(cross_mask, d, np.inf)
            idx = int(np.argmin(masked))
            cross_wall.append(
                {
                    "node": idx,
                    "cosine_distance": float(d[idx]),
                    "xy_distance": float(node_xy_dist[idx]),
                    "cell_bfs_distance": int(node_bfs[idx]),
                    "node_episode": int(graph_episode[idx]),
                    "node_step": int(graph_steps[idx]),
                }
            )

        chosen = []
        if same_region:
            chosen.append(
                ("same_region", min(same_region, key=lambda r: r["xy_distance"]))
            )
        if cross_wall:
            chosen.append(
                ("cross_wall", min(cross_wall, key=lambda r: r["cosine_distance"]))
            )
        if not chosen:
            print(f"query train_{episode}: no candidate attachment within top-k")

        for kind, record in chosen:
            node = record["node"]
            with h5py.File(args.dataset, "r") as handle:
                tgt_row = int(graph_source_row[node])
                tgt_pixels = np.asarray(handle["pixels"][tgt_row])
                tgt_qpos = np.asarray(handle["qpos"][tgt_row], dtype=np.float64)
                tgt_qvel = np.asarray(handle["qvel"][tgt_row], dtype=np.float64)
            trials.append(
                {
                    "source_qpos": qpos[qi],
                    "source_qvel": qvel[qi],
                    "target_qpos": tgt_qpos,
                    "target_qvel": tgt_qvel,
                    "target_pixels": tgt_pixels,
                    "target_embedding": graph.z[node],
                    "target_xy": graph_xy[node].astype(np.float64),
                }
            )
            trial_meta.append(
                {
                    "kind": kind,
                    "query_episode": int(episode),
                    "query_step": int(args.episode_step),
                    "graph_node": int(node),
                    **record,
                }
            )

    if not trials:
        raise RuntimeError("No attachment trials could be constructed")

    runner = LocalReachRunner(
        cfg,
        policy,
        max_steps=args.max_steps,
        xy_tolerance=goal_tol,
        base_seed=args.base_seed,
    )
    results = runner.run(trials)

    args.rollout_log.parent.mkdir(parents=True, exist_ok=True)
    with args.rollout_log.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for meta, row in zip(trial_meta, results):
            record = {
                "cosine_distance": float(meta["cosine_distance"]),
                "local_success": bool(row["local_success"]),
                "source_split": "train",
                "source_episode_uid": f"train_{meta['query_episode']}",
                "query_step": meta["query_step"],
                "kind": meta["kind"],
                "graph_node": meta["graph_node"],
                "node_episode": meta["node_episode"],
                "node_step": meta["node_step"],
                "xy_distance": meta["xy_distance"],
                "cell_bfs_distance": meta["cell_bfs_distance"],
                "min_xy_distance": float(row["min_xy_distance"]),
                "min_cosine_distance": float(row["min_cosine_distance"]),
                "state_replay_ok": bool(row["state_replay_ok"]),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))

    positives = sum(1 for r in results if r["local_success"])
    print(f"wrote {len(results)} trials ({positives} positive) -> {args.output}")


if __name__ == "__main__":
    main()
