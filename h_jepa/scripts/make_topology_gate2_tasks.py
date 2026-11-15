#!/usr/bin/env python3
"""Gate B: build 12 TRAIN-only local waypoint tasks for the frozen controller.

The gate asks one question: can the UNCHANGED H-JEPA controller reach a
topology waypoint (an adjacent task-space cell) inside a short local budget?

Starts are raw TRAIN frames from episodes that did NOT contribute to the graph
(the graph used episodes 0-149; these are 150-161), so a start is a genuinely
held-out real AntMaze state. Goals are graph cells 3-5 hops away, and the
actual task goal is the adapter's lookahead-1 waypoint on that route.

The payload uses the same format as `generate_antmaze_dev_tasks.py`; the
`xy_neutral` evaluation re-settles and re-renders, so no TRAIN qpos is needed.
Scoring happens later from the trace with REAL simulator XY.

  python scripts/make_topology_gate2_tasks.py \
    --graph artifacts/hide/topology_train_live.npz \
    --dense-train-xy artifacts/hide/dense_train_xy.npz \
    --output artifacts/hide/topology_gate2_tasks.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from topo_graph import load_topology

NUM_STARTS = 12
START_EPISODES = tuple(range(150, 162))  # disjoint from the graph's 0-149
MIN_HOPS, MAX_HOPS = 3, 5
LOOKAHEAD = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--dense-train-xy", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args()

    graph = load_topology(str(args.graph))
    dense = np.load(args.dense_train_xy, allow_pickle=False)
    dense_meta = json.loads(str(dense["metadata"].item()))
    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse non-training HDF5")
    if dense_meta.get("dataset_name") != args.dataset.stem:
        raise ValueError("Start positions must come from the graph's TRAIN dataset")
    with h5py.File(args.dataset, "r") as h:
        ep_offset = np.asarray(h["ep_offset"][:], dtype=np.int64)
        ep_len = np.asarray(h["ep_len"][:], dtype=np.int64)
        start_rows = {e: int(ep_offset[e]) for e in START_EPISODES if e < len(ep_len)}
        start_xy_by_ep = {
            e: np.asarray(h["xy"][r], dtype=np.float64) for e, r in start_rows.items()
        }

    # hop distance between every pair of cells (unit edge weight)
    nodes = len(graph.cells)
    hops = np.full((nodes, nodes), np.inf)
    for src in range(nodes):
        hops[src, src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v, _, _ in graph.adj[u]:
                    if not np.isfinite(hops[src, v]):
                        hops[src, v] = hops[src, u] + 1
                        nxt.append(v)
            frontier = nxt

    tasks, info = [], []
    for k, episode in enumerate(START_EPISODES):
        if len(tasks) >= NUM_STARTS:
            break
        if episode not in start_xy_by_ep:
            print(f"skip episode {episode}: not in dataset")
            continue
        start_xy = start_xy_by_ep[episode]
        node = graph.locate(start_xy, radius=graph.resolution / 2.0, margin=0.5)
        if node is None:
            print(f"skip episode {episode}: start does not attach")
            continue
        candidates = [
            j for j in range(nodes)
            if MIN_HOPS <= hops[node, j] <= MAX_HOPS
        ]
        if not candidates:
            print(f"skip episode {episode}: no goal cell at {MIN_HOPS}-{MAX_HOPS} hops")
            continue
        # deterministic choice: farthest candidate, tie broken by index
        goal_node = max(candidates, key=lambda j: (hops[node, j], -j))
        goal_xy = graph.center(goal_node)
        suggestion = graph.suggest(
            start_xy, goal_xy,
            attachment_radius=graph.resolution / 2.0,
            attachment_margin=0.5,
            lookahead_cells=LOOKAHEAD,
        )
        if suggestion is None:
            print(f"skip episode {episode}: no waypoint suggestion")
            continue
        waypoint_xy = np.asarray(suggestion.waypoint_xy, dtype=np.float64)
        tasks.append(
            {
                "episode": int(episode),
                "start_xy": start_xy.tolist(),
                "goal_node": int(goal_node),
                "waypoint_node": int(suggestion.waypoint_node),
                "waypoint_xy": waypoint_xy.tolist(),
                "route": [int(n) for n in suggestion.nodes],
                "hops_to_goal": float(hops[node, goal_node]),
                "waypoint_distance_from_start": float(
                    np.linalg.norm(waypoint_xy - start_xy)
                ),
            }
        )
        info.append(
            {
                "episode_idx": len(tasks) - 1,
                "train_episode": int(episode),
                "start_xy": start_xy.tolist(),
                "waypoint_node": int(suggestion.waypoint_node),
                "waypoint_xy": waypoint_xy.tolist(),
                "goal_node": int(goal_node),
                "hops_to_goal": float(hops[node, goal_node]),
            }
        )

    if len(tasks) != NUM_STARTS:
        raise SystemExit(
            f"NO-GO: only {len(tasks)}/{NUM_STARTS} usable TRAIN waypoint tasks"
        )

    data = []
    for t in tasks:
        sx, sy = t["start_xy"]
        gx, gy = t["waypoint_xy"]
        xy_pair = torch.tensor([[sx, sy], [gx, gy]], dtype=torch.float64)
        data.append(
            {
                "pixels": torch.zeros((2, 3, 64, 64), dtype=torch.uint8),
                "xy": xy_pair,
                "qpos": torch.zeros((2, 15), dtype=torch.float64),
                "qvel": torch.zeros((2, 14), dtype=torch.float64),
                "proprio": torch.zeros((2, 27), dtype=torch.float32),
                "prev_qpos": torch.zeros((2, 15), dtype=torch.float64),
                "prev_qvel": torch.zeros((2, 14), dtype=torch.float64),
                "success": torch.zeros(2, dtype=torch.float64),
                "init_ij": torch.zeros((2, 2), dtype=torch.int32),
                "goal_ij": torch.zeros((2, 2), dtype=torch.int32),
                "init_xy": torch.tensor([[sx, sy]] * 2, dtype=torch.float32),
                "goal_xy": torch.tensor([[gx, gy]] * 2, dtype=torch.float32),
                "env_name": np.array(["AntMaze"]),
            }
        )

    payload = {
        "format": "ant_dev_grid_trajs_v1",
        "data": data,
        "columns": list(data[0].keys()),
        "episodes_idx": list(range(len(data))),
        "start_steps": [0] * len(data),
        "goal_steps": [1] * len(data),
        "segment_lengths": [2] * len(data),
        "goal_offset_steps": None,
        "eval_budget": 40,
        "process": None,
        "task_info": {
            "maze_type": "medium",
            "d_low": MIN_HOPS,
            "d_high": MAX_HOPS,
            "num_episodes": len(data),
            "goal_cell_set": "train_topology_waypoints",
            "seed": None,
            "generator": "make_topology_gate2_tasks.py",
            "note": "TRAIN-only local waypoint targets; real XY used only for scoring",
            "tasks": info,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "num_tasks": len(tasks),
        "lookahead_cells": LOOKAHEAD,
        "grid_width": graph.resolution,
        "waypoint_distance_from_start": sorted(
            round(t["waypoint_distance_from_start"], 3) for t in tasks
        ),
        "tasks": tasks,
    }
    dest = args.summary or args.output.with_suffix(".summary.json")
    Path(dest).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("num_tasks", "waypoint_distance_from_start")}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
