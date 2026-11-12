#!/usr/bin/env python
"""Generate an independent AntMaze development task pack.

The eval protocol uses ``start_state_mode=goal_state_mode=xy_neutral``: the
physical start state comes from ``settled_info_from_xy(start_xy)`` and the goal
image is re-rendered at ``goal_xy``.  The recorded expert segment is therefore
only a container for the start/goal XY pair, so dev tasks can be built without
the OGBench expert policy.

Tasks are sampled exactly like the official ``expert_grid_d3`` pack:
uniform start cell over all walkable cells, goal cell at BFS grid distance
``--distance``.  Pairs that already appear (in either direction) in the
official pack are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym

import stable_worldmodel.envs  # noqa: F401  (registers swm/*)


def bfs_from(maze_map: np.ndarray, start):
    dist = np.full(maze_map.shape, -1, dtype=np.int32)
    dist[start] = 0
    queue = deque([start])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if (
                0 <= ni < maze_map.shape[0]
                and 0 <= nj < maze_map.shape[1]
                and maze_map[ni, nj] == 0
                and dist[ni, nj] < 0
            ):
                dist[ni, nj] = dist[i, j] + 1
                queue.append((ni, nj))
    return dist


def official_pairs(path: str | None):
    if not path:
        return set()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs = set()
    for task in payload.get("task_info", {}).get("tasks", []):
        if "start_ij" in task and "goal_ij" in task:
            s, g = tuple(task["start_ij"]), tuple(task["goal_ij"])
            pairs.add((s, g))
            pairs.add((g, s))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--distance", type=int, default=3)
    ap.add_argument("--maze-type", default="medium")
    ap.add_argument(
        "--exclude",
        default="assets/eval_trajs/ant/expert_grid_d3_n50.pt",
        help="official pack whose (start,goal) pairs must not be reused; '' to skip",
    )
    args = ap.parse_args()

    env = gym.make(
        "swm/OGBAntMaze-v0",
        maze_type=args.maze_type,
        dataset_type="stitch",
        ob_type="pixels",
        width=64,
        height=64,
        camera_name="back",
        terminate_at_goal=True,
        render_mode="rgb_array",
    )
    base = env.unwrapped
    maze_map = np.asarray(base.maze_map)
    unit = float(base._maze_unit)
    offset_x = float(base._offset_x)
    offset_y = float(base._offset_y)
    env.close()

    def ij_to_xy(ij):
        i, j = ij
        return j * unit - offset_x, i * unit - offset_y

    cells = [tuple(c) for c in np.argwhere(maze_map == 0)]
    forbidden = official_pairs(args.exclude)
    rng = np.random.default_rng(args.seed)

    tasks = []
    seen = set()
    attempts = 0
    while len(tasks) < args.num_episodes:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("could not sample enough unique dev tasks")
        start = cells[int(rng.integers(len(cells)))]
        dist = bfs_from(maze_map, start)
        candidates = [
            tuple(c) for c in np.argwhere(dist == args.distance)
        ]
        if not candidates:
            continue
        goal = candidates[int(rng.integers(len(candidates)))]
        if (start, goal) in forbidden or (start, goal) in seen:
            continue
        seen.add((start, goal))
        tasks.append(
            {
                "start_ij": [int(x) for x in start],
                "goal_ij": [int(x) for x in goal],
            }
        )

    data = []
    task_info = []
    for idx, task in enumerate(tasks):
        sx, sy = (float(v) for v in ij_to_xy(task["start_ij"]))
        gx, gy = (float(v) for v in ij_to_xy(task["goal_ij"]))
        xy = torch.tensor([[sx, sy], [gx, gy]], dtype=torch.float64)
        zeros_px = torch.zeros((2, 3, 64, 64), dtype=torch.uint8)
        ep = {
            "pixels": zeros_px,
            "xy": xy,
            "qpos": torch.zeros((2, 15), dtype=torch.float64),
            "qvel": torch.zeros((2, 14), dtype=torch.float64),
            "proprio": torch.zeros((2, 27), dtype=torch.float32),
            "prev_qpos": torch.zeros((2, 15), dtype=torch.float64),
            "prev_qvel": torch.zeros((2, 14), dtype=torch.float64),
            "success": torch.zeros(2, dtype=torch.float64),
            "init_ij": torch.tensor([task["start_ij"]] * 2, dtype=torch.int32),
            "goal_ij": torch.tensor([task["goal_ij"]] * 2, dtype=torch.int32),
            "init_xy": torch.tensor([[sx, sy]] * 2, dtype=torch.float32),
            "goal_xy": torch.tensor([[gx, gy]] * 2, dtype=torch.float32),
            "env_name": np.array(["AntMaze"]),
        }
        data.append(ep)
        task_info.append(
            {
                "episode_idx": idx,
                "grid_distance": args.distance,
                "start_ij": task["start_ij"],
                "goal_ij": task["goal_ij"],
                "requested_start_xy": [sx, sy],
                "requested_goal_xy": [gx, gy],
                "maze_unit": float(unit),
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
        "eval_budget": 350,
        "process": None,
        "task_info": {
            "maze_type": args.maze_type,
            "d_low": args.distance,
            "d_high": args.distance,
            "num_episodes": len(data),
            "goal_cell_set": "all",
            "seed": args.seed,
            "generator": "generate_antmaze_dev_tasks.py",
            "note": "start/goal cells only; xy_neutral eval re-settles and re-renders",
            "excluded_official_pack": args.exclude,
            "tasks": task_info,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n")
    meta = {
        "file": out.name,
        "sha256": digest,
        "seed": args.seed,
        "distance": args.distance,
        "num_episodes": len(data),
        "maze_type": args.maze_type,
        "tasks": task_info,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    print(f"wrote {out}  sha256={digest}")
    for t in task_info:
        print(f"  task {t['episode_idx']:02d}: {t['start_ij']} -> {t['goal_ij']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
