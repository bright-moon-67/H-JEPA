#!/usr/bin/env python3
"""Independent held-out AntMaze task pack (never used for tuning).

`generate_antmaze_dev_tasks.py` samples (start, goal) cells and only excludes
the official 50-task pack, so a fresh seed still reuses 8/12 of our development
pairs. This script instead ENUMERATES every ordered cell pair at a given BFS
distance and subtracts both exclusion sets before sampling, so disjointness is
guaranteed rather than hoped for.

Payload format is identical to the dev pack (`ant_dev_grid_trajs_v1`); the
environment re-settles and re-renders from (init_xy, goal_xy) at eval time.

  python scripts/make_topology_heldout_tasks.py \
    --dev audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
    --output artifacts/hide/topology_v11_heldout_n12.pt --seed 20261115
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import stable_worldmodel  # noqa: F401  (registers the swm/ gym namespace)


def bfs(maze_map: np.ndarray, start):
    dist = np.full(maze_map.shape, -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if (
                0 <= ni < maze_map.shape[0]
                and 0 <= nj < maze_map.shape[1]
                and maze_map[ni, nj] == 0
                and dist[ni, nj] < 0
            ):
                dist[ni, nj] = dist[i, j] + 1
                q.append((ni, nj))
    return dist


def pairs_from(path: Path | None) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    if path is None or not path.exists():
        return set()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = set()
    for task in payload.get("task_info", {}).get("tasks", []):
        out.add((tuple(task["start_ij"]), tuple(task["goal_ij"])))
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--dev", default="audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt")
    ap.add_argument(
        "--official",
        default="assets/eval_trajs/ant/expert_grid_d3_n50.pt",
        help="official pack that must stay untouched; '' to skip",
    )
    ap.add_argument("--num-episodes", type=int, default=12)
    ap.add_argument("--distances", default="3")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--maze-type", default="medium")
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
    off_x, off_y = float(base._offset_x), float(base._offset_y)
    env.close()

    def ij_to_xy(ij):
        i, j = ij
        return j * unit - off_x, i * unit - off_y

    dev_path = Path(args.dev) if args.dev else None
    off_path = Path(args.official) if args.official else None
    excluded = pairs_from(dev_path) | pairs_from(off_path)

    cells = [tuple(map(int, c)) for c in np.argwhere(maze_map == 0)]
    distances = [int(x) for x in args.distances.split(",") if x.strip()]
    candidates = []
    for start in cells:
        dist = bfs(maze_map, start)
        for d in distances:
            for goal in map(tuple, np.argwhere(dist == d)):
                goal = tuple(map(int, goal))
                if (start, goal) in excluded:
                    continue
                candidates.append((start, goal, d))
    if len(candidates) < args.num_episodes:
        raise SystemExit(
            f"only {len(candidates)} candidate pairs after exclusions; "
            f"need {args.num_episodes}"
        )

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(candidates), size=args.num_episodes, replace=False)
    tasks = [candidates[int(i)] for i in pick]

    data, info = [], []
    for idx, (start, goal, d) in enumerate(tasks):
        sx, sy = (float(v) for v in ij_to_xy(start))
        gx, gy = (float(v) for v in ij_to_xy(goal))
        data.append(
            {
                "pixels": torch.zeros((2, 3, 64, 64), dtype=torch.uint8),
                "xy": torch.tensor([[sx, sy], [gx, gy]], dtype=torch.float64),
                "qpos": torch.zeros((2, 15), dtype=torch.float64),
                "qvel": torch.zeros((2, 14), dtype=torch.float64),
                "proprio": torch.zeros((2, 27), dtype=torch.float32),
                "prev_qpos": torch.zeros((2, 15), dtype=torch.float64),
                "prev_qvel": torch.zeros((2, 14), dtype=torch.float64),
                "success": torch.zeros(2, dtype=torch.float64),
                "init_ij": torch.tensor([list(start)] * 2, dtype=torch.int32),
                "goal_ij": torch.tensor([list(goal)] * 2, dtype=torch.int32),
                "init_xy": torch.tensor([[sx, sy]] * 2, dtype=torch.float32),
                "goal_xy": torch.tensor([[gx, gy]] * 2, dtype=torch.float32),
                "env_name": np.array(["AntMaze"]),
            }
        )
        info.append(
            {
                "episode_idx": idx,
                "grid_distance": int(d),
                "start_ij": list(start),
                "goal_ij": list(goal),
                "requested_start_xy": [sx, sy],
                "requested_goal_xy": [gx, gy],
                "maze_unit": unit,
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
            "d_low": min(distances),
            "d_high": max(distances),
            "num_episodes": len(data),
            "goal_cell_set": "heldout_disjoint_from_dev_and_official",
            "seed": args.seed,
            "generator": "make_topology_heldout_tasks.py",
            "note": (
                "enumerated pairs minus dev/official; never used for any "
                "threshold, calibration or tuning decision"
            ),
            "excluded_dev": str(dev_path) if dev_path else None,
            "excluded_dev_sha256": sha(dev_path) if dev_path and dev_path.exists() else None,
            "excluded_official": str(off_path) if off_path else None,
            "excluded_official_sha256": (
                sha(off_path) if off_path and off_path.exists() else None
            ),
            "num_excluded_pairs": len(excluded),
            "num_candidate_pairs": len(candidates),
            "tasks": info,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    overlap_dev = len({(tuple(t["start_ij"]), tuple(t["goal_ij"])) for t in info} & pairs_from(dev_path))
    overlap_off = len({(tuple(t["start_ij"]), tuple(t["goal_ij"])) for t in info} & pairs_from(off_path))
    print(
        json.dumps(
            {
                "num_tasks": len(info),
                "candidates": len(candidates),
                "excluded_pairs": len(excluded),
                "overlap_with_dev": overlap_dev,
                "overlap_with_official": overlap_off,
                "distances": [t["grid_distance"] for t in info],
            },
            indent=2,
        )
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
