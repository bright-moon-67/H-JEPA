"""Analyze an AntMaze trace produced by ``antmaze_trace.py``.

Pure CPU / offline: reads the recorded JSONL, reconstructs each task's XY
path, and computes maze-topology-aware diagnostics.

Outputs (under ``--out``):
    summary.json             per-task metrics + heuristic failure class
    figures/overview.png     all traced tasks on the maze
    figures/task_XXX.png     per-task path + distance-to-goal + subgoal cosine
    figures/case_task_XXX.png 3-panel case study (failing tasks by default)

The failure classes are heuristics meant to be read together with the
figures; they never override the raw traces.

    A  route-choice error   (wrong corridor, little progress, no long stall)
    B  local-motion failure (stuck / fallen / blocked near a wall)
    C  planning instability (frequent subgoal reversals with oscillation)
    D  insufficient evidence
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# OGBench locomaze constants (ogbench/locomaze/maze.py: _offset_x/_offset_y=4)
MAZE_OFFSET = 4.0


def xy_to_ij(xy, maze_unit: float):
    i = int((xy[1] + MAZE_OFFSET + 0.5 * maze_unit) / maze_unit)
    j = int((xy[0] + MAZE_OFFSET + 0.5 * maze_unit) / maze_unit)
    return i, j


def ij_to_xy(ij, maze_unit: float):
    i, j = ij
    return j * maze_unit - MAZE_OFFSET, i * maze_unit - MAZE_OFFSET


def bfs_distances(maze_map: np.ndarray, goal_ij) -> np.ndarray:
    """Cell distance to goal over free cells; -1 for walls / unreachable."""
    h, w = maze_map.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if maze_map[goal_ij] != 0:
        # snap to nearest free cell
        free = np.argwhere(maze_map == 0)
        goal_ij = tuple(free[np.argmin(np.abs(free - np.array(goal_ij)).sum(1))])
    dist[goal_ij] = 0
    queue = [goal_ij]
    while queue:
        i, j = queue.pop(0)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and maze_map[ni, nj] == 0 and dist[ni, nj] < 0:
                dist[ni, nj] = dist[i, j] + 1
                queue.append((ni, nj))
    return dist


def snap_cell(maze_map: np.ndarray, ij):
    if 0 <= ij[0] < maze_map.shape[0] and 0 <= ij[1] < maze_map.shape[1] and maze_map[ij] == 0:
        return ij
    free = np.argwhere(maze_map == 0)
    d = np.abs(free - np.array(ij)).sum(1)
    return tuple(free[np.argmin(d)])


def load_trace(trace_dir: Path):
    meta = json.loads((trace_dir / "meta.json").read_text())
    steps = [json.loads(line) for line in (trace_dir / "steps.jsonl").read_text().splitlines()]
    planning = [
        json.loads(line) for line in (trace_dir / "planning.jsonl").read_text().splitlines()
    ]
    done = {}
    if (trace_dir / "trace_done.json").exists():
        done = json.loads((trace_dir / "trace_done.json").read_text())
    return meta, steps, planning, done


def group_by_task(steps, planning):
    steps_by_task = defaultdict(list)
    for rec in steps:
        steps_by_task[rec["task"]].append(rec)
    for recs in steps_by_task.values():
        recs.sort(key=lambda r: r["t"])
    plan_by_task = defaultdict(list)
    for rec in planning:
        for i, task in enumerate(rec["tasks"]):
            per_env = {
                "level": rec["level"],
                "call_idx": rec["call_idx"],
                "steps_taken": rec["steps_taken"],
                "horizon": rec["horizon"],
                "n_iters": None if rec["n_iters"] is None else rec["n_iters"][i],
                "pred_cos_prev": (
                    rec["pred_cos_prev"][i] if rec["pred_cos_prev"] is not None else None
                ),
                "pred_norm_last": (
                    rec["pred_norm_last"][i] if rec["pred_norm_last"] is not None else None
                ),
                "actions": None if rec["actions"] is None else rec["actions"][i],
            }
            plan_by_task[task].append(per_env)
    return steps_by_task, plan_by_task


def detect_stagnation(xy: np.ndarray, threshold: float = 0.05, min_len: int = 10):
    """Return (start_idx, length) of the longest window with tiny displacement."""
    if len(xy) < 2:
        return None, 0
    disp = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    best_start, best_len = None, 0
    run_start, run_len = 0, 0
    for k, d in enumerate(disp):
        if d < threshold:
            if run_len == 0:
                run_start = k
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    if best_len < min_len:
        return None, best_len
    return best_start, best_len


def cos(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def analyze_task(task, step_recs, plan_recs, maze_map, maze_unit, start_ij, goal_ij):
    ts = np.array([r["t"] for r in step_recs])
    xy = np.array([r["xy"] if r["xy"] is not None else [np.nan, np.nan] for r in step_recs])
    qpos = np.array([r["qpos"] if r["qpos"] is not None else [] for r in step_recs])
    success_idx = next((i for i, r in enumerate(step_recs) if r["terminated"] or r["success"]), None)

    dist_map = bfs_distances(maze_map, tuple(goal_ij))
    cells = [snap_cell(maze_map, xy_to_ij(p, maze_unit)) for p in xy]
    dcell = np.array([dist_map[c] if dist_map[c] >= 0 else np.nan for c in cells], dtype=float)

    if success_idx is not None:
        horizon_slice = slice(0, success_idx + 1)
    else:
        horizon_slice = slice(None)
    d_eval = dcell[horizon_slice]
    finite = np.isfinite(d_eval)
    d0 = float(d_eval[finite][0]) if finite.any() else None
    dmin = float(np.nanmin(d_eval)) if finite.any() else None
    dfinal = float(d_eval[finite][-1]) if finite.any() else None
    progress = None if d0 is None or dfinal is None else d0 - dfinal
    progress_ratio = None if not d0 else (progress / d0 if d0 > 0 else 1.0)

    xy_eval = xy[horizon_slice]
    stag_start, stag_len = detect_stagnation(xy_eval)
    disp = np.linalg.norm(np.diff(xy_eval, axis=0), axis=1) if len(xy_eval) > 1 else np.array([])

    # loitering / revisits inside the successful horizon
    cell_counts = Counter(cells[: (success_idx + 1 if success_idx is not None else len(cells))])
    revisit_steps = sum(v - 1 for v in cell_counts.values() if v > 1)

    # fallen-ant heuristic: body height (qpos[2]) sustained below threshold
    z = qpos[:, 2] if qpos.ndim == 2 and qpos.shape[1] > 2 else None
    fallen_steps = int((z < 0.2).sum()) if z is not None else 0
    min_z = float(np.nanmin(z)) if z is not None and len(z) else None

    # planning diagnostics per level
    level_stats = {}
    for level in sorted({p["level"] for p in plan_recs}):
        recs = [p for p in plan_recs if p["level"] == level]
        recs.sort(key=lambda r: r["steps_taken"])
        cosines = [r["pred_cos_prev"] for r in recs if r["pred_cos_prev"] is not None]
        reversals = sum(1 for c in cosines if c < 0.0)
        # change of the level-1 first planned action across replans
        action_change = []
        prev = None
        for r in recs:
            if r["actions"] is None:
                continue
            a = np.asarray(r["actions"], dtype=np.float64).reshape(-1)
            if prev is not None and prev.shape == a.shape:
                denom = max(np.linalg.norm(prev), np.linalg.norm(a), 1e-9)
                action_change.append(float(np.linalg.norm(a - prev) / denom))
            prev = a
        iters = [r["n_iters"] for r in recs if r["n_iters"] is not None]
        level_stats[level] = {
            "n_calls": len(recs),
            "horizons": sorted({r["horizon"] for r in recs if r["horizon"] is not None}),
            "subgoal_cos_median": float(np.median(cosines)) if cosines else None,
            "subgoal_cos_min": float(np.min(cosines)) if cosines else None,
            "subgoal_reversals": int(reversals),
            "action_change_median": (
                float(np.median(action_change)) if action_change else None
            ),
            "n_iters_mean": float(np.mean(iters)) if iters else None,
            "n_iters_max": int(np.max(iters)) if iters else None,
        }

    # earliest anomaly
    anomalies = []
    if stag_start is not None and stag_start > 0:
        anomalies.append((int(stag_start), f"stagnation >= {stag_len} steps (disp<0.05)"))
    if len(d_eval) > 1:
        increases = np.where(np.diff(d_eval) > 0)[0]
        if len(increases) and d0 is not None and d0 > 0:
            # first sustained move away from the goal after any progress
            progressed = np.where(np.diff(d_eval) < 0)[0]
            if len(progressed):
                after = increases[increases > progressed[0]]
                if len(after):
                    anomalies.append((int(after[0]), "distance-to-goal increases after progress"))
    for level, st in level_stats.items():
        if st["subgoal_reversals"]:
            recs = [p for p in plan_recs if p["level"] == level]
            recs.sort(key=lambda r: r["steps_taken"])
            for r in recs:
                if r["pred_cos_prev"] is not None and r["pred_cos_prev"] < 0:
                    anomalies.append(
                        (int(r["steps_taken"]), f"L{level} subgoal reversal (cos<0)")
                    )
                    break
    earliest = min(anomalies) if anomalies else None

    # heuristic class
    if success_idx is not None:
        cls = "success"
        reason = f"reached goal at step {success_idx}"
    elif stag_len >= 30:
        cls = "B"
        reason = f"stuck for {stag_len} consecutive steps (max disp<0.05)"
    elif min_z is not None and fallen_steps >= 20:
        cls = "B"
        reason = f"fallen/low posture for {fallen_steps} steps (min z={min_z:.2f})"
    elif dmin is not None and dmin <= 1 and stag_len >= 10:
        cls = "B"
        reason = "reached the goal cell vicinity but could not finish"
    else:
        l3 = level_stats.get(3, {})
        reversals = l3.get("subgoal_reversals") or 0
        if reversals >= 4 and (progress_ratio is None or progress_ratio < 0.5):
            cls = "C"
            reason = f"{reversals} L3 subgoal reversals with limited progress"
        elif progress_ratio is not None and progress_ratio < 0.15:
            cls = "A"
            reason = "little shortest-path progress along the maze topology"
        else:
            cls = "D"
            reason = "no dominant, clearly observable failure signature"

    return {
        "task": int(task),
        "success": success_idx is not None,
        "steps_to_success": None if success_idx is None else int(success_idx),
        "n_steps": int(len(step_recs)),
        "start_ij": list(start_ij) if start_ij is not None else None,
        "goal_ij": list(goal_ij) if goal_ij is not None else None,
        "dist_to_goal_cells": {
            "initial": d0,
            "min": dmin,
            "final": dfinal,
            "progress": progress,
            "progress_ratio": progress_ratio,
        },
        "stagnation": {
            "longest_steps": int(stag_len),
            "start_step": None if stag_start is None else int(stag_start),
            "mean_disp": float(np.nanmean(disp)) if len(disp) else None,
        },
        "loitering": {
            "distinct_cells": int(len(cell_counts)),
            "revisit_steps": int(revisit_steps),
            "most_visited": cell_counts.most_common(3),
        },
        "posture": {"min_z": min_z, "steps_below_0.2": fallen_steps},
        "levels": level_stats,
        "earliest_anomaly": None if earliest is None else {
            "step": earliest[0],
            "signal": earliest[1],
        },
        "classification": {"class": cls, "reason": reason},
    }


def draw_maze(ax, maze_map, maze_unit):
    h, w = maze_map.shape
    ax.set_xlim(-maze_unit * 1.5, (w - 1) * maze_unit + maze_unit * 1.5)
    ax.set_ylim(-maze_unit * 1.5, (h - 1) * maze_unit + maze_unit * 1.5)
    for i in range(h):
        for j in range(w):
            x, y = ij_to_xy((i, j), maze_unit)
            if maze_map[i, j] != 0:
                ax.add_patch(
                    Rectangle(
                        (x - maze_unit / 2, y - maze_unit / 2),
                        maze_unit,
                        maze_unit,
                        facecolor="#333333",
                        edgecolor="none",
                    )
                )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_task(path, task, step_recs, plan_recs, maze_map, maze_unit, start_ij, goal_ij, detail=False):
    xy = np.array([r["xy"] for r in step_recs])
    ts = np.array([r["t"] for r in step_recs])
    success_idx = next((i for i, r in enumerate(step_recs) if r["terminated"] or r["success"]), None)

    if detail:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]

    ax = axes[0]
    draw_maze(ax, maze_map, maze_unit)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=ts, cmap="viridis", s=8, zorder=3)
    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=0.7, alpha=0.6, zorder=2)
    sx, sy = ij_to_xy(tuple(start_ij), maze_unit)
    gx, gy = ij_to_xy(tuple(goal_ij), maze_unit)
    ax.scatter([sx], [sy], marker="o", s=120, facecolor="none", edgecolor="green", lw=2, zorder=5)
    ax.scatter([gx], [gy], marker="*", s=220, color="red", zorder=5)
    if success_idx is not None:
        ax.scatter(
            [xy[success_idx, 0]], [xy[success_idx, 1]],
            marker="X", s=140, color="lime", edgecolor="black", zorder=6,
        )
    # planning call positions (state at steps_taken)
    calls = {}
    for r in plan_recs:
        calls.setdefault(int(r["steps_taken"]), r["call_idx"])
    idx_by_t = {int(r["t"]): i for i, r in enumerate(step_recs)}
    cx = [xy[idx_by_t[t], 0] for t in calls if t in idx_by_t]
    cy = [xy[idx_by_t[t], 1] for t in calls if t in idx_by_t]
    ax.scatter(cx, cy, s=22, facecolor="none", edgecolor="black", lw=0.8, zorder=4)
    ax.set_title(f"task {task:02d}  {'SUCCESS' if success_idx is not None else 'FAIL'}")

    if detail:
        dist_map = bfs_distances(maze_map, tuple(goal_ij))
        cells = [snap_cell(maze_map, xy_to_ij(p, maze_unit)) for p in xy]
        dcell = np.array([dist_map[c] for c in cells], dtype=float)
        axes[1].plot(ts, dcell, color="tab:red")
        axes[1].set_xlabel("env step")
        axes[1].set_ylabel("maze BFS distance to goal (cells)")
        axes[1].grid(alpha=0.3)
        if success_idx is not None:
            axes[1].axvline(success_idx, color="green", ls="--", lw=1)
        for level, color in ((3, "tab:purple"), (2, "tab:orange"), (1, "tab:blue")):
            recs = sorted([r for r in plan_recs if r["level"] == level], key=lambda r: r["steps_taken"])
            if not recs:
                continue
            xs = [r["steps_taken"] for r in recs if r["pred_cos_prev"] is not None]
            ys = [r["pred_cos_prev"] for r in recs if r["pred_cos_prev"] is not None]
            if xs:
                axes[2].plot(xs, ys, marker="o", ms=3, lw=1, color=color, label=f"L{level}")
        axes[2].axhline(0, color="black", lw=0.8, ls=":")
        axes[2].set_xlabel("env step (planning call)")
        axes[2].set_ylabel("cos(subgoal_t, subgoal_{t-1})")
        axes[2].set_ylim(-1.05, 1.05)
        axes[2].legend(loc="lower right")
        axes[2].grid(alpha=0.3)
        fig.suptitle(
            f"task {task:02d} | {'SUCCESS' if success_idx is not None else 'FAIL'} | "
            f"L3 reversals={sum(1 for r in plan_recs if r['level']==3 and (r['pred_cos_prev'] or 1) < 0)}",
            y=1.02,
        )
        fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02, label="env step")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace directory (meta.json, *.jsonl)")
    ap.add_argument("--out", default=None, help="analysis output directory")
    ap.add_argument("--case-tasks", nargs="*", type=int, default=None,
                    help="tasks to render as 3-panel case studies (default: all failures)")
    args = ap.parse_args()

    trace_dir = Path(args.trace)
    out_dir = Path(args.out) if args.out else trace_dir / "analysis"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta, steps, planning, done = load_trace(trace_dir)
    maze_map = np.asarray(meta["maze_map"])
    maze_unit = float(meta["maze_unit"])
    steps_by_task, plan_by_task = group_by_task(steps, planning)

    tasks = sorted(steps_by_task)
    summary = {
        "trace_dir": str(trace_dir),
        "worker": meta.get("worker"),
        "seed": meta.get("seed"),
        "n_tasks": len(tasks),
        "trace_done": done,
        "tasks": {},
    }

    for task in tasks:
        recs = steps_by_task[task]
        first = recs[0]
        goal_xy = first.get("goal_xy")
        start_xy = first.get("xy")
        start_ij = xy_to_ij(start_xy, maze_unit) if start_xy else None
        goal_ij = xy_to_ij(goal_xy, maze_unit) if goal_xy else None
        result = analyze_task(
            task, recs, plan_by_task.get(task, []), maze_map, maze_unit, start_ij, goal_ij
        )
        summary["tasks"][str(task)] = result
        plot_task(
            fig_dir / f"task_{task:03d}.png",
            task, recs, plan_by_task.get(task, []), maze_map, maze_unit, start_ij, goal_ij,
        )

    # overview grid
    n = len(tasks)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, task in zip(axes, tasks):
        recs = steps_by_task[task]
        xy = np.array([r["xy"] for r in recs])
        first = recs[0]
        start_ij = xy_to_ij(first["xy"], maze_unit)
        goal_ij = xy_to_ij(first["goal_xy"], maze_unit)
        draw_maze(ax, maze_map, maze_unit)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=0.8)
        sx, sy = ij_to_xy(tuple(start_ij), maze_unit)
        gx, gy = ij_to_xy(tuple(goal_ij), maze_unit)
        ax.scatter([sx], [sy], s=60, facecolor="none", edgecolor="green", lw=1.5)
        ax.scatter([gx], [gy], marker="*", s=120, color="red")
        res = summary["tasks"][str(task)]
        ok = res["success"]
        ax.set_title(
            f"task {task:02d}  {'OK' if ok else res['classification']['class']}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "overview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # case studies
    case_tasks = args.case_tasks
    if case_tasks is None:
        case_tasks = [
            t for t in tasks if not summary["tasks"][str(t)]["success"]
        ]
    for task in case_tasks:
        recs = steps_by_task[task]
        first = recs[0]
        start_ij = xy_to_ij(first["xy"], maze_unit)
        goal_ij = xy_to_ij(first["goal_xy"], maze_unit)
        plot_task(
            fig_dir / f"case_task_{task:03d}.png",
            task, recs, plan_by_task.get(task, []), maze_map, maze_unit, start_ij, goal_ij,
            detail=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    fails = [t for t in tasks if not summary["tasks"][str(t)]["success"]]
    print(f"traced tasks: {len(tasks)}  successes: {len(tasks) - len(fails)}  failures: {fails}")
    print(f"{'task':>4} {'cls':>5} {'succ':>5} {'steps':>6} {'d0':>4} {'dmin':>5} "
          f"{'dfin':>5} {'stag':>5} {'L3cos':>6} {'rev':>4}")
    for task in tasks:
        r = summary["tasks"][str(task)]
        d = r["dist_to_goal_cells"]
        l3 = r["levels"].get(3, r["levels"].get("3", {}))
        fmt = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"{task:>4} {r['classification']['class']:>5} {str(r['success']):>5} "
              f"{str(r['steps_to_success']):>6} {fmt(d['initial']):>4} {fmt(d['min']):>5} "
              f"{fmt(d['final']):>5} {r['stagnation']['longest_steps']:>5} "
              f"{fmt(l3.get('subgoal_cos_median')):>6} {l3.get('subgoal_reversals', 0):>4}")
    print(f"summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
