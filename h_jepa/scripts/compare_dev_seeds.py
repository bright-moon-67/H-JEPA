"""Compare two AntMaze dev-set runs that differ only in the settle env seed.

Both runs must use the same task pack, planner seed and batch grouping; the
only difference is ``eval.env_seed``.  Writes a JSON summary plus a Markdown
table and prints the success-flip accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def row(task: str, a: dict, b: dict) -> dict:
    da, db = a["dist_to_goal_cells"], b["dist_to_goal_cells"]
    return {
        "task": int(task),
        "success_env_seed_a": a["success"],
        "success_env_seed_b": b["success"],
        "flip": (
            None
            if a["success"] == b["success"]
            else ("fail->success" if b["success"] else "success->fail")
        ),
        "steps_a": a["steps_to_success"],
        "steps_b": b["steps_to_success"],
        "class_a": a["classification"]["class"],
        "class_b": b["classification"]["class"],
        "dmin_cells_a": da["min"],
        "dmin_cells_b": db["min"],
        "dfinal_cells_a": da["final"],
        "dfinal_cells_b": db["final"],
        "min_xy_a": a["dist_to_goal_xy"]["min"],
        "min_xy_b": b["dist_to_goal_xy"]["min"],
        "stag_a": a["stagnation"]["longest_steps"],
        "stag_b": b["stagnation"]["longest_steps"],
        "l3_cos_a": a["levels"].get("3", {}).get("subgoal_cos_median"),
        "l3_cos_b": b["levels"].get("3", {}).get("subgoal_cos_median"),
        "l3_rev_a": a["levels"].get("3", {}).get("subgoal_reversals"),
        "l3_rev_b": b["levels"].get("3", {}).get("subgoal_reversals"),
        "anomaly_a": (a.get("earliest_anomaly") or {}).get("step"),
        "anomaly_b": (b.get("earliest_anomaly") or {}).get("step"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-a", required=True)
    ap.add_argument("--summary-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    sa, sb = load(args.summary_a), load(args.summary_b)
    tasks = sorted(sa["tasks"], key=int)
    assert tasks == sorted(sb["tasks"], key=int), "task sets differ"

    rows = [row(t, sa["tasks"][t], sb["tasks"][t]) for t in tasks]
    n = len(rows)
    ok_a = sum(r["success_env_seed_a"] for r in rows)
    ok_b = sum(r["success_env_seed_b"] for r in rows)
    always_ok = [r["task"] for r in rows if r["success_env_seed_a"] and r["success_env_seed_b"]]
    always_fail = [
        r["task"] for r in rows if not r["success_env_seed_a"] and not r["success_env_seed_b"]
    ]
    flips = [r for r in rows if r["flip"]]

    def class_counts(key):
        counts = {}
        for r in rows:
            if r[key] == "success":
                continue
            counts[r[key]] = counts.get(r[key], 0) + 1
        return counts

    out = {
        "format": "antmaze_dev_seed_comparison_v1",
        "env_seed_a": sa.get("trace_dir"),
        "env_seed_b": sb.get("trace_dir"),
        "n_tasks": n,
        "success": {
            "env_seed_a": {"count": ok_a, "rate_pct": 100.0 * ok_a / n},
            "env_seed_b": {"count": ok_b, "rate_pct": 100.0 * ok_b / n},
        },
        "always_success": always_ok,
        "always_failure": always_fail,
        "flips": [
            {
                "task": r["task"],
                "direction": r["flip"],
                "steps_a": r["steps_a"],
                "steps_b": r["steps_b"],
            }
            for r in flips
        ],
        "failure_classes": {
            "env_seed_a": class_counts("class_a"),
            "env_seed_b": class_counts("class_b"),
        },
        "tasks": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))

    lines = [
        "| task | succ A | succ B | flip | class A/B | dmin A/B | dfin A/B | min|xy−g| A/B | stag A/B | L3 cos A/B | L3 rev A/B |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']:02d} | {r['success_env_seed_a']} | {r['success_env_seed_b']} | "
            f"{r['flip'] or '—'} | {r['class_a']}/{r['class_b']} | "
            f"{fmt(r['dmin_cells_a'],0)}/{fmt(r['dmin_cells_b'],0)} | "
            f"{fmt(r['dfinal_cells_a'],0)}/{fmt(r['dfinal_cells_b'],0)} | "
            f"{fmt(r['min_xy_a'])}/{fmt(r['min_xy_b'])} | "
            f"{r['stag_a']}/{r['stag_b']} | "
            f"{fmt(r['l3_cos_a'])}/{fmt(r['l3_cos_b'])} | "
            f"{r['l3_rev_a']}/{r['l3_rev_b']} |"
        )
    table = "\n".join(lines)
    if args.md:
        Path(args.md).write_text(table + "\n")

    print(f"success A={ok_a}/{n} ({100.0*ok_a/n:.1f}%)  B={ok_b}/{n} ({100.0*ok_b/n:.1f}%)")
    print(f"always success: {always_ok}")
    print(f"always failure: {always_fail}")
    print(f"flips: {[(f['task'], f['direction']) for f in out['flips']]}")
    print(f"failure classes A/B: {out['failure_classes']}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
