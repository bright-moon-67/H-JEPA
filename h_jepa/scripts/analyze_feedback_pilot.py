"""Compare B0 (original 10-step execution) vs B1 (Feedback-5) paired runs.

Reads the two `metrics.yaml` files plus the opt-in solver dumps
(`RAGC_DUMP=...jsonl`) and reports, per task and in aggregate:

  * success, steps_to_success
  * rescued / destroyed / net
  * number of solver calls (planning cost) and the steps they happened at
  * wall-clock evaluation time

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/analyze_feedback_pilot.py \
      --b0-metrics path/to/fb_b0_dev16/metrics.yaml \
      --b1-metrics path/to/fb_b1_dev16/metrics.yaml \
      --b0-dump /tmp/fb_b0.jsonl --b1-dump /tmp/fb_b1.jsonl \
      --out /tmp/feedback_dev/pilot_report.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def load_metrics(path):
    cfg = OmegaConf.load(path)
    d = OmegaConf.to_container(cfg, resolve=True)
    succ = np.asarray(d.get("episode_successes") or [], dtype=bool)
    steps = d.get("steps_to_success") or []
    steps = np.asarray([np.nan if s is None else float(s) for s in steps])
    return d, succ, steps


def load_calls(path):
    per_task = {}
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "solve":
                continue
            per_task.setdefault(int(r["task"]), []).append(int(r["steps_taken"]))
    return {k: sorted(v) for k, v in per_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-metrics", required=True)
    ap.add_argument("--b1-metrics", required=True)
    ap.add_argument("--b0-dump", required=True)
    ap.add_argument("--b1-dump", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m0, s0, t0 = load_metrics(args.b0_metrics)
    m1, s1, t1 = load_metrics(args.b1_metrics)
    c0, c1 = load_calls(args.b0_dump), load_calls(args.b1_dump)
    n = max(len(s0), len(s1))
    if len(s0) != len(s1):
        raise SystemExit(f"task count mismatch: {len(s0)} vs {len(s1)}")

    rows = []
    print(f"{'task':>4} {'B0':>6} {'B1':>6} {'Δ':>3} "
          f"{'B0steps':>8} {'B1steps':>8} {'B0calls':>8} {'B1calls':>8}")
    for i in range(n):
        d = None
        if s0[i] and not s1[i]:
            d = "DESTROY"
        elif s1[i] and not s0[i]:
            d = "RESCUE"
        elif s0[i] and s1[i]:
            d = "both"
        else:
            d = "-"
        row = {
            "task": i,
            "b0_success": bool(s0[i]), "b1_success": bool(s1[i]),
            "b0_steps": None if np.isnan(t0[i]) else float(t0[i]),
            "b1_steps": None if np.isnan(t1[i]) else float(t1[i]),
            "delta": d,
            "b0_calls": len(c0.get(i, [])), "b1_calls": len(c1.get(i, [])),
            "b0_call_steps": c0.get(i, []), "b1_call_steps": c1.get(i, []),
        }
        rows.append(row)
        print(f"{i:>4} {str(row['b0_success']):>6} {str(row['b1_success']):>6} "
              f"{d:>3} {str(row['b0_steps']):>8} {str(row['b1_steps']):>8} "
              f"{row['b0_calls']:>8} {row['b1_calls']:>8}")

    rescued = sum(r["delta"] == "RESCUE" for r in rows)
    destroyed = sum(r["delta"] == "DESTROY" for r in rows)
    both = sum(r["delta"] == "both" for r in rows)
    neither = sum(r["delta"] == "-" for r in rows)

    step_gain = [r for r in rows if r["delta"] == "both"
                 and r["b0_steps"] and r["b1_steps"]]
    faster = sum(r["b1_steps"] < r["b0_steps"] for r in step_gain)
    slower = sum(r["b1_steps"] > r["b0_steps"] for r in step_gain)

    summary = {
        "n_tasks": n,
        "b0_success": int(s0.sum()), "b1_success": int(s1.sum()),
        "rescued": int(rescued), "destroyed": int(destroyed),
        "net": int(rescued - destroyed),
        "both_success": int(both), "both_fail": int(neither),
        "both_success_faster": int(faster), "both_success_slower": int(slower),
        "b0_solver_calls_total": int(sum(len(v) for v in c0.values())),
        "b1_solver_calls_total": int(sum(len(v) for v in c1.values())),
        "b0_eval_seconds": float(m0.get("evaluation_time") or 0),
        "b1_eval_seconds": float(m1.get("evaluation_time") or 0),
        "b0_call_step_histogram": dict(Counter(
            s for v in c0.values() for s in v)),
    }
    print()
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "b0_call_step_histogram"}, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows},
                                         indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
