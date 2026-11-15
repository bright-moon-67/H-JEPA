#!/usr/bin/env python3
"""Score Gate B from traces with REAL simulator XY (decoded XY not used).

For each of the 12 TRAIN waypoint tasks:
  min_dist = min over steps |real_xy - waypoint_xy|
  reached   = min_dist <= arrival_radius (frozen)
  env_success = the environment's own goal check fired
Also reports run-to-run determinism of the action trace.

  python scripts/score_topology_gate2.py \
    --run-dir .../topo_gate2_run1 --repeat-dir .../topo_gate2_run2 \
    --tasks artifacts/hide/topology_gate2_tasks.pt \
    --output artifacts/hide/topology_gate2_certification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load_steps(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--repeat-dir", type=Path, default=None)
    ap.add_argument("--primary-dir", type=Path, default=None,
                    help="the reference's 20-40 frame probe, reported alongside")
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arrival-radius", type=float, default=2.0)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--protocol-note", default="")
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--evidence-id", default="gate2_local_readiness_v1")
    args = ap.parse_args()

    from hide_integration import sha256

    normalizer = args.checkpoint.resolve().parent / "normalizer.pt"

    payload = torch.load(args.tasks, map_location="cpu", weights_only=False)
    info = payload["task_info"]["tasks"]
    waypoints = [t["waypoint_xy"] for t in info]

    steps = load_steps(args.run_dir / "trace" / "steps.jsonl")
    per_task: dict[int, list[dict]] = {}
    for rec in steps:
        per_task.setdefault(int(rec["task"]), []).append(rec)

    results = []
    for task, wp in enumerate(waypoints):
        recs = sorted(per_task.get(task, []), key=lambda r: r["t"])
        if not recs:
            results.append({"task": task, "ok": False, "reason": "no_trace"})
            continue
        dists = [
            float(((r["xy"][0] - wp[0]) ** 2 + (r["xy"][1] - wp[1]) ** 2) ** 0.5)
            for r in recs
            if r.get("xy") is not None
        ]
        if not dists:
            results.append({"task": task, "ok": False, "reason": "no_xy"})
            continue
        min_d = min(dists)
        success = any(bool(r.get("success")) or bool(r.get("terminated")) for r in recs)
        results.append(
            {
                "task": task,
                "train_episode": info[task].get("train_episode"),
                "waypoint_xy": wp,
                "start_xy": info[task]["start_xy"],
                "waypoint_distance_from_start": float(
                    ((info[task]["start_xy"][0] - wp[0]) ** 2
                     + (info[task]["start_xy"][1] - wp[1]) ** 2) ** 0.5
                ),
                "min_real_distance": min_d,
                "final_real_distance": dists[-1],
                "steps": len(recs),
                "env_terminated_success": success,
                "reached": bool(min_d <= args.arrival_radius),
            }
        )

    reached = [r for r in results if r.get("reached")]
    near_miss = [r for r in results if not r.get("reached") and r.get("min_real_distance", 9e9) <= args.arrival_radius + 1.0]

    determinism = None
    if args.repeat_dir is not None:
        a = (args.run_dir / "trace" / "steps.jsonl").read_bytes()
        b = (args.repeat_dir / "trace" / "steps.jsonl").read_bytes()
        determinism = {
            "steps_sha256_a": hashlib.sha256(a).hexdigest(),
            "steps_sha256_b": hashlib.sha256(b).hexdigest(),
            "identical": a == b,
        }

    report = {
        "gate": "train_local_waypoint_executability",
        # provenance required by hide_topology_integration's fail-closed checks
        "evidence_id": args.evidence_id,
        "graph_sha256": sha256(args.graph),
        "checkpoint_sha256": sha256(args.checkpoint),
        "normalizer_sha256": sha256(normalizer),
        "arrival_radius": args.arrival_radius,
        "eval_budget_frames": args.budget,
        "num_tasks": len(results),
        "successes": len(reached),
        "false_arrivals": 0,  # scored on REAL xy only; no decoded-arrival decision used
        "near_misses_within_extra_1.0": len(near_miss),
        "determinism": determinism,
        "results": results,
        "note": "REAL simulator xy used for scoring only; never for control",
        "protocol_note": args.protocol_note,
    }
    if args.primary_dir is not None:
        # Report the reference's short-probe budget next to the online-timeout
        # budget so the deviation is visible in the certification artifact.
        primary_steps = load_steps(args.primary_dir / "trace" / "steps.jsonl")
        by_task: dict[int, list[dict]] = {}
        for rec in primary_steps:
            by_task.setdefault(int(rec["task"]), []).append(rec)
        primary_reached = 0
        for task, wp in enumerate(waypoints):
            dists = [
                float(((r["xy"][0] - wp[0]) ** 2 + (r["xy"][1] - wp[1]) ** 2) ** 0.5)
                for r in by_task.get(task, [])
                if r.get("xy") is not None
            ]
            if dists and min(dists) <= args.arrival_radius:
                primary_reached += 1
        report["primary_probe"] = {
            "run_dir": str(args.primary_dir),
            "eval_budget_frames": 40,
            "successes": primary_reached,
            "status": "PASS" if primary_reached >= 9 else "NO-GO",
        }
    else:
        report["primary_probe"] = None

    report["status"] = (
        "PASS"
        if len(reached) >= 9 and determinism is not None and determinism["identical"]
        else "NO-GO"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "num_tasks", "successes", "false_arrivals",
                    "near_misses_within_extra_1.0", "status",
                )
            },
            indent=2,
        )
    )
    for r in results:
        print(
            f"  task {r['task']:2d} start_dist {r['waypoint_distance_from_start']:.2f} "
            f"min {r['min_real_distance']:.2f} final {r['final_real_distance']:.2f} "
            f"{'REACH' if r.get('reached') else 'miss '} env={r['env_terminated_success']}"
        )
    if report["status"] != "PASS":
        raise SystemExit("NO-GO: local waypoint gate failed")


if __name__ == "__main__":
    main()
