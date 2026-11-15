#!/usr/bin/env python3
"""Deterministic paired AntMaze 12 tasks x 2 environment seeds HiDe comparison.

Example:
 python scripts/compare_hide_dev.py \
   --baseline-1000 outputs/b0_env1000 \
   --hide-1000 outputs/hide_env1000 \
   --baseline-2000 outputs/b0_env2000 \
   --hide-2000 outputs/hide_env2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import yaml


def load_eval(path: Path) -> dict:
    with (path/"metrics.yaml").open() as f:
        result=yaml.safe_load(f)
    if len(result["episode_successes"])!=12:
        raise ValueError(f"{path}: requires exactly 12 paired dev tasks")
    if not result.get("seeds") or len(result["seeds"])!=12:
        raise ValueError(f"{path}: missing full deterministic episode seeds")
    record=path/"planning_compute.json"
    result["_compute_seconds"]=None
    if record.is_file():
        records=json.loads(record.read_text())
        result["_compute_seconds"]=sum(
            float(x["wall_clock"]) for x in records
        )
    return result


def compare(b0: dict, hide: dict):
    if b0["seeds"] != hide["seeds"]:
        raise ValueError("Not paired: environment seed/task mapping differs")
    a=np.asarray(b0["episode_successes"],dtype=bool)
    b=np.asarray(hide["episode_successes"],dtype=bool)
    return {
        "baseline_successes":int(a.sum()),
        "hide_successes":int(b.sum()),
        "rescued_task_ids":np.flatnonzero(~a & b).tolist(),
        "destroyed_task_ids":np.flatnonzero(a & ~b).tolist(),
        "rescue":int((~a & b).sum()),
        "destroy":int((a & ~b).sum()),
        "net":int(b.sum()-a.sum()),
        "compute_ratio": (
            hide["_compute_seconds"]/b0["_compute_seconds"]
            if b0["_compute_seconds"] and hide["_compute_seconds"] is not None
            else None
        ),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-1000",type=Path,required=True)
    p.add_argument("--hide-1000",type=Path,required=True)
    p.add_argument("--baseline-2000",type=Path,required=True)
    p.add_argument("--hide-2000",type=Path,required=True)
    args=p.parse_args()
    one=compare(load_eval(args.baseline_1000),load_eval(args.hide_1000))
    two=compare(load_eval(args.baseline_2000),load_eval(args.hide_2000))
    total={
        "baseline_successes":one["baseline_successes"]+two["baseline_successes"],
        "hide_successes":one["hide_successes"]+two["hide_successes"],
        "rescue":one["rescue"]+two["rescue"],
        "destroy":one["destroy"]+two["destroy"],
        "net":one["net"]+two["net"],
        "pairs":{"env1000":one,"env2000":two},
    }
    ratios=[r["compute_ratio"] for r in (one,two)]
    total["max_compute_ratio"]=max(ratios) if all(r is not None for r in ratios) else None
    total["engineering_go"]=bool(
        total["net"]>=2
        and total["destroy"]<=1
        and total["max_compute_ratio"] is not None
        and total["max_compute_ratio"]<=1.5
    )
    print(json.dumps(total,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
