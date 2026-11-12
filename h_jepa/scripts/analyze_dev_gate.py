"""Summarise the RAGC Stage-0B oracle gate from pilot JSON records.

Baseline reference = candidate index 0 (the A* / original-action branch), which
the pilot asserts reproduces the baseline continuation exactly. Realized cost
difference is therefore candidate_cost[i] - candidate_cost[0]; lower is better.

Usage:
    python scripts/analyze_dev_gate.py /tmp/ragc_dev_gate.json [more.json ...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def summarize(paths):
    rows = []
    for p in paths:
        payload = json.loads(Path(p).read_text())
        for r in payload["records"]:
            rows.append(r)
    if not rows:
        print("no records")
        return

    by_task = {}
    for r in rows:
        by_task.setdefault(r["episode"], []).append(r)

    print(f"tasks={len(by_task)} decisions={len(rows)}")
    hdr = f"{'ep':>3} {'step':>4} {'base_ok':>7} {'A*_cost':>8} " \
          f"{'cand_ok':>18} {'dCost':>26}"
    print(hdr)
    recovered, fragile = [], []
    all_dcost, all_ok = [], []
    for ep in sorted(by_task):
        for r in sorted(by_task[ep], key=lambda x: x["step"]):
            costs = np.asarray(r["candidate_costs"], dtype=np.float64)
            oks = list(r["candidate_successes"])
            dc = costs[1:] - costs[0]
            all_dcost.extend(dc.tolist())
            all_ok.extend(oks[1:])
            if (not r["baseline_success"]) and any(oks[1:]) and ep not in recovered:
                recovered.append(ep)
            if r["baseline_success"] and not all(oks[1:]) and ep not in fragile:
                fragile.append(ep)
            print(f"{ep:>3} {r['step']:>4} {str(r['baseline_success']):>7} "
                  f"{costs[0]:>8.4f} {str(oks[1:]):>18} "
                  f"{np.array2string(dc, precision=4, floatmode='fixed'):>26}")

    n_base_fail = len([e for e in by_task
                       if not any(r["baseline_success"] for r in by_task[e])])
    n_base_ok = len(by_task) - n_base_fail
    print()
    print(f"baseline success/failure tasks : {n_base_ok} / {n_base_fail}")
    print(f"recoverable failure tasks      : {len(recovered)} {recovered}")
    print(f"fragile success tasks          : {len(fragile)} {fragile}")
    if all_dcost:
        dc = np.asarray(all_dcost)
        print(f"realized cost diff vs A*       : mean={dc.mean():+.4f} "
              f"median={np.median(dc):+.4f} min={dc.min():+.4f} max={dc.max():+.4f} "
              f"improved={int((dc < -1e-9).sum())}/{len(dc)}")
        print(f"candidate success rate         : {sum(all_ok)}/{len(all_ok)}")
    if n_base_fail:
        print(f"\nGO/NO-GO (>=3 of ~10 dev failures recoverable): "
              f"{'GO' if len(recovered) >= 3 else 'NO-GO'} "
              f"({len(recovered)}/{n_base_fail})")


if __name__ == "__main__":
    summarize(sys.argv[1:])
