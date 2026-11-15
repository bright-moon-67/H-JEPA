#!/usr/bin/env python3
"""Calibrate query-to-graph attachment using held-out TRAIN-ONLY local rollouts.

Input JSONL (one trial per query) example:
  {"cosine_distance":0.015,"local_success":true,"source_split":"train",
   "source_episode_uid":123}
  {"cosine_distance":0.23,"local_success":false,"source_split":"train",
   "source_episode_uid":456}

These must be actual frozen H-JEPA local-controller trials, not nearest-neighbor
labels or maze-BFS reachability labels. Negative examples must include
physically nearby cross-wall pairs.

No threshold sweep: choose conservative boundary once, then freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def calibrate(rows: list[dict]) -> dict:
    if len(rows)<8:
        raise ValueError("Need >=8 local-controller trials")
    if any(r.get("source_split")!="train" for r in rows):
        raise ValueError("Attachment calibration must be train-only")
    pos=np.array([float(r["cosine_distance"]) for r in rows
                  if r["local_success"]],dtype=np.float64)
    neg=np.array([float(r["cosine_distance"]) for r in rows
                  if not r["local_success"]],dtype=np.float64)
    if len(pos)<3 or len(neg)<3:
        raise ValueError("Need >=3 positive and >=3 negative trials")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("Invalid cosine distances")
    # A conservative separability gate, not an optimization of success labels:
    # all accepted positives must lie strictly below the closest known negative.
    positive_max=float(pos.max())
    negative_min=float(neg.min())
    if positive_max >= negative_min:
        raise ValueError(
            "Local reachability is NOT separable by cosine distance. "
            "Do not enable graph attachment; more threshold tuning is disallowed."
        )
    cutoff=(positive_max+negative_min)/2.
    return {
        "calibration_passed":True,
        "source_split":"train",
        "max_cosine_distance":cutoff,
        "num_trials":len(rows),
        "positive_trials":len(pos),
        "negative_trials":len(neg),
        "positive_max":positive_max,
        "negative_min":negative_min,
        "evidence_id":"train_only_local_rollouts",
        "rule":"midpoint_between_max_positive_and_min_negative",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trials-jsonl",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(x) for x in args.trials_jsonl.read_text().splitlines()
          if x.strip()]
    cert=calibrate(rows)
    cert["evidence_sha256"]=hashlib.sha256(
        args.trials_jsonl.read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(cert,indent=2))
    print(json.dumps(cert,indent=2))


if __name__=="__main__":
    main()
