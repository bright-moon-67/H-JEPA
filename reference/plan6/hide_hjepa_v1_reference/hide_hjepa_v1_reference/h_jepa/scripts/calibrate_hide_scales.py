#!/usr/bin/env python3
"""Calibrate two FIXED HiDe navigation cost scales from FROZEN TRAIN rollouts.

The input is generated on TRAIN episodes using the existing H-JEPA
LevelCostModel and proposed NavigationCostAdapter with alpha excluded.
Both arrays must be measured on the SAME candidate actions / same [B,N].
No use of dev success labels. Scaling = median(base)/median(nav) independently
for L3 and L2. Once generated, do not edit scales on dev set.

Input NPZ arrays:
 base_l2, nav_l2, base_l3, nav_l3: flattened nonnegative costs
 metadata: {"source_split":"train","calibration":"frozen_rollouts",
            "checkpoint_sha256": "...","head_sha256":"...",
            "normalizer_sha256":"..."}.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from hide_integration import sha256


def scale_costs(base: np.ndarray, nav: np.ndarray) -> float:
    base = np.asarray(base,dtype=np.float64).reshape(-1)
    nav = np.asarray(nav,dtype=np.float64).reshape(-1)
    if len(base)<32 or len(nav)!=len(base):
        raise ValueError("Need >=32 paired base/navigation cost samples")
    if not np.isfinite(base).all() or not np.isfinite(nav).all():
        raise ValueError("Nonfinite costs")
    if np.any(base<0) or np.any(nav<0):
        raise ValueError("Expected nonnegative costs")
    med_base=float(np.median(base))
    med_nav=float(np.median(nav))
    if med_base<1e-7 or med_nav<1e-7:
        raise ValueError("Degenerate cost medians; cannot normalize safely")
    scale=med_base/med_nav
    if not 1e-6<scale<1e6:
        raise ValueError("Pathological scale ratio")
    return float(scale)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=Path,required=True)
    p.add_argument("--projector",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    with np.load(args.samples,allow_pickle=False) as f:
        meta=json.loads(str(f["metadata"].item()))
        arrays={k:np.asarray(f[k]) for k in ("base_l2","nav_l2","base_l3","nav_l3")}
    if meta.get("source_split")!="train" or meta.get("calibration")!="frozen_rollouts":
        raise ValueError("V1 cost scaling must use frozen TRAIN rollouts")
    if meta.get("head_sha256")!=sha256(args.projector):
        raise ValueError("Different navigation projector from calibration")
    if not meta.get("checkpoint_sha256") or not meta.get("normalizer_sha256"):
        raise ValueError("Calibration missing model hashes")
    out={
        **meta,
        "scale_l2":scale_costs(arrays["base_l2"], arrays["nav_l2"]),
        "scale_l3":scale_costs(arrays["base_l3"], arrays["nav_l3"]),
        "num_l2":len(arrays["base_l2"].reshape(-1)),
        "num_l3":len(arrays["base_l3"].reshape(-1)),
        "sample_file_sha256":sha256(args.samples),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
