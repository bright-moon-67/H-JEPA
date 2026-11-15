#!/usr/bin/env python3
"""Propose (DO NOT CONNECT!) possible inter-trajectory local edges.

Offline training-set XY is permitted ONLY to prioritize controlled validation,
never as online planner input. Every proposed edge still requires local H-JEPA
rollouts before inclusion in the graph.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num-anchors",type=int,default=128)
    p.add_argument("--per-anchor",type=int,default=2)
    p.add_argument("--max-xy-distance",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=420042)
    args=p.parse_args()
    with np.load(args.states,allow_pickle=False) as d:
        z=np.asarray(d["z"],dtype=np.float32)
        ep=d["episode"]
        xy=np.asarray(d["xy_diagnostic_only"],dtype=np.float32)
    if len(z) != len(ep) or xy.shape != (len(z),2):
        raise ValueError("Inconsistent train-only state export")
    rng=np.random.default_rng(args.seed)
    anchors=rng.choice(len(z),size=min(args.num_anchors,len(z)),replace=False)
    norm=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    proposals=[]
    for a in anchors:
        # A spatially close, cross-trajectory candidate is not automatically
        # reachable! Physical check belongs in an independent controlled trial.
        distance=np.linalg.norm(xy-xy[a],axis=1)
        allow=(ep != ep[a])&(distance>0.05)&(
            distance<=args.max_xy_distance
        )
        if not np.any(allow):
            continue
        similarities=norm@norm[a]
        sims=np.where(allow,similarities,-np.inf)
        nearest=np.argsort(-sims,kind="stable")[:args.per_anchor]
        for b in nearest:
            if not np.isfinite(sims[b]):
                continue
            proposals.append({
                "from_node":int(a),"to_node":int(b),
                "latent_cosine_distance":float(max(0.,1.-sims[b])),
                "xy_distance_diagnostic_only":float(distance[b]),
                "status":"UNVERIFIED_DO_NOT_ADD_AS_EDGE",
            })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as f:
        for r in proposals:
            f.write(json.dumps(r,sort_keys=True)+"\n")
    print(f"Wrote {len(proposals)} UNVERIFIED proposals to {args.output}")
    print("Evaluate each direction with frozen H-JEPA local controller.")
    print("Only append successes=trials>=3 to validated_links.jsonl.")


if __name__=="__main__":
    main()
