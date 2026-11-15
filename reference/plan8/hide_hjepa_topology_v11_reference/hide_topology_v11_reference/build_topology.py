#!/usr/bin/env python3
"""Offline-only train XY graph builder; neither dev nor official eval tasks.

Prepare two TRAIN-only exports:
  dense_train_xy.npz: xy[N,2], episode[N], step[N], metadata[JSON]
     raw adjacent environment frames, NOT stride-10 sparse samples.
  live_train_states.npz: xy[M,2], embed_1[M,D1], embed_2[M,D2],
     embed_3[M,D3], metadata[JSON]
     must have original *_goal and *_state encoder parity.

The input layout is an adapter contract; adapt the real B1 exporter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from topo_graph import build_observed_topology,save_topology


def load_train_npz(path: Path,required: tuple[str,...]):
    with np.load(path,allow_pickle=False) as f:
        missing=[k for k in required if k not in f]
        if missing: raise ValueError(f"Missing {missing} in {path}")
        out={k:f[k] for k in required}
        meta=json.loads(str(f["metadata"].item()))
    if meta.get("source_split")!="train":
        raise ValueError(f"Train only: {path}")
    return out,meta


def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dense-train-xy",type=Path,required=True)
    p.add_argument("--live-train-states",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--grid-width",type=float,default=1.0)
    p.add_argument("--max-displacement",type=float,default=None)
    p.add_argument("--min-edge-support",type=int,default=1)
    args=p.parse_args()
    dense,meta=load_train_npz(
        args.dense_train_xy,("xy","episode","step")
    )
    live,live_meta=load_train_npz(
        args.live_train_states,("xy","embed_1","embed_2","embed_3")
    )
    ckpt=live_meta.get("checkpoint_sha256")
    norm=live_meta.get("normalizer_sha256")
    if not ckpt or not norm:
        raise ValueError("Live embeddings must carry checkpoint/normalizer SHA")
    if meta.get("dataset_name")!=live_meta.get("dataset_name"):
        raise ValueError("TRAIN dataset mismatch between dense and LIVE export")
    graph=build_observed_topology(
        dense["xy"],dense["episode"],dense["step"],
        resolution=args.grid_width,
        min_edge_support=args.min_edge_support,
        max_displacement_per_step=args.max_displacement,
        source_split="train",
        evidence_id=f"train_dense_sha256:{sha(args.dense_train_xy)}",
        checkpoint_sha256=ckpt,normalizer_sha256=norm,
    )
    graph.attach_live_representatives(
        live["xy"],
        {k:live[k] for k in ("embed_1","embed_2","embed_3")}
    )
    if len(graph.cells)==0 or len(graph.edges)==0 or not len(graph.reps):
        raise ValueError("No usable observed topology / live waypoint payload")
    save_topology(graph,str(args.output))
    status=graph.diagnostic()
    status["discarded_nonlocal_step_transitions"]=graph.discarded_nonlocal_transitions
    status["checkpoint_sha256"]=ckpt
    status["normalizer_sha256"]=norm
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(status,indent=2)
    )
    print(json.dumps(status,indent=2))


if __name__=="__main__":
    main()
