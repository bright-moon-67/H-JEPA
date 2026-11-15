#!/usr/bin/env python3
"""Build the V1.1 TRAIN-only observed topology with LIVE all-level waypoints.

Thin wrapper over the reference core (`topo_graph.build_observed_topology` /
`attach_live_representatives` / `save_topology`). It exists because the
reference CLI cannot express the one thing this task space needs:

GRID GEOMETRY (frozen from TRAIN evidence, see the audit report):
  * rounding TRAIN xy to the nearest multiple of 4 yields exactly 26 unique
    (x, y) centres on a 6x6 lattice {0,4,8,12,16,20}, i.e. the task space is
    a 4.0-pitch cell lattice;
  * consecutive-frame displacement is p99.9 = 0.234 / max = 0.307, so a
    1.0-unit teleport rejection threshold is >=3x the fastest real motion.

Therefore: resolution = 4.0, origin = (-2, -2) so that each bin is centred on
a lattice point (= one task-space cell). With width 0.0-free bins aligned to
cell centres, pooling different episodes inside a bin cannot join opposite
sides of a wall: the bin IS the maze cell. The reference default origin (0,0)
would instead put bin boundaries through cell centres.

  python scripts/build_topology_hjepa.py \
    --dense-train-xy artifacts/hide/dense_train_xy.npz \
    --live-train-states artifacts/hide/live_train_states_v11.npz \
    --output artifacts/hide/topology_train_live.npz \
    --grid-width 4.0 --max-displacement 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from topo_graph import build_observed_topology, save_topology


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_train(path: Path, required):
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in required if k not in f]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")
        out = {k: f[k] for k in required}
        meta = json.loads(str(f["metadata"].item()))
    if meta.get("source_split") != "train":
        raise ValueError(f"Train only: {path}")
    return out, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense-train-xy", type=Path, required=True)
    p.add_argument("--live-train-states", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid-width", type=float, required=True)
    p.add_argument("--max-displacement", type=float, required=True)
    p.add_argument("--min-edge-support", type=int, default=1)
    args = p.parse_args()

    dense, dense_meta = load_train(args.dense_train_xy, ("xy", "episode", "step"))
    live, live_meta = load_train(
        args.live_train_states,
        ("xy", "embed_1", "embed_2", "embed_3"),
    )
    ckpt = live_meta.get("checkpoint_sha256")
    norm = live_meta.get("normalizer_sha256")
    if not ckpt or not norm:
        raise ValueError("Live embeddings must carry checkpoint/normalizer SHA")
    if dense_meta.get("dataset_name") != live_meta.get("dataset_name"):
        raise ValueError("TRAIN dataset mismatch between dense and LIVE export")
    if dense_meta.get("frame_stride") != 1:
        raise ValueError("Dense export must contain consecutive frames")
    if live_meta.get("all_level_goal_parity_passed") is not True:
        raise ValueError("All-level state/goal encoder parity must be established")

    width = float(args.grid_width)
    origin = (-width / 2.0, -width / 2.0)
    graph = build_observed_topology(
        dense["xy"],
        dense["episode"],
        dense["step"],
        resolution=width,
        origin=origin,
        min_edge_support=args.min_edge_support,
        max_displacement_per_step=args.max_displacement,
        source_split="train",
        evidence_id=f"train_dense_sha256:{sha(args.dense_train_xy)}",
        checkpoint_sha256=ckpt,
        normalizer_sha256=norm,
    )
    dims = graph.attach_live_representatives(
        live["xy"],
        {k: live[k] for k in ("embed_1", "embed_2", "embed_3")},
    )
    if len(graph.cells) == 0 or len(graph.edges) == 0 or not len(graph.reps):
        raise ValueError("No usable observed topology / live waypoint payload")
    save_topology(graph, str(args.output))

    # every node must be reachable from at least one other node (no isolated
    # bins) and every edge keeps its observed support
    degrees = np.zeros(len(graph.cells), dtype=np.int64)
    for a, b in graph.edges:
        degrees[int(a)] += 1
        degrees[int(b)] += 1
    status = graph.diagnostic()
    status.update(
        {
            "grid_width": width,
            "origin": list(origin),
            "max_displacement_per_step": float(args.max_displacement),
            "min_edge_support": int(args.min_edge_support),
            "discarded_nonlocal_step_transitions": int(
                graph.discarded_nonlocal_transitions
            ),
            "support_total": int(graph.support.sum()),
            "support_min": int(graph.support.min()),
            "support_median": float(np.median(graph.support)),
            "isolated_nodes": int((degrees == 0).sum()),
            "nodes_with_rep": int(len(graph.reps)),
            "rep_coverage": float(len(graph.reps) / len(graph.cells)),
            "embed_dims": dims,
            "checkpoint_sha256": ckpt,
            "normalizer_sha256": norm,
            "dense_train_xy_sha256": sha(args.dense_train_xy),
            "live_train_states_sha256": sha(args.live_train_states),
        }
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
