#!/usr/bin/env python3
"""TRAIN-ONLY dense XY transition export for the V1.1 topology graph.

Contract consumed by `build_topology.py`:
  xy[N,2], episode[N], step[N], metadata{"source_split":"train","dataset_name":...}

Every row is a RAW consecutive environment frame (step diff = 1), not the
stride-10 subsample the B1 embedding export uses: the graph may only contain
edges that were physically traversed between adjacent frames.

Episodes are the SAME 150 that produced `live_train_states.npz`, so the dense
positions and the LIVE-rendered representatives come from one data subset.

  python scripts/export_topology_train_xy.py \
    --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
    --output artifacts/hide/dense_train_xy.npz --max-episodes 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=150)
    args = ap.parse_args()
    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse non-training HDF5")
    if args.max_episodes < 12:
        raise ValueError("Need >=12 training episodes")

    with h5py.File(args.dataset, "r") as h:
        ep_len = np.asarray(h["ep_len"][: args.max_episodes], dtype=np.int64)
        ep_offset = np.asarray(h["ep_offset"][: args.max_episodes], dtype=np.int64)
        rows = np.concatenate(
            [np.arange(int(o), int(o) + int(n)) for o, n in zip(ep_offset, ep_len)]
        )
        # h5py fancy indexing needs increasing order; the concatenation is
        # already sorted because episodes are contiguous and ordered.
        if not np.all(np.diff(rows) > 0):
            raise RuntimeError("Episode rows are not strictly increasing")
        xy = np.asarray(h["xy"][rows], dtype=np.float64)
        episode = np.repeat(np.arange(len(ep_len), dtype=np.int64), ep_len)
        step = np.concatenate([np.arange(int(n), dtype=np.int64) for n in ep_len])

    if not (len(xy) == len(episode) == len(step)):
        raise RuntimeError("Axis mismatch")
    if not np.isfinite(xy).all():
        raise ValueError("Non-finite TRAIN xy")
    if any(
        episode[i] > episode[i + 1]
        or (episode[i] == episode[i + 1] and step[i] >= step[i + 1])
        for i in range(len(episode) - 1)
    ):
        raise RuntimeError("Rows must be sorted by (episode, step)")

    meta = {
        "source_split": "train",
        "dataset_name": args.dataset.stem,
        "source_data_sha256": sha256_file(args.dataset),
        "num_episodes": int(len(ep_len)),
        "num_frames": int(len(xy)),
        "frame_stride": 1,
        "note": "raw consecutive TRAIN frames; no dev/val/official task data",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xy=xy.astype(np.float32),
        episode=episode,
        step=step,
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    print(
        json.dumps(
            {k: meta[k] for k in ("num_episodes", "num_frames", "frame_stride")},
            indent=2,
        )
    )
    print(f"Saved {args.output}")
    print(f"rows {len(xy)}  xy range {xy.min(0)} .. {xy.max(0)}")


if __name__ == "__main__":
    main()
