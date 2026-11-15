#!/usr/bin/env python3
"""Export TRAIN-only AntMaze L3 state embeddings using the EXACT eval encoder path.

HDF format: flat 'pixels', 'qpos', 'qvel', 'xy', 'ep_len', 'ep_offset' datasets.
Never uses eval/validation data to construct graph memory.

Usage from H-JEPA/h_jepa:
 PYTHONPATH=. python scripts/export_pahrl_ant_states.py \
   --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
   --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
   --output artifacts/pahrl/states_train.npz \
   --every 10 --max-episodes 300 --batch-size 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from planning_eval import (
    _build_eval_process_from_policy_normalizer,
    _build_policy,
    img_transform,
    load_model,
)


def sha256_file(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            hash_.update(block)
    return hash_.hexdigest()


def positions_from_hdf(
    lengths: np.ndarray, offsets: np.ndarray, *, every: int, max_episodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if every <= 0:
        raise ValueError("--every must be positive")
    rows, eps, times = [], [], []
    for ep, length in enumerate(lengths[:max_episodes]):
        n = int(length)
        for t in range(0, n, every):
            rows.append(int(offsets[ep]) + t)
            eps.append(ep)
            times.append(t)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(eps, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse graph state export from non-training HDF5")
    if args.max_episodes < 1 or args.batch_size < 1:
        raise ValueError("max-episodes and batch-size must be >= 1")

    cfg = OmegaConf.load(args.config)
    cfg.policy = str(args.checkpoint.resolve())
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    policy = _build_policy(
        cfg, process, {"pixels": img_transform(cfg)}, model=model
    )
    solver = policy.solver
    if len(solver.level_models) != 3:
        raise ValueError("Only H-JEPA L3 is supported")

    all_z = []
    with h5py.File(args.dataset, "r") as f:
        row_idx, episode, steps = positions_from_hdf(
            f["ep_len"][:], f["ep_offset"][:],
            every=args.every, max_episodes=args.max_episodes,
        )
        if row_idx.size < 2:
            raise ValueError("No exportable training states")
        print(f"Exporting {len(row_idx)} TRAIN states")
        for start in range(0, len(row_idx), args.batch_size):
            rows = row_idx[start:start + args.batch_size]
            # h5py fancy-index selection needs sorted indices.
            pixels = np.asarray(f["pixels"][rows])
            qpos = np.asarray(f["qpos"][rows], dtype=np.float32)
            qvel = np.asarray(f["qvel"][rows], dtype=np.float32)
            if pixels.ndim != 4 or pixels.shape[-1] not in (1, 3):
                raise ValueError(f"Expected HWC RGB pixels, got {pixels.shape}")
            if qpos.ndim != 2 or qvel.ndim != 2:
                raise ValueError("Expected row-wise qpos/qvel")
            # Exactly matches ant/base.yaml's qpos[2:] + qvel -> 27D proprio.
            proprio = np.concatenate([qpos[:, 2:], qvel], axis=-1)
            raw = {
                "pixels": pixels[:, None],
                "proprio": proprio[:, None],
            }
            with torch.inference_mode():
                info = policy._prepare_info(raw)
                encoded = solver._encode_all_levels(info, key="pixels")
                z = encoded["embed_3"]
            if z.ndim != 3 or z.shape[1] != 1:
                raise ValueError(f"Bad online-compatible L3 shape {z.shape}")
            all_z.append(z[:, 0].float().cpu().numpy())
            if (start // args.batch_size) % 20 == 0:
                print(f"  encoded {min(start + args.batch_size, len(row_idx))}/{len(row_idx)}")

        xy = np.asarray(f["xy"][row_idx], dtype=np.float32)  # audit only
    z_all = np.concatenate(all_z, axis=0)
    if not np.isfinite(z_all).all():
        raise ValueError("Non-finite encodings")
    normalizer_file = args.checkpoint.parent / "normalizer.pt"
    if not normalizer_file.is_file():
        raise FileNotFoundError(
            "Missing checkpoint-matched normalizer.pt: "
            f"{normalizer_file}"
        )
    meta = {
        "normalizer_sha256": sha256_file(normalizer_file),
        "format": "pahrl_hjepa_state_export_v1",
        "source_split": "train",
        "dataset_name": args.dataset.stem,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_path": str(args.checkpoint),
        "dataset_path": str(args.dataset),
        "every_raw_steps": args.every,
        "num_episodes_exported": args.max_episodes,
        "encoder_contract": "WorldModelPolicy._prepare_info -> HierarchicalSolver._encode_all_levels",
        "eval_config": str(args.config),
        "embed_dim": int(z_all.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, z=z_all, episode=episode, steps=steps,
        source_row=row_idx, xy_diagnostic_only=xy,
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    print("Saved", args.output, z_all.shape)


if __name__ == "__main__":
    main()
