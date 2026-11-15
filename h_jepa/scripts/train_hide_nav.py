#!/usr/bin/env python3
"""Train a single fixed XY readout from TRAIN-only, LIVE-rendered AntMaze L3 z.

Example:
  PYTHONPATH=. python scripts/train_hide_nav.py \
    --input artifacts/hide/live_train_states.npz \
    --output artifacts/hide/nav_projector.pt --epochs 30 --seed 42

NEVER supply dev/official episodes or HDF5-domain latent cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hide_nav import NavigationProjector


def dataset_splits(episode: np.ndarray, *, seed: int = 42):
    ep = np.unique(episode)
    if len(ep) < 12:
        raise ValueError("Need >=12 distinct training episodes for episode-separated splits")
    rng = np.random.default_rng(seed)
    order = rng.permutation(ep)
    n_train = int(0.70 * len(ep))
    n_val = int(0.15 * len(ep))
    train_ep, val_ep, test_ep = order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]
    tr = np.flatnonzero(np.isin(episode, train_ep))
    va = np.flatnonzero(np.isin(episode, val_ep))
    te = np.flatnonzero(np.isin(episode, test_ep))
    if min(map(len, [tr, va, te])) < 4:
        raise ValueError("Episode split produced too few samples")
    assert not (set(train_ep)&set(val_ep) or set(train_ep)&set(test_ep) or set(val_ep)&set(test_ep))
    return (tr, va, te), (train_ep, val_ep, test_ep)


def error_metrics(est: np.ndarray, true: np.ndarray) -> dict:
    e = np.linalg.norm(est.astype(np.float64) - true.astype(np.float64), axis=-1)
    return {
        "num": int(len(e)),
        "median_xy_error": float(np.median(e)),
        "p90_xy_error": float(np.quantile(e, .90)),
        "mean_xy_error": float(e.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goal-radius", type=float, default=1.0)
    args = ap.parse_args()
    if args.epochs != 30:
        raise ValueError("V1 fixes 30 training epochs. Do not hyperparameter-sweep dev tasks.")
    if args.batch_size < 1 or args.goal_radius <= 0:
        raise ValueError("Bad optimizer settings")

    with np.load(args.input, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        xy = np.asarray(data["xy"], dtype=np.float32)
        episode = np.asarray(data["episode"], dtype=np.int64)
        meta = json.loads(str(data["metadata"].item()))
    if meta.get("source_split") != "train" or meta.get("render_domain") != "live_eval_pipeline":
        raise ValueError("Refuse non-TRAIN or mismatched render-domain data")
    if not meta.get("source_dataset", "").endswith("_train"):
        raise ValueError("Refuse non-training source dataset")
    if meta.get("goal_encoder_parity_passed") is not True:
        raise ValueError("State vs goal encoder parity was not established")
    if not meta.get("checkpoint_sha256") or not meta.get("normalizer_sha256") or not meta.get("source_data_sha256"):
        raise ValueError("Incomplete checkpoint/data provenance")
    if z.ndim != 2 or z.shape[1] != 384 or xy.shape != (len(z),2) or len(episode) != len(z):
        raise ValueError("Expected z[N,384], xy[N,2] and episode[N]")
    if not np.isfinite(z).all() or not np.isfinite(xy).all():
        raise ValueError("Non-finite training data")
    (tr, va, te), split_eps = dataset_splits(episode, seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_xy = xy[tr]
    xy_mean = np.mean(train_xy,axis=0)
    xy_std = np.maximum(np.std(train_xy,axis=0), 1e-3)
    head = NavigationProjector(384, 128, xy_mean, xy_std).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    x = torch.from_numpy(z).to(device)
    y = torch.from_numpy(xy).to(device)
    mean = torch.from_numpy(xy_mean).to(device)
    std = torch.from_numpy(xy_std).to(device)
    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(30):
        head.train()
        order = torch.as_tensor(tr, device=device)[
            torch.randperm(len(tr), device=device)
        ]
        for subset in order.split(args.batch_size):
            pred = head(x[subset])
            loss = ((pred-y[subset])/std).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            val = ((head(x[va])-y[va])/std).square().mean().item()
        if val < best:
            best = val
            best_epoch = epoch + 1
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        vpred = head(x[va]).cpu().numpy()
        tpred = head(x[te]).cpu().numpy()
    report = {
        "source_split": "train",
        "render_domain": "live_eval_pipeline",
        "seed": args.seed,
        "train": int(len(tr)),
        "val": error_metrics(vpred, xy[va]),
        "test": error_metrics(tpred, xy[te]),
        "episode_counts": {
            "train":len(split_eps[0]),"val":len(split_eps[1]),"test":len(split_eps[2]),
        },
        "best_epoch":best_epoch,
        "goal_radius": args.goal_radius,
    }
    # Readout accuracy gate (prediction-latent audit is a separate requirement).
    report["real_state_gate"] = bool(
        report["test"]["median_xy_error"] < args.goal_radius and
        report["test"]["p90_xy_error"] < 2*args.goal_radius
    )
    payload = {
        "architecture": {"latent_dim": 384, "hidden_dim": 128},
        "state_dict": best_state,
        "metadata": {
            "format": "hide_nav_projector_v1",
            "source_split": "train",
            "render_domain": "live_eval_pipeline",
            "source_dataset": meta["source_dataset"],
            "source_data_sha256": meta["source_data_sha256"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "normalizer_sha256": meta["normalizer_sha256"],
            "train_episode_count":len(split_eps[0]),
            "val_episode_count":len(split_eps[1]),
            "test_episode_count":len(split_eps[2]),
            "test_episodes":[int(x) for x in split_eps[2]],
            "train_episodes":[int(x) for x in split_eps[0]],
            "train_seed":args.seed,
            "real_state_gate":report["real_state_gate"],
            "goal_encoder_parity_passed":True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".train_metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report["real_state_gate"]:
        raise SystemExit("NO-GO: real-state projector error gate failed")


if __name__ == "__main__":
    main()
