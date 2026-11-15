#!/usr/bin/env python3
"""Gate: can the XY head interpret FROZEN L3 predicted (not only encoded) z?

`pairs.npz` MUST be generated from held-out TRAIN episodes with the ACTUAL
checkpoint and recorded L3 macro-action encoding, not synthetic z or
nearest-neighbor interpolation.

Contract:
  pred_z [N,384]: original frozen L3 rollout predicted end-state latent
  start_z [N,384]: same trajectory's start L3 latent
  true_end_z [N,384]: encoded, LIVE-rendered observed end state
  true_end_xy [N,2]: measured TRAIN env XY at corresponding future timestep
  episode [N]: held-out TRAIN episode IDs
  metadata: JSON with source_split='train_heldout',
            predictor_source='frozen_l3_true_macro_action_rollout',
            checkpoint_sha256, time_scale_verified=True
V1 fail-closed if <32 pairs or offline prediction gate fails.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hide_nav import NavigationProjector
from hide_integration import sha256


def evaluate_prediction_pairs(
    projector: NavigationProjector,
    start_z: np.ndarray,
    predicted_z: np.ndarray,
    actual_end_z: np.ndarray,
    actual_end_xy: np.ndarray,
    *,
    goal_radius: float = 1.0,
) -> dict:
    if min(len(start_z), len(predicted_z), len(actual_end_z), len(actual_end_xy)) < 32:
        raise ValueError("Need >=32 independent held-out prediction pairs")
    if not (start_z.shape == predicted_z.shape == actual_end_z.shape
            and start_z.ndim == 2 and actual_end_xy.shape == (len(start_z),2)):
        raise ValueError("Inconsistent [N,D] and [N,2] arrays")
    if not all(np.isfinite(x).all() for x in [start_z,predicted_z,actual_end_z,actual_end_xy]):
        raise ValueError("Non-finite prediction pairs")
    device = next(projector.parameters()).device
    projector.eval()
    with torch.inference_mode():
        def decode(x):
            return projector(torch.as_tensor(x, device=device, dtype=torch.float32)).cpu().numpy()
        start_xy = decode(start_z)
        pred_xy = decode(predicted_z)
        encoded_end_xy = decode(actual_end_z)
    real_error = np.linalg.norm(encoded_end_xy - actual_end_xy, axis=1)
    pred_error = np.linalg.norm(pred_xy - actual_end_xy, axis=1)
    actual_delta = actual_end_xy - start_xy
    predicted_delta = pred_xy - start_xy
    real_norm = np.linalg.norm(actual_delta, axis=1)
    pred_norm = np.linalg.norm(predicted_delta, axis=1)
    mask = real_norm >= 0.5
    denom = np.maximum(real_norm*pred_norm, 1e-8)
    cosine = (actual_delta*predicted_delta).sum(axis=1)/denom
    if mask.sum() < 12:
        raise ValueError("Not enough NONTRIVIAL movement in prediction pairs")
    report = {
        "num_pairs":len(pred_error),
        "nontrivial_pairs":int(mask.sum()),
        "median_real_xy_error":float(np.median(real_error)),
        "median_pred_xy_error":float(np.median(pred_error)),
        "p90_pred_xy_error":float(np.quantile(pred_error,.90)),
        "mean_motion_direction_cosine":float(np.mean(cosine[mask])),
        "prediction_vs_real_median_ratio":float(
            np.median(pred_error)/max(float(np.median(real_error)),.05)
        ),
    }
    report["status"] = (
        "PASS" if (
            report["median_pred_xy_error"] < goal_radius
            and report["p90_pred_xy_error"] < 2*goal_radius
            and report["mean_motion_direction_cosine"] >= 0.5
        ) else "NO-GO"
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs",type=Path,required=True)
    ap.add_argument("--head",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--goal-radius",type=float,default=1.0)
    args=ap.parse_args()
    payload=torch.load(args.head,map_location="cpu",weights_only=True)
    head=NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"],strict=True)
    with np.load(args.pairs,allow_pickle=False) as data:
        meta=json.loads(str(data["metadata"].item()))
        start=np.asarray(data["start_z"],dtype=np.float32)
        pred=np.asarray(data["pred_z"],dtype=np.float32)
        end=np.asarray(data["true_end_z"],dtype=np.float32)
        xy=np.asarray(data["true_end_xy"],dtype=np.float32)
        episodes=np.asarray(data["episode"],dtype=np.int64)
    if meta.get("source_split")!="train_heldout" or meta.get("predictor_source")!="frozen_l3_true_macro_action_rollout":
        raise ValueError("Only genuine held-out TRAIN L3 predictions are permitted")
    if meta.get("time_scale_verified") is not True:
        raise ValueError("Macro-action/upper temporal stride must be verified")
    if meta.get("checkpoint_sha256")!=payload["metadata"]["checkpoint_sha256"]:
        raise ValueError("Prediction pairs and projector use different checkpoints")
    heldout_eps=set(payload["metadata"]["test_episodes"])
    if len(episodes)!=len(pred) or not set(episodes.tolist()).issubset(heldout_eps):
        raise ValueError("Prediction pairs are not from the frozen projector TEST episode split")
    report=evaluate_prediction_pairs(
        head,start,pred,end,xy,goal_radius=args.goal_radius
    )
    report.update({
        "source_split":"train_heldout",
        "head_sha256":sha256(args.head),
        "checkpoint_sha256":payload["metadata"]["checkpoint_sha256"],
        "pairs_sha256":sha256(args.pairs),
    })
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if report["status"]!="PASS":
        raise SystemExit("NO-GO: predicted L3 latent navigation-readout gate failed")


if __name__=="__main__":
    main()
