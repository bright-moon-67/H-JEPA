#!/usr/bin/env python3
"""Diagnostic only: where does the frozen L3 prediction lose navigation meaning?

The mandatory HiDe audit (`scripts/audit_hide_predictions.py`) scores the
full L3 planning horizon. When it fails, the two possible causes are very
different, and this script separates them WITHOUT tuning anything:

  * readout failure  -> even 1-step predictions decode badly
  * predictor drift  -> 1-step is fine, error grows with the horizon

For every held-out TRAIN window it re-renders the real state at each
`frames_per_l3` boundary and compares, step by step:
  - decoded XY of the frozen L3 rollout vs the true TRAIN XY
  - decoded XY of the ENCODED real state (readout floor)
  - L3 latent cosine(pred, real)
  - motion-direction cosine of the predicted displacement

No planning, no gradient, no dev tasks, no parameter changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf

from data import NORMALIZER_ARTIFACT_FILENAME
from hide_integration import sha256
from hide_nav import NavigationProjector
from make_hide_prediction_pairs import (
    encode_l3, encode_window_actions, l1_action_token_width, render_states,
    temporal_scale,
)
from planning_eval import (
    _build_eval_process_from_policy_normalizer, _build_policy,
    img_transform, load_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=24)
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    norm = ckpt.parent / NORMALIZER_ARTIFACT_FILENAME
    payload = torch.load(args.head, map_location="cpu", weights_only=True)
    head = NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    with np.load(args.pairs, allow_pickle=False) as d:
        pm = json.loads(str(d["metadata"].item()))
        episode = np.asarray(d["episode"], dtype=np.int64)
        l3_index = np.asarray(d["l3_index"], dtype=np.int64)
        frame_start = np.asarray(d["frame_start"], dtype=np.int64)
    if pm.get("checkpoint_sha256") != sha256(ckpt):
        raise ValueError("Pairs were generated from a different checkpoint")
    horizon = int(pm["l3_horizon"])
    frames_per_l3 = int(pm["env_frames_per_l3_step"])

    cfg = OmegaConf.load(args.config)
    cfg.policy = str(ckpt)
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    policy = _build_policy(
        cfg, process, {"pixels": img_transform(cfg), "goal": img_transform(cfg)},
        model=model,
    )
    solver = policy.solver
    cost_model = solver.level_solvers[3].model
    device = next(solver.level_models[2].parameters()).device
    head.to(device).requires_grad_(False)

    pick = np.linspace(0, len(episode) - 1, min(args.max_pairs, len(episode)))
    pick = np.unique(np.round(pick).astype(int))
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    cfg.world.num_envs = args.batch_size
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    world = swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),) * 2)
    try:
        world.set_policy(policy)
        policy.reset()
        seeds = [420042 + i for i in range(args.batch_size)]
        for i, env in enumerate(world.envs.unwrapped.envs):
            env.action_space.seed(int(seeds[i]))
        world.reset(seed=seeds, options=[{} for _ in range(args.batch_size)])

        with h5py.File(args.dataset, "r") as h:
            scale = temporal_scale(model, int(h["action"].shape[-1]),
                                   l1_action_token_width(model))
            if scale["frames_per_l3_token"] != frames_per_l3:
                raise ValueError("Temporal scale disagrees with the pairs file")
            ep_offset = h["ep_offset"][:]
            offset = ep_offset
            jobs = []
            for i in pick:
                base = int(offset[episode[i]] + frame_start[i])
                for step in range(horizon + 1):
                    jobs.append((int(episode[i]), base + step * frames_per_l3))
            z_by_row = {}
            xy_by_row = {}
            for start in range(0, len(jobs), args.batch_size):
                chunk = jobs[start:start + args.batch_size]
                rows = [int(j[1]) for j in chunk]
                qpos = np.stack([h["qpos"][r] for r in rows]).astype(np.float64)
                qvel = np.stack([h["qvel"][r] for r in rows]).astype(np.float64)
                pad = args.batch_size - len(rows)
                if pad:
                    qpos = np.concatenate((qpos, np.repeat(qpos[-1:], pad, 0)), 0)
                    qvel = np.concatenate((qvel, np.repeat(qvel[-1:], pad, 0)), 0)
                pixels, proprio, xy = render_states(world, policy, qpos, qvel)
                z = encode_l3(policy, pixels, proprio)
                for k, row in enumerate(rows):
                    z_by_row[row] = z[k]
                    xy_by_row[row] = xy[k]
                print(f"Encoded {min(start+args.batch_size, len(jobs))}/{len(jobs)}",
                      flush=True)

            raw = []
            for i in pick:
                base = int(offset[episode[i]] + frame_start[i])
                raw.append(
                    np.asarray(
                        h["action"][base: base + horizon * frames_per_l3],
                        dtype=np.float32,
                    )
                )
            raw_t = torch.from_numpy(np.stack(raw)).to(device)
            with torch.no_grad():
                acts = encode_window_actions(model, raw_t)
                start_z = torch.from_numpy(
                    np.stack([
                        z_by_row[int(offset[episode[i]] + frame_start[i])]
                        for i in pick
                    ])
                ).to(device)
                info = {
                    "embed_0": start_z[:, None, None, :],
                    "pixels": torch.zeros(len(pick), 1, 1, 1, 1, 3, device=device),
                }
                rolled = cost_model.rollout(info, acts[:, None])
                preds = rolled["predicted_embed_0"][:, 0]  # [N, H+1, D]
                start_xy = head(start_z)
                per_step = []
                for step in range(horizon + 1):
                    true_xy = np.stack([
                        xy_by_row[int(offset[episode[i]] + frame_start[i])
                                  + step * frames_per_l3]
                        for i in pick
                    ])
                    real_z = torch.from_numpy(np.stack([
                        z_by_row[int(offset[episode[i]] + frame_start[i])
                                 + step * frames_per_l3]
                        for i in pick
                    ])).to(device)
                    pred_z = preds[:, step]
                    pred_xy = head(pred_z)
                    enc_xy = head(real_z)
                    true_t = torch.from_numpy(true_xy).to(device)
                    pred_err = (pred_xy - true_t).norm(dim=-1).cpu().numpy()
                    real_err = (enc_xy - true_t).norm(dim=-1).cpu().numpy()
                    cos = torch.nn.functional.cosine_similarity(
                        pred_z, real_z, dim=-1
                    ).cpu().numpy()
                    d_pred = (pred_xy - start_xy).cpu().numpy()
                    d_true = true_xy - start_xy.cpu().numpy()
                    n_true = np.linalg.norm(d_true, axis=1)
                    n_pred = np.linalg.norm(d_pred, axis=1)
                    mask = n_true >= 0.5
                    mc = (d_pred * d_true).sum(1) / np.maximum(
                        n_true * n_pred, 1e-8
                    )
                    per_step.append(
                        {
                            "step": step,
                            "env_frames": step * frames_per_l3,
                            "median_pred_xy_error": float(np.median(pred_err)),
                            "p90_pred_xy_error": float(np.quantile(pred_err, .9)),
                            "median_real_xy_error": float(np.median(real_err)),
                            "mean_latent_cosine": float(cos.mean()),
                            "mean_motion_cosine": (
                                float(mc[mask].mean()) if mask.sum() else None
                            ),
                            "nontrivial": int(mask.sum()),
                        }
                    )
                    print(json.dumps(per_step[-1]), flush=True)
    finally:
        world.close()

    report = {
        "diagnostic_only": True,
        "note": "no parameter, alpha or candidate-action tuning was performed",
        "num_pairs": int(len(pick)),
        "pairs_sha256": sha256(args.pairs),
        "checkpoint_sha256": sha256(ckpt),
        "env_frames_per_l3_step": frames_per_l3,
        "per_step": per_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
