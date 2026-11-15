#!/usr/bin/env python3
"""TRAIN-ONLY frozen-L3 prediction pairs for the mandatory HiDe audit.

The HiDe reference package deliberately leaves this step to the repo agent
(README B3 / AGENT_TASK III), because it needs the checkpoint's real
hierarchical temporal stride, action encoders and TRAIN dataset layout.

What is produced (README contract):
  start_z     [N,384]  live-rendered L3 latent at the window start frame
  pred_z      [N,384]  frozen L3 `rollout` final latent, driven by the TRUE
                       recorded macro actions of that window
  true_end_z  [N,384]  live-rendered L3 latent at the window end frame
  true_end_xy [N,2]    TRAIN env XY at the end frame
  episode     [N]      TRAIN episode ids (must be the head's test_episodes)

Nothing here is a random latent, a linear interpolation, or a GD candidate:
the actions are encoded from the recorded transitions with the checkpoint's
own level-1/2/3 action machinery, and the target is the encoded real future.

Temporal scale (verified against the checkpoint, not assumed):
  level-1 action token  = `frameskip` env frames, derived from the level-1
                          action patch-embed in_channels / action_dim
  level-2 token         = level2.temporal_stride level-1 tokens
  level-3 token         = level3.temporal_stride level-2 tokens
The L3 planner optimizes `hierarchical_plan_config.level3.horizon` macro
actions, and `JEPA.criterion` scores the LAST predicted step (`cost_last_n`),
so the audited horizon is exactly `horizon * frames_per_l3_token` env frames.

Example (from h_jepa, PYTHONPATH=., MUJOCO_GL=egl):
  python scripts/make_hide_prediction_pairs.py \
    --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
    --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
    --head artifacts/hide/nav_projector.pt \
    --output artifacts/hide/prediction_pairs_train_heldout.npz
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
from stable_worldmodel.world import _find_env_method

from data import NORMALIZER_ARTIFACT_FILENAME
from hide_integration import sha256
from planning_eval import (
    _build_eval_process_from_policy_normalizer, _build_policy,
    img_transform, load_model,
)


def l1_action_token_width(model) -> int:
    """Width of one level-1 action token (`frameskip * action_dim`).

    Its patch-embed is a Conv1d over exactly that many channels, so the value
    is recoverable from the checkpoint itself instead of being hard-coded.
    """
    conv = getattr(model.get_level(1).action_embed, "patch_embed", None)
    if conv is None or conv.in_channels is None:
        raise RuntimeError("Cannot derive level-1 action token width")
    return int(conv.in_channels)


def temporal_scale(model, action_dim: int, l1_width: int) -> dict:
    j2, j3 = model.get_level(2), model.get_level(3)
    per_l1 = int(l1_width)
    if per_l1 % int(action_dim) != 0:
        raise ValueError(
            f"Level-1 action token width {per_l1} is not a multiple of "
            f"action_dim {action_dim}"
        )
    frames_l1 = per_l1 // int(action_dim)
    s2 = int(getattr(j2, "temporal_stride", 1))
    s3 = int(getattr(j3, "temporal_stride", 1))
    return {
        "frames_per_l1_token": frames_l1,
        "l1_per_l2_token": s2,
        "l2_per_l3_token": s3,
        "frames_per_l3_token": frames_l1 * s2 * s3,
    }


def select_windows(
    ep_len: np.ndarray,
    xy: np.ndarray,
    ep_offset: np.ndarray,
    episodes: list[int],
    *,
    frames_per_l3: int,
    horizon: int,
    per_episode: int,
    min_displacement: float,
):
    """Pick non-overlapping L3 windows whose REAL displacement is nontrivial."""
    span = frames_per_l3 * horizon
    chosen = []
    for ep in episodes:
        length = int(ep_len[ep])
        offset = int(ep_offset[ep])
        max_k = (length - 1 - span) // frames_per_l3
        if max_k < 0:
            continue
        starts = list(range(0, max_k + 1, horizon))
        pool = starts[:: max(1, len(starts) // max(per_episode, 1))][:per_episode]
        for k in pool:
            f0 = k * frames_per_l3
            f1 = f0 + span
            p0 = np.asarray(xy[offset + f0], dtype=np.float64)
            p1 = np.asarray(xy[offset + f1], dtype=np.float64)
            if np.linalg.norm(p1 - p0) < min_displacement:
                continue
            chosen.append(
                {
                    "episode": int(ep),
                    "l3_index": int(k),
                    "frame_start": int(f0),
                    "frame_end": int(f1),
                    "row_start": int(offset + f0),
                    "row_end": int(offset + f1),
                    "action_row_start": int(offset + f0),
                }
            )
    return chosen


def encode_window_actions(model, raw_actions: torch.Tensor) -> torch.Tensor:
    """True recorded L3 macro actions for a window of env actions.

    raw_actions: [B, horizon*frames_per_l3, action_dim]
    Returns [B, n_l3, 4] using the checkpoint's own level-1 embed, level-2
    action encoder and level-3 action encoder, mirroring
    `HJEPA.encode_hierarchical` exactly (level-1 token summary, then
    `_action_windows` chunking at each upper level).
    """
    j1, j2, j3 = model.get_level(1), model.get_level(2), model.get_level(3)
    conv = getattr(j1.action_embed, "patch_embed", None)
    if conv is None:
        raise RuntimeError("Level-1 action embed has no patch-embed")
    width = int(conv.in_channels)
    action_dim = int(raw_actions.shape[-1])
    if width % action_dim != 0:
        raise ValueError("Level-1 action token width is not a multiple of action_dim")
    frames = width // action_dim
    B, T, A = raw_actions.shape
    if T % frames != 0:
        raise ValueError(f"{T} env actions is not a whole number of {frames}-frame tokens")

    # level-1 token: frameskip raw actions -> one 384-D action embedding
    a1 = j1.action_embed(raw_actions.reshape(B, T // frames, frames * A))
    n1 = a1.shape[1]
    s2 = int(getattr(j2, "temporal_stride", 1))
    starts = torch.arange(0, n1, s2, device=a1.device)
    chunks, mask = j2._action_windows(
        a1, starts=starts, state_window_size=1, stride=s2
    )
    lat2 = j2.action_encoder(chunks, mask)
    a2 = j2.action_embed(lat2)
    n2 = a2.shape[1]
    s3 = int(getattr(j3, "temporal_stride", 1))
    starts3 = torch.arange(0, n2, s3, device=a2.device)
    chunks3, mask3 = j3._action_windows(
        a2, starts=starts3, state_window_size=1, stride=s3
    )
    return j3.action_encoder(chunks3, mask3)


def render_states(
    world,
    policy,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore each row into the genuine World and re-render it live."""
    pixels, proprio, xy = [], [], []
    for i, env in enumerate(world.envs.unwrapped.envs):
        set_state = _find_env_method(env, "set_state")
        get_ob = _find_env_method(env, "get_ob")
        get_xy = _find_env_method(env, "get_xy")
        if any(f is None for f in (set_state, get_ob, get_xy)):
            raise RuntimeError("Missing required World wrapper state/render method")
        set_state(qpos=qpos[i], qvel=qvel[i])
        img = np.asarray(get_ob())
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Expected live HWC RGB pixels, got {img.shape}")
        pixels.append(img)
        xy.append(np.asarray(get_xy(), dtype=np.float32))
        proprio.append(np.concatenate((qpos[i, 2:], qvel[i])).astype(np.float32))
    return (
        np.asarray(pixels),
        np.asarray(proprio),
        np.asarray(xy, dtype=np.float32),
    )


def encode_l3(policy, pixels: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    raw = {
        "pixels": np.asarray(pixels)[:, None],
        "proprio": np.asarray(proprio)[:, None],
    }
    with torch.inference_mode():
        prepared = policy._prepare_info(raw)
        out = policy.solver._encode_all_levels(prepared, key="pixels")
        z3 = out["embed_3"]
    return z3[:, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--min-displacement", type=float, default=1.0)
    ap.add_argument("--max-pairs", type=int, default=96)
    ap.add_argument("--seed", type=int, default=420042)
    args = ap.parse_args()

    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse non-training HDF5")
    if args.batch_size < 1:
        raise ValueError("Invalid batch-size")

    ckpt = args.checkpoint.resolve()
    norm = ckpt.parent / NORMALIZER_ARTIFACT_FILENAME
    if not norm.is_file():
        raise FileNotFoundError(norm)
    payload = torch.load(args.head, map_location="cpu", weights_only=True)
    head_meta = payload["metadata"]
    ckpt_hash = sha256(ckpt)
    if head_meta.get("checkpoint_sha256") != ckpt_hash:
        raise ValueError("Head was trained for a different checkpoint")
    heldout = [int(x) for x in head_meta["test_episodes"]]
    if len(heldout) < 12:
        raise ValueError("Head test split has too few episodes")

    cfg = OmegaConf.load(args.config)
    cfg.policy = str(ckpt)
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    policy = _build_policy(
        cfg, process, {"pixels": img_transform(cfg), "goal": img_transform(cfg)},
        model=model,
    )
    if len(policy.solver.level_models) != 3:
        raise ValueError("Expected a 3-level H-JEPA checkpoint")
    solver = policy.solver
    cost_model = solver.level_solvers[3].model
    horizon = int(cfg.hierarchical_plan_config.level3.horizon)
    cost_last_n = int(cfg.hierarchical_plan_config.level3.get("cost_last_n", 1))
    if cost_last_n != 1:
        raise ValueError("This generator targets the configured cost_last_n=1")

    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    cfg.world.num_envs = args.batch_size
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))

    with h5py.File(args.dataset, "r") as h:
        action_dim = int(h["action"].shape[-1])
        scale = temporal_scale(model, action_dim, l1_action_token_width(model))
        frames_per_l3 = scale["frames_per_l3_token"]
        span = frames_per_l3 * horizon
        windows = select_windows(
            h["ep_len"][:], h["xy"][:], h["ep_offset"][:], heldout,
            frames_per_l3=frames_per_l3, horizon=horizon,
            per_episode=args.per_episode,
            min_displacement=args.min_displacement,
        )
        if len(windows) > args.max_pairs:
            rng = np.random.default_rng(args.seed)
            pick = rng.choice(len(windows), size=args.max_pairs, replace=False)
            windows = [windows[i] for i in sorted(pick)]
        if len(windows) < 32:
            raise SystemExit(
                f"Only {len(windows)} held-out windows with >= {args.min_displacement} "
                "displacement; the audit needs >= 32"
            )

        world = swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),) * 2)
        try:
            world.set_policy(policy)
            policy.reset()
            seeds = [args.seed + i for i in range(args.batch_size)]
            for i, env in enumerate(world.envs.unwrapped.envs):
                env.action_space.seed(int(seeds[i]))
            world.reset(seed=seeds, options=[{} for _ in range(args.batch_size)])

            # --- live-render + encode both endpoints of every window ---
            jobs = []
            for w in windows:
                jobs.append((w["row_start"], w))
                jobs.append((w["row_end"], w))
            z_by_row: dict[int, np.ndarray] = {}
            xy_by_row: dict[int, np.ndarray] = {}
            for start in range(0, len(jobs), args.batch_size):
                chunk = jobs[start:start + args.batch_size]
                rows = [int(j[0]) for j in chunk]
                # h5py fancy indexing needs increasing order; endpoint rows
                # alternate start/end, so read them one by one.
                qpos = np.stack([h["qpos"][r] for r in rows]).astype(np.float64)
                qvel = np.stack([h["qvel"][r] for r in rows]).astype(np.float64)
                pad = args.batch_size - len(rows)
                if pad:
                    qpos = np.concatenate((qpos, np.repeat(qpos[-1:], pad, 0)), 0)
                    qvel = np.concatenate((qvel, np.repeat(qvel[-1:], pad, 0)), 0)
                pixels, proprio, xy = render_states(world, policy, qpos, qvel)
                z = encode_l3(policy, pixels, proprio)
                for i, row in enumerate(rows):
                    z_by_row[row] = z[i]
                    xy_by_row[row] = xy[i]
                print(
                    f"Encoded {min(start+args.batch_size, len(jobs))}/{len(jobs)} "
                    "live endpoint states", flush=True,
                )

            # --- true recorded macro actions + frozen L3 rollout ---
            device = next(solver.level_models[2].parameters()).device
            raw = np.stack(
                [
                    np.asarray(
                        h["action"][
                            w["action_row_start"]: w["action_row_start"] + span
                        ],
                        dtype=np.float32,
                    )
                    for w in windows
                ]
            )
            raw_t = torch.from_numpy(raw).to(device)
            with torch.no_grad():
                acts = encode_window_actions(model, raw_t)
                if acts.shape[1] != horizon:
                    raise RuntimeError(
                        f"Expected {horizon} macro actions, got {acts.shape[1]}"
                    )
                queue = cost_model.get_latent_action_queue()
                start_z = torch.from_numpy(
                    np.stack([z_by_row[w["row_start"]] for w in windows])
                ).to(device)
                info = {
                    "embed_0": start_z[:, None, None, :],
                    "pixels": torch.zeros(
                        len(windows), 1, 1, 1, 1, 3, device=device
                    ),
                }
                rolled = cost_model.rollout(info, acts[:, None])
                predicted = rolled["predicted_embed_0"][:, 0]
                if predicted.shape[1] != horizon + 1:
                    raise RuntimeError(
                        f"Unexpected L3 rollout length {tuple(predicted.shape)}"
                    )
                pred_z = predicted[:, -1]
            true_end_z = np.stack([z_by_row[w["row_end"]] for w in windows])
            true_end_xy = np.stack([xy_by_row[w["row_end"]] for w in windows])
            start_np = start_z.cpu().numpy()
            pred_np = pred_z.float().cpu().numpy()
            if not np.isfinite(pred_np).all():
                raise ValueError("Non-finite L3 predictions")
            act_np = acts.float().cpu().numpy()
            meta = {
                "source_split": "train_heldout",
                "predictor_source": "frozen_l3_true_macro_action_rollout",
                "checkpoint_sha256": ckpt_hash,
                "normalizer_sha256": sha256(norm),
                "head_sha256": sha256(args.head),
                "time_scale_verified": True,
                "time_scale": scale,
                "l3_horizon": horizon,
                "cost_last_n": cost_last_n,
                "env_frames_per_l3_step": frames_per_l3,
                "audited_env_frame_horizon": span,
                "num_pairs": int(len(windows)),
                "heldout_episodes": heldout,
                "action_source": "recorded_hdf5_actions_via_checkpoint_action_encoders",
                "action_queue_std_mean": float(queue.std(dim=0).mean().item()),
                "macro_action_abs_mean": float(np.abs(act_np).mean()),
                "selection": "non_overlapping_L3_windows_min_true_displacement",
                "min_displacement": args.min_displacement,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output,
                start_z=start_np,
                pred_z=pred_np,
                true_end_z=true_end_z,
                true_end_xy=true_end_xy,
                episode=np.asarray([w["episode"] for w in windows], dtype=np.int64),
                l3_index=np.asarray([w["l3_index"] for w in windows], dtype=np.int64),
                frame_start=np.asarray(
                    [w["frame_start"] for w in windows], dtype=np.int64
                ),
                frame_end=np.asarray(
                    [w["frame_end"] for w in windows], dtype=np.int64
                ),
                macro_actions=act_np,
                metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            )
            print(json.dumps(meta, indent=2))
            print(f"Saved {args.output}: {pred_np.shape}")
        finally:
            world.close()


if __name__ == "__main__":
    main()
