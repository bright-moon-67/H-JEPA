#!/usr/bin/env python3
"""TRAIN ONLY: restore qpos/qvel, render through LIVE AntMaze World, encode L3.

Reuses the official World wrapper chain: MegaWrapper -> SyncWorld -> VariationWrapper.
Do NOT reuse the original HDF5 pixels to train the navigation head: PAHRL V1
observed a severe stored-vs-live embedding mismatch.

Example from H-JEPA/h_jepa with PYTHONPATH=. and MUJOCO_GL=egl:
 python scripts/export_hide_live_train.py \
   --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
   --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
   --output artifacts/hide/live_train_states.npz \
   --max-episodes 150 --every 10 --batch-size 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf
from stable_worldmodel.world import _find_env_method
from data import NORMALIZER_ARTIFACT_FILENAME
from planning_eval import (
    _build_eval_process_from_policy_normalizer, _build_policy,
    img_transform, load_model
)


def sha256_file(file: Path) -> str:
    hasher=hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def rows_from_train(episodes: np.ndarray, offsets: np.ndarray,
                    *, max_episodes: int, every: int):
    if every < 1 or max_episodes < 12:
        raise ValueError("Must sample from >=12 episodes and every>=1")
    rows, episode, steps=[], [], []
    for i, length in enumerate(episodes[:max_episodes]):
        for t in range(0, int(length), every):
            rows.append(int(offsets[i]) + t)
            episode.append(i)
            steps.append(t)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(episode, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--max-episodes",type=int,default=150)
    ap.add_argument("--every",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--seed",type=int,default=420042)
    args=ap.parse_args()
    if not args.dataset.stem.endswith("_train"):
        raise ValueError("Refuse non-training HDF5")
    if args.batch_size < 1:
        raise ValueError("Invalid batch-size")
    ckpt=args.checkpoint.resolve()
    norm=ckpt.parent/NORMALIZER_ARTIFACT_FILENAME
    if not norm.is_file():
        raise FileNotFoundError(norm)
    cfg=OmegaConf.load(args.config)
    cfg.policy=str(ckpt)
    model=load_model(cfg)
    process=_build_eval_process_from_policy_normalizer(cfg,model=model)
    policy=_build_policy(
        cfg,process,{"pixels":img_transform(cfg),"goal":img_transform(cfg)},
        model=model
    )
    if len(policy.solver.level_models)!=3:
        raise ValueError("Expected L3 world model")
    cfg.world.max_episode_steps=2*int(cfg.eval.eval_budget)
    cfg.world.num_envs=args.batch_size
    world_cfg=OmegaConf.create(OmegaConf.to_container(cfg.world,resolve=True))
    world=swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),)*2)
    z_parts=[]
    z2_parts=[]
    goal_z_parts=[]
    goal_max_abs=0.0
    xy_parts=[]
    try:
        world.set_policy(policy)
        policy.reset()
        seeds=[args.seed+i for i in range(args.batch_size)]
        for i, env in enumerate(world.envs.unwrapped.envs):
            env.action_space.seed(int(seeds[i]))
        world.reset(seed=seeds,options=[{} for _ in range(args.batch_size)])
        with h5py.File(args.dataset,"r") as h:
            rows,episode,steps=rows_from_train(
                h["ep_len"][:],h["ep_offset"][:],
                max_episodes=args.max_episodes,every=args.every
            )
            for start in range(0,len(rows),args.batch_size):
                current=rows[start:start+args.batch_size]
                # Fixed World batch; pad final batch with one genuine TRAIN row.
                qpos_src=np.asarray(h["qpos"][current],dtype=np.float64)
                qvel_src=np.asarray(h["qvel"][current],dtype=np.float64)
                count=len(current)
                # h5py fancy indexing forbids duplicated final indices.
                # Repeat the last ROW IN MEMORY for a fixed World batch.
                qpos=np.concatenate((
                    qpos_src, np.repeat(qpos_src[-1:], args.batch_size-count, axis=0)
                ),axis=0)
                qvel=np.concatenate((
                    qvel_src, np.repeat(qvel_src[-1:], args.batch_size-count, axis=0)
                ),axis=0)
                pixels=[]
                proprio=[]
                xy=[]
                for i, env in enumerate(world.envs.unwrapped.envs):
                    set_state=_find_env_method(env,"set_state")
                    get_ob=_find_env_method(env,"get_ob")
                    get_xy=_find_env_method(env,"get_xy")
                    if any(f is None for f in (set_state,get_ob,get_xy)):
                        raise RuntimeError("Missing required World wrapper state/render method")
                    set_state(qpos=qpos[i],qvel=qvel[i])
                    img=np.asarray(get_ob())
                    pos=np.asarray(get_xy(),dtype=np.float32)
                    if img.ndim!=3 or img.shape[-1]!=3:
                        raise ValueError(f"Expected live HWC RGB pixels, got {img.shape}")
                    pixels.append(img)
                    xy.append(pos)
                    proprio.append(
                        np.concatenate((qpos[i,2:],qvel[i])).astype(np.float32)
                    )
                raw={
                    "pixels":np.asarray(pixels)[:,None],
                    "proprio":np.asarray(proprio)[:,None],
                    "goal":np.asarray(pixels)[:,None].copy(),
                    "goal_proprio":np.asarray(proprio)[:,None].copy(),
                }
                with torch.inference_mode():
                    prepared=policy._prepare_info(raw)
                    result=policy.solver._encode_all_levels(prepared,key="pixels")
                    goals=policy.solver._encode_all_levels(prepared,key="goal")
                    zz=result["embed_3"]
                    zz2=result["embed_2"]
                    zz_goal=goals["embed_3"]
                if zz.shape[:2]!=(args.batch_size,1) or zz.shape[-1]!=384:
                    raise ValueError(f"L3 embedding mismatch: {tuple(zz.shape)}")
                if zz2.shape[:2]!=(args.batch_size,1):
                    raise ValueError(f"L2 embedding mismatch: {tuple(zz2.shape)}")
                err=(zz-zz_goal).detach().abs().amax().item()
                goal_max_abs=max(goal_max_abs,float(err))
                z_parts.append(zz[:count,0].float().cpu().numpy())
                z2_parts.append(zz2[:count,0].float().cpu().numpy())
                goal_z_parts.append(zz_goal[:count,0].float().cpu().numpy())
                xy_parts.append(np.asarray(xy,dtype=np.float32)[:count,:2])
                if start % (args.batch_size*20)==0:
                    print(f"Encoded {min(start+args.batch_size,len(rows))}/{len(rows)} live TRAIN states",flush=True)
    finally:
        world.close()
    z=np.concatenate(z_parts)
    z2=np.concatenate(z2_parts)
    goal_z=np.concatenate(goal_z_parts)
    xy=np.concatenate(xy_parts)
    if goal_max_abs > 1e-4:
        raise RuntimeError(
            "NO-GO: identical pixels+proprio encoded differently as state vs "
            f"goal, max |ΔL3|={goal_max_abs}. Investigate preprocessing."
        )
    assert len(z)==len(xy)==len(rows)
    if not np.isfinite(z).all() or not np.isfinite(xy).all():
        raise ValueError("Nonfinite exported data")
    meta={
        "source_split":"train",
        "source_dataset":args.dataset.stem,
        "source_data_sha256":sha256_file(args.dataset),
        "render_domain":"live_eval_pipeline",
        "encoder_contract":"WorldModelPolicy._prepare_info -> HierarchicalSolver._encode_all_levels",
        "checkpoint_sha256":sha256_file(ckpt),
        "normalizer_sha256":sha256_file(norm),
        "num_episodes":min(args.max_episodes,len(np.unique(episode))),
        "frame_stride":args.every,
        "num_states":int(len(z)),
        "no_dev_or_official_tasks":True,
        "goal_encoder_parity_passed":True,
        "goal_encoder_max_abs":goal_max_abs,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        args.output,z=z,z2=z2,goal_z=goal_z,xy=xy,episode=episode,
        source_row=rows,source_step=steps,
        metadata=np.asarray(json.dumps(meta,sort_keys=True)),
    )
    print(f"Saved {args.output}: {z.shape}, {xy.shape}")


if __name__=="__main__":
    main()
