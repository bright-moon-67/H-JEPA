#!/usr/bin/env python3
"""TRAIN ONLY: derive L2/L3 base & navigation cost scales on same frozen rollouts.

Uses the ORIGINAL LevelCostModel projection for both L2 and L3; sampled
macro-actions from each checkpoint's own latent action queue. No optimization,
no eval tasks, no official/test labels. This is cost-scale calibration only.

Requirement: export_hide_live_train.py saved z[N,384], z2[N,D2], xy[N,2],
episode/source_step, and the associated hashes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from planning_eval import load_model
from hierarchical_solver import build_hierarchical_solver
from hide_nav import NavigationProjector, navigation_match_cost
from hide_integration import sha256


def pair_indices(
    episode: np.ndarray,
    xy: np.ndarray,
    train_episodes: list[int],
    *,
    count: int = 64,
    seed: int = 42,
):
    """Choose train-episode successive pairs with physical movement."""
    train_set = set(train_episodes)
    ix = [
        i for i in range(len(episode)-1)
        if (int(episode[i]) == int(episode[i+1])
            and int(episode[i]) in train_set
            and np.linalg.norm(xy[i+1]-xy[i]) >= 0.35)
    ]
    if len(ix) < count:
        raise ValueError(f"Only {len(ix)} nontrivial TRAIN pairs, need {count}")
    rng = np.random.default_rng(seed)
    return rng.choice(ix,size=count,replace=False).astype(np.int64)


def cost_pairs(cost_model, level, cfg, start, goals, head, *, seed=42, samples=4):
    device=next(cost_model.parameters()).device
    start=torch.from_numpy(start).to(device)
    goals=torch.from_numpy(goals).to(device)
    B=start.shape[0]
    H=int(cfg.hierarchical_plan_config[f"level{level}"].horizon)
    queue=cost_model.get_latent_action_queue()
    if not torch.is_tensor(queue) or queue.ndim!=2 or len(queue)<2:
        raise ValueError(f"Level {level} latent action queue is unavailable")
    queue=queue.detach().to(device)
    gen=torch.Generator(device=device).manual_seed(seed+level)
    indices=torch.randint(len(queue),(B,samples,H),generator=gen,device=device)
    actions=queue[indices].detach()
    info={
        "embed_0":start[:,None,None,:].expand(B,samples,1,-1),
        "goal_embed_0":goals[:,None,None,:].expand(B,samples,1,-1),
        "goal_embed_0_level":3,
        "goal_embed_0_in_upper_space":level==2,
        "dense_upper_goal_projection":False,
        "cost_last_n":1,
        "intermediate_cost_weight":0.0,
        "action_cost_weight":0.0,
    }
    with torch.no_grad():
        original=cost_model.get_cost(dict(info),actions)
        rolled=cost_model.rollout(dict(info),actions)
        pred=rolled["predicted_embed_0"]
        projected,_=cost_model._project_to_goal_space(
            pred,None,
            use_upper_space=(level==2),
            dense=False,goal_level=3
        )
        nav=navigation_match_cost(head,projected,info["goal_embed_0"],cost_last_n=1)
    if original.shape!=(B,samples) or nav.shape!=(B,samples):
        raise ValueError("Calibration cost shape mismatch")
    return original.cpu().numpy().reshape(-1), nav.cpu().numpy().reshape(-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--states",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--head",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num-pairs",type=int,default=64)
    p.add_argument("--candidates",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--config",type=Path,default=Path("config/eval/ant_l3.yaml"))
    args=p.parse_args()
    if args.num_pairs < 8 or args.candidates < 1:
        raise ValueError("Bad calibration budget")
    with np.load(args.states,allow_pickle=False) as f:
        z=np.asarray(f["z"],dtype=np.float32)
        z2=np.asarray(f["z2"],dtype=np.float32)
        ep=np.asarray(f["episode"],dtype=np.int64)
        xy=np.asarray(f["xy"],dtype=np.float32)
        meta=json.loads(str(f["metadata"].item()))
    if meta.get("source_split")!="train" or meta.get("render_domain")!="live_eval_pipeline":
        raise ValueError("Refuse nontrain calibration")
    ckpt=args.checkpoint.resolve()
    if meta.get("checkpoint_sha256")!=sha256(ckpt):
        raise ValueError("Training-state export does not match checkpoint")
    payload=torch.load(args.head,map_location="cpu",weights_only=True)
    if payload["metadata"].get("source_data_sha256") != meta["source_data_sha256"]:
        raise ValueError("Head and cost scale must share the original source dataset")
    ids=pair_indices(ep,xy,payload["metadata"]["train_episodes"],
                     count=args.num_pairs,seed=args.seed)
    head=NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"],strict=True)
    cfg=OmegaConf.load(args.config)
    cfg.policy=str(ckpt)
    model=load_model(cfg)
    solver=build_hierarchical_solver(cfg,model)
    head.to("cuda").eval().requires_grad_(False)
    base3, nav3=cost_pairs(solver.level_solvers[3].model,3,cfg,z[ids],z[ids+1],head,
                            seed=args.seed,samples=args.candidates)
    base2, nav2=cost_pairs(solver.level_solvers[2].model,2,cfg,z2[ids],z[ids+1],head,
                            seed=args.seed,samples=args.candidates)
    calmeta={
        "source_split":"train",
        "calibration":"frozen_rollouts",
        "checkpoint_sha256":sha256(ckpt),
        "normalizer_sha256":meta["normalizer_sha256"],
        "head_sha256":sha256(args.head),
        "num_physical_pairs":len(ids),
        "num_candidates":args.candidates,
        "sampler":"checkpoint_latent_action_queue",
        "macro_action_time_scale":"original_hierarchical_plan_config",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,base_l2=base2,nav_l2=nav2,
                        base_l3=base3,nav_l3=nav3,
                        metadata=np.asarray(json.dumps(calmeta,sort_keys=True)))
    print(f"Saved training calibration pairs {args.output}; {len(base2)} L2 & {len(base3)} L3")


if __name__=="__main__":
    main()
