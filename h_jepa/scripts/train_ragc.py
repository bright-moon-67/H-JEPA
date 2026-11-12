"""Train ONLY RealizedAdvantageHead on already labeled real counterfactual pairs.

Run from h_jepa with PYTHONPATH=.
This script NEVER generates labels. Those must come from REAL simulator
candidate rollouts, preferably with frozen-baseline episode continuations.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ragc import RAGCConfig, RealizedAdvantageHead, paired_realized_advantage_loss
from ragc_data import (
    load_ragc_npz,
    assert_anchor_splits,
    assert_episode_splits,
    to_tensors,
)


def validate(head, tensors, device):
    head.eval()
    with torch.no_grad():
        p = {k: v.to(device) for k, v in tensors.items()}
        qp = head(p["z0"], p["goal"], p["plus"], p["goal_is_upper"])
        qm = head(p["z0"], p["goal"], p["minus"], p["goal_is_upper"])
        diff_t = p["plus_cost"] - p["minus_cost"]
        diff_p = qp - qm
        mse = (diff_t - diff_p).square().mean()
        zero = diff_t.square().mean().clamp_min(1e-9)
        decisive = diff_t.abs() > 1e-4
        accuracy = (
            ((diff_p[decisive] * diff_t[decisive]) > 0).float().mean().item()
            if bool(decisive.any()) else float("nan")
        )
    head.train()
    return {"pair_r2": 1 - mse.item()/zero.item(), "pair_accuracy": accuracy}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_train = load_ragc_npz(args.train)
    data_val = load_ragc_npz(args.val)
    assert_anchor_splits(data_train, data_val)
    assert_episode_splits(data_train, data_val)
    train = to_tensors(data_train)
    val = to_tensors(data_val)

    cfg = RAGCConfig()
    head = RealizedAdvantageHead(cfg).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print("initial:", validate(head, val, args.device), flush=True)
    for step in range(1, args.steps+1):
        idx = torch.randint(len(train["z0"]), (args.batch_size,), generator=generator)
        batch = {k: v[idx].to(args.device) for k, v in train.items()}
        head.train()
        loss, metrics = paired_realized_advantage_loss(
            head, batch["z0"], batch["goal"], batch["plus"], batch["minus"],
            batch["plus_cost"], batch["minus_cost"], batch["goal_is_upper"],
            epsilon=batch["epsilon"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps//5) == 0:
            print(f"step {step}: loss {loss.item():.5f}; val={validate(head, val, args.device)}", flush=True)

    final = validate(head, val, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "ragc_config": asdict(cfg),
        "train_steps": args.steps,
        "validation": final,
        "supervision": "baseline_continuation",
    }, args.out)
    print("saved", args.out, "metrics", final, flush=True)


if __name__ == "__main__":
    main()
