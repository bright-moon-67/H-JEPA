"""Tiny-data overfit check for ActionResponseNet (plan2 §3.2).

Verifies that the head CAN fit a handful of counterfactual pairs. A failure
here means the wiring/data/targets are broken; success says nothing about
generalization.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/action_response_overfit_check.py \
        --data /tmp/pusht_cf/train.h5 --ckpt "$HJEPA_HOME/ckpts/.../pusht_hjepa_l2_object.ckpt" \
        --pairs 16 --steps 800
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from torch.nn import functional as F

from action_response import ActionResponseNet, ResponseConfig
from action_response_data import build_pair_tensors, encode_anchors, load_anchor_h5
from action_response_metrics import compute_response_metrics, format_metrics_row
from train_action_response import encode_rgb, load_checkpoint, prepare_tensors


def load_pairs(path, model, device, encode_batch_size):
    if path.endswith(".h5"):
        data = load_anchor_h5(path)
        z0, z_end = encode_anchors(model, data, device, encode_batch_size)
        return build_pair_tensors(z0, z_end, data["actions"])
    return prepare_tensors(path, model, device, encode_batch_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model = load_checkpoint(args.ckpt, device)
    tensors = load_pairs(args.data, model, device, args.encode_batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n = min(args.pairs, len(tensors["z0"]))
    if n < args.pairs:
        print(f"warning: only {n} pairs available, using all", flush=True)

    z = tensors["z0"][:n].to(device)
    a = tensors["a"][:n].to(device)
    b = tensors["b"][:n].to(device)
    y = tensors["target"][:n].to(device)

    net = ActionResponseNet(
        ResponseConfig(state_dim=z.shape[-1], action_dim=2, horizon=5)
    ).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def evaluate():
        net.eval()
        with torch.no_grad():
            pred = net(z, a, b)
        metrics = compute_response_metrics(pred.cpu(), y.cpu())
        net.train()
        return metrics

    initial = evaluate()
    print("initial " + format_metrics_row("tiny-train", initial), flush=True)
    for step in range(1, args.steps + 1):
        pred = net(z, a, b)
        loss = F.mse_loss(pred, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"step={step:4d} loss={loss.item():.5f}", flush=True)

    final = evaluate()
    print("final   " + format_metrics_row("tiny-train", final), flush=True)

    assert all(torch.isfinite(t).all() for t in net.parameters()), "non-finite params"
    if not final["mse"] < initial["mse"]:
        raise SystemExit(
            f"FAILED to reduce training MSE: {initial['mse']:.5f} -> {final['mse']:.5f}"
        )
    print(f"overfit check passed: mse {initial['mse']:.5f} -> {final['mse']:.5f}; "
          f"R2={final['r2']:+.4f}, cos={final['cosine']:+.4f}, "
          f"mag={final['magnitude_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
