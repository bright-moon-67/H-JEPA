"""Compare Zero / frozen-WM-difference / ActionResponseNet / shuffled-actions.

plan2 P2: is a dedicated counterfactual head worth anything beyond what the
original Level-1 predictor already encodes?

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/compare_response_methods.py \
        --data /tmp/pusht_cf/test.h5 \
        --head /tmp/action_response_head.pt \
        --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt"
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from action_response import ActionResponseNet, ResponseConfig
from action_response_data import build_pair_tensors, encode_anchors, load_anchor_h5
from action_response_metrics import compute_response_metrics, format_metrics_row
from action_response_wm import (
    get_action_stats,
    numeric_consistency_check,
    wm_response,
)
from train_action_response import load_checkpoint, prepare_tensors


def load_pairs(path, model, device, encode_batch_size):
    if path.endswith(".h5"):
        data = load_anchor_h5(path)
        z0, z_end = encode_anchors(model, data, device, encode_batch_size)
        pairs = build_pair_tensors(z0, z_end, data["actions"])
        pairs["contacts"] = torch.as_tensor(data["contacts"])[
            pairs["anchor_idx"], pairs["branch_i"]
        ] + torch.as_tensor(data["contacts"])[pairs["anchor_idx"], pairs["branch_j"]]
        return pairs
    pairs = prepare_tensors(path, model, device, encode_batch_size)
    pairs["contacts"] = torch.zeros(len(pairs["z0"]), dtype=torch.int32)
    return pairs


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = ActionResponseNet(ResponseConfig(**payload["config"])).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


@torch.no_grad()
def batched_wm_response(model, z0, a, b, mean, std, device, batch_size=64):
    outs = []
    for k in range(0, len(z0), batch_size):
        outs.append(wm_response(
            model, z0[k:k + batch_size].to(device),
            a[k:k + batch_size].to(device),
            b[k:k + batch_size].to(device),
            mean, std,
        ).float().cpu())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--shuffle-seed", type=int, default=123)
    p.add_argument("--limit", type=int, default=0, help="cap number of pairs (0=all)")
    p.add_argument("--consistency-atol", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device)
    mean, std = get_action_stats(model, args.ckpt, device)

    pairs = load_pairs(args.data, model, device, args.encode_batch_size)
    z0 = pairs["z0"]
    a = pairs["a"]
    b = pairs["b"]
    target = pairs["target"]
    contacts = pairs["contacts"]
    if args.limit and args.limit < len(z0):
        z0, a, b, target, contacts = (
            z0[:args.limit], a[:args.limit], b[:args.limit],
            target[:args.limit], contacts[:args.limit],
        )
    n = len(z0)

    # ---- 1) numeric consistency gate ----
    if args.data.endswith(".h5"):
        data = load_anchor_h5(args.data)
        m = min(8, len(data["rgb0"]))
        diff, _, _ = numeric_consistency_check(
            model, data["rgb0"][:m], data["actions"][:m, 0],
            mean, std, device, atol=args.consistency_atol,
        )
        print(f"numeric consistency vs JEPA.rollout: max|diff|={diff:.3e} (PASS)")

    net, payload = load_head(args.head, device)
    print(f"head: {args.head} (steps={payload.get('steps')}), pairs={n}")

    # ---- 2) predictions ----
    zero = torch.zeros_like(target)
    arn = net(z0.to(device), a.to(device), b.to(device)).float().cpu()

    gen = torch.Generator().manual_seed(args.shuffle_seed)
    perm = torch.randperm(n, generator=gen)
    same = perm == torch.arange(n)
    if same.any():  # avoid identity matches in the shuffled control
        perm[same] = torch.roll(perm[same], 1)
    arn_shuffled = net(
        z0.to(device),
        a[perm].to(device),
        b[perm].to(device),
    ).float().cpu()

    # ---- 3) table ----
    print()
    print(f"{'method':<24s} {'group':<10s} metrics")
    rows = {
        "zero-baseline": zero,
        "frozen-WM-diff": None,  # filled in below
        "ActionResponseNet": arn,
        "ARN action-shuffled": arn_shuffled,
    }
    for name, pred in rows.items():
        if pred is None:
            pred = batched_wm_response(model, z0.cpu(), a, b, mean, std, device)
            rows[name] = pred
        metrics = compute_response_metrics(pred, target)
        print(format_metrics_row(name, metrics))
        if contacts is not None:
            for label, mask in (
                ("contact", contacts > 0),
                ("no-contact", contacts == 0),
            ):
                if mask.any():
                    m2 = compute_response_metrics(pred[mask], target[mask])
                    print(format_metrics_row(f"{name} [{label}]", m2))

    print()
    print(f"target_rms={compute_response_metrics(target, target)['target_rms']:.4f} "
          f"(dataset action-response signal)")


if __name__ == "__main__":
    main()
