"""Anchor-grouped paired bootstrap for ARN vs frozen-WM response MSE (P2.5).

The 6 pairs of one anchor are NOT independent samples; resample ANCHORS.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/response_significance.py \
        --data /tmp/pusht_cf_full/test.h5 \
        --head /tmp/action_response_head_full3k.pt \
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
from action_response_wm import get_action_stats, wm_response
from train_action_response import load_checkpoint


def paired_anchor_bootstrap(wm_error, arn_error, anchor_idx, n_boot=2000, seed=42):
    """wm_error/arn_error: per-pair MSE [N]; anchor_idx: [N].

    Returns dict with mean per-anchor MSE gain (WM - ARN; positive = ARN better)
    and its bootstrap 95% CI over anchors.
    """
    improvement = wm_error - arn_error
    anchors, inverse = np.unique(anchor_idx, return_inverse=True)
    counts = np.bincount(inverse)
    anchor_improvement = np.bincount(inverse, weights=improvement) / counts
    anchor_wm = np.bincount(inverse, weights=wm_error) / counts
    anchor_arn = np.bincount(inverse, weights=arn_error) / counts

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(anchors), size=(n_boot, len(anchors)))
    boots = anchor_improvement[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])

    return {
        "mean_mse_gain": float(anchor_improvement.mean()),
        "ci95": (float(lo), float(hi)),
        "n_anchors": int(len(anchors)),
        "wm_mse": float(anchor_wm.mean()),
        "arn_mse": float(anchor_arn.mean()),
        "rel_gain_pct": float(100.0 * anchor_improvement.mean() / max(anchor_wm.mean(), 1e-12)),
        "anchors_arn_better": float((anchor_improvement > 0).mean()),
    }


@torch.no_grad()
def batched_wm(model, z0, a, b, mean, std, device, batch_size=64):
    outs = []
    for k in range(0, len(z0), batch_size):
        outs.append(wm_response(
            model, z0[k:k + batch_size].to(device),
            a[k:k + batch_size].to(device), b[k:k + batch_size].to(device),
            mean, std,
        ).float().cpu())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device)
    mean, std = get_action_stats(model, args.ckpt, device)

    data = load_anchor_h5(args.data)
    z0, z_end = encode_anchors(model, data, device, 16)
    pairs = build_pair_tensors(z0, z_end, data["actions"])
    z0p, a, b, target = pairs["z0"], pairs["a"], pairs["b"], pairs["target"]
    anchor_idx = pairs["anchor_idx"].numpy()
    contacts = (
        torch.as_tensor(data["contacts"])[pairs["anchor_idx"], pairs["branch_i"]]
        + torch.as_tensor(data["contacts"])[pairs["anchor_idx"], pairs["branch_j"]]
    ).numpy()

    wm = batched_wm(model, z0p, a, b, mean, std, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    net = ActionResponseNet(ResponseConfig(**payload["config"])).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    with torch.no_grad():
        arn = net(z0p.to(device), a.to(device), b.to(device)).float().cpu()

    wm_err = (wm - target).square().mean(dim=1).numpy()
    arn_err = (arn - target).square().mean(dim=1).numpy()

    print(f"data={args.data} pairs={len(target)} anchors={len(np.unique(anchor_idx))}")
    print(f"head={args.head} (steps={payload.get('steps')})\n")

    groups = {
        "ALL": np.ones(len(target), dtype=bool),
        "contact": contacts > 0,
        "no-contact": contacts == 0,
    }
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        # anchor_idx is already subset-specific; re-index for this group
        res = paired_anchor_bootstrap(
            wm_err[mask], arn_err[mask], anchor_idx[mask],
            n_boot=args.n_boot, seed=args.seed,
        )
        lo, hi = res["ci95"]
        verdict = "ARN better (CI>0)" if lo > 0 else (
            "WM better (CI<0)" if hi < 0 else "INCONCLUSIVE (CI crosses 0)"
        )
        print(f"[{name}] n_pairs={int(mask.sum())} n_anchors={res['n_anchors']}")
        print(f"  WM anchor-MSE={res['wm_mse']:.5f}  ARN anchor-MSE={res['arn_mse']:.5f}")
        print(f"  mean gain (WM-ARN)={res['mean_mse_gain']:+.5f} "
              f"[95% CI {lo:+.5f}, {hi:+.5f}]  rel={res['rel_gain_pct']:+.2f}%")
        print(f"  anchors where ARN better={res['anchors_arn_better']:.1%}  -> {verdict}\n")


if __name__ == "__main__":
    main()
