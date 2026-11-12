"""Test A: is the head's DIRECTIONAL DERIVATIVE at A* consistent with reality?

Training supervises the secant
    (Q(A+) - Q(A-)) / (2 eps)  vs  (C(A+) - C(A-)) / (2 eps)
but `RAGCRefiner` would descend on  grad_A Q(A*).
Those two are only equivalent for a locally linear Q, which is exactly what a
success/failure jump breaks.

This script checks, per pair, whether
    sign( grad_A Q(A*) . u )  ==  sign( C(A+) - C(A-) )
where u is the unit perturbation direction of that pair.

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/ragc_gradient_check.py \
      --ckpt /tmp/ragc_merged/head.pt \
      --val /tmp/ragc_merged/val.npz [--train /tmp/ragc_merged/train.npz]
"""
import argparse

import numpy as np
import torch

from ragc import RAGCConfig, RealizedAdvantageHead
from ragc_data import load_ragc_npz


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RAGCConfig(**payload["ragc_config"])
    head = RealizedAdvantageHead(cfg).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head


def directional(diff_p, diff_t, label):
    dec = np.abs(diff_t) > 1e-4
    if not dec.any():
        print(f"  {label:<26} no decisive pairs")
        return
    agree = np.mean(np.sign(diff_p[dec]) == np.sign(diff_t[dec]))
    print(f"  {label:<26} n={int(dec.sum()):<3} "
          f"direction_agree={agree:.3f} "
          f"(chance 0.500)  |grad|median="
          f"{np.median(np.abs(diff_p[dec])):.4f}")


def run(path, head, device, big):
    d = load_ragc_npz(path)
    z0 = torch.as_tensor(d["z0"], dtype=torch.float32, device=device)
    goal = torch.as_tensor(d["goal"], dtype=torch.float32, device=device)
    upper = torch.as_tensor(d["goal_is_upper"], dtype=torch.bool, device=device)
    astar = torch.as_tensor(d["astar_actions"], dtype=torch.float32,
                            device=device)  # [N,2,10]
    plus = d["plus"]
    minus = d["minus"]
    dt = d["plus_cost"] - d["minus_cost"]
    base_ok = d["baseline_success"].astype(bool)
    has_a = d["has_astar_actions"].astype(bool)

    a = astar.clone().requires_grad_(True)
    q = head(z0, goal, a, upper).sum()
    g, = torch.autograd.grad(q, a)
    g = g.detach().cpu().numpy().reshape(len(dt), -1)
    u = (plus - minus).reshape(len(dt), -1) / 2.0
    norm = np.linalg.norm(u, axis=1, keepdims=True)
    u = u / np.maximum(norm, 1e-12)
    proj = (g * u).sum(axis=1)

    print(f"\n{path}  pairs={len(dt)}  with_astar={int(has_a.sum())}")
    m = has_a
    directional(proj[m], dt[m], "all (baseline any)")
    directional(proj[m & base_ok], dt[m & base_ok], "baseline SUCCESS")
    directional(proj[m & ~base_ok], dt[m & ~base_ok], "baseline FAILURE")
    bm = m & (np.abs(dt) > big)
    directional(proj[bm], dt[bm], f"|dC| > {big}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("--big", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    head = load_head(args.ckpt, device)
    print("=" * 78)
    print("TEST A — gradient direction at A* vs real finite difference")
    print("=" * 78)
    if args.train:
        run(args.train, head, device, args.big)
    run(args.val, head, device, args.big)


if __name__ == "__main__":
    main()
