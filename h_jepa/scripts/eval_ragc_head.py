"""Evaluate a trained RealizedAdvantageHead on the Stage-1 validation pairs.

Reports the breakdowns required for the Stage-1C review:

  * pair accuracy / R^2 overall, split by baseline SUCCESS vs FAILURE anchors
  * accuracy on pairs with a LARGE realized difference (|dC| > threshold)
  * reference baselines measured on the same pairs:
      - constant-zero predictor (predicts a tie -> never right on a decisive pair)
      - majority-sign predictor (fitted on TRAIN, applied to val)
  * checkpoint reload sanity: predictions must be identical after load

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/eval_ragc_head.py \
      --ckpt /tmp/ragc_stage1_smoke/head_100.pt \
      --train /tmp/ragc_stage1_smoke/train.npz \
      --val /tmp/ragc_stage1_smoke/val.npz
"""
import argparse

import numpy as np
import torch

from ragc import RAGCConfig, RealizedAdvantageHead
from ragc_data import load_ragc_npz, to_tensors


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RAGCConfig(**payload["ragc_config"])
    head = RealizedAdvantageHead(cfg).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head, payload


def predict(head, tensors, device):
    with torch.no_grad():
        p = {k: v.to(device) for k, v in tensors.items()}
        qp = head(p["z0"], p["goal"], p["plus"], p["goal_is_upper"])
        qm = head(p["z0"], p["goal"], p["minus"], p["goal_is_upper"])
    return (qp - qm).cpu().numpy()


def group_report(name, diff_p, diff_t):
    decisive = np.abs(diff_t) > 1e-4
    n = len(diff_t)
    if n == 0:
        print(f"  {name:<28} n=0")
        return
    mse = float(np.mean((diff_t - diff_p) ** 2))
    zero = float(np.mean(diff_t ** 2)) if np.any(diff_t) else 1e-9
    r2 = 1.0 - mse / max(zero, 1e-9)
    acc = (float(np.mean(np.sign(diff_p[decisive]) == np.sign(diff_t[decisive])))
           if decisive.any() else float("nan"))
    print(f"  {name:<28} n={n:<3} decisive={int(decisive.sum()):<3} "
          f"pair_R2={r2:+.3f}  pair_acc={acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--big", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    head, payload = load_head(args.ckpt, device)
    data_tr = load_ragc_npz(args.train)
    data_va = load_ragc_npz(args.val)
    tr, va = to_tensors(data_tr), to_tensors(data_va)

    dp = predict(head, va, device)
    dt = va["plus_cost"].numpy() - va["minus_cost"].numpy()
    base_ok = data_va["baseline_success"].astype(bool)

    print("=" * 72)
    print("RAGC HEAD — VALIDATION BREAKDOWN")
    print("=" * 72)
    print(f"checkpoint: {args.ckpt}")
    print(f"val pairs : {len(dt)}   train pairs: {len(tr['plus_cost'])}")

    print("\n-- groups --")
    group_report("all", dp, dt)
    group_report("baseline SUCCESS anchors", dp[base_ok], dt[base_ok])
    group_report("baseline FAILURE anchors", dp[~base_ok], dt[~base_ok])
    big = np.abs(dt) > args.big
    group_report(f"|dC| > {args.big}", dp[big], dt[big])
    small = ~big
    group_report(f"|dC| <= {args.big}", dp[small], dt[small])

    # --- reference baselines on the same decisive pairs -------------------
    dec = np.abs(dt) > 1e-4
    print("\n-- reference baselines (val, decisive pairs only) --")
    if dec.any():
        zero_acc = float(np.mean(np.sign(np.zeros_like(dt[dec])) == np.sign(dt[dec])))
        tr_dt = tr["plus_cost"].numpy() - tr["minus_cost"].numpy()
        tr_dec = np.abs(tr_dt) > 1e-4
        maj = 1.0 if (tr_dt[tr_dec] > 0).sum() >= (tr_dt[tr_dec] < 0).sum() else -1.0
        maj_acc = float(np.mean(np.sign(maj * np.ones_like(dt[dec])) == np.sign(dt[dec])))
        print(f"  constant-zero predictor : {zero_acc:.3f}  (ties never win)")
        print(f"  majority sign (train={maj:+.0f}) : {maj_acc:.3f}")
        print(f"  head                    : "
              f"{float(np.mean(np.sign(dp[dec]) == np.sign(dt[dec]))):.3f}")
    print("\n  NOTE: val has only "
          f"{int(dec.sum())} decisive pairs -> 1 error = "
          f"{100.0 / max(int(dec.sum()), 1):.1f} pp. This is a WIRING test, "
          "not a generalisation claim.")

    # --- reload consistency ----------------------------------------------
    head2, _ = load_head(args.ckpt, device)
    dp2 = predict(head2, va, device)
    print(f"\nreload check: max|delta| = {np.abs(dp - dp2).max():.3e} "
          f"{'OK' if np.allclose(dp, dp2, atol=1e-6) else 'MISMATCH'}")
    print(f"supervision flag: {payload.get('supervision')}  "
          f"steps: {payload.get('steps')}")


if __name__ == "__main__":
    main()
