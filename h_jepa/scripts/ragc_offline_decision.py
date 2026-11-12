"""Test B: OFFLINE decision test — "modify vs keep A*", evaluated on stored
counterfactual outcomes. No simulator calls, no threshold tuning on val.

For every anchor the decision set is

    A* , A*+eps*u1 , A*-eps*u1 , A*+eps*u2 , A*-eps*u2

The head scores all five. It switches away from A* only when

    Q(A*) - min_c Q(c) > margin          (margin FIXED in advance)

and otherwise keeps A*. The chosen action's REAL outcome is then read from the
stored counterfactual results, so we report

    rescued   (baseline fail -> chosen success)
    destroyed (baseline success -> chosen fail)
    net = rescued - destroyed

The primary margin is 0.05 (fixed a priori). The sensitivity table is printed
for transparency only and MUST NOT be used to pick a threshold after seeing the
validation numbers.

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/ragc_offline_decision.py \
      --ckpt /tmp/ragc_merged/head.pt \
      --val /tmp/ragc_merged/val.npz [--train /tmp/ragc_merged/train.npz]
"""
import argparse

import numpy as np
import torch

from ragc import RAGCConfig, RealizedAdvantageHead
from ragc_data import load_ragc_npz

MARGIN = 0.05  # fixed a priori; do not tune on validation


def load_head(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RAGCConfig(**payload["ragc_config"])
    head = RealizedAdvantageHead(cfg).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head


def build_decision_set(path):
    """Group pairs into anchors -> A* + 4 candidates + REAL outcomes."""
    d = load_ragc_npz(path)
    anchor_ids = np.asarray(d["anchor_id"])
    uniq = list(dict.fromkeys(anchor_ids.tolist()))
    items = []
    for aid in uniq:
        idx = np.where(anchor_ids == aid)[0]
        if len(idx) < 2:
            continue  # need both perturbation directions for the 5-action set
        i0, i1 = idx[0], idx[1]
        # Anchors collected before Stage 1D did not store the realized A* plan;
        # scoring a zero placeholder as "keep the original action" would be
        # meaningless, so they are excluded from the decision test.
        if not bool(d["has_astar_actions"][i0]):
            continue
        items.append({
            "anchor_id": int(aid),
            "z0": d["z0"][i0].astype(np.float32),
            "goal": d["goal"][i0].astype(np.float32),
            "goal_is_upper": bool(d["goal_is_upper"][i0]),
            "actions": np.stack([
                d["astar_actions"][i0],
                d["plus"][i0], d["minus"][i0],
                d["plus"][i1], d["minus"][i1],
            ]).astype(np.float32),                       # [5,2,10]
            "real_success": np.asarray([
                bool(d["baseline_success"][i0]),
                bool(d["plus_success"][i0]), bool(d["minus_success"][i0]),
                bool(d["plus_success"][i1]), bool(d["minus_success"][i1]),
            ]),
        })
    return items


def score_all(head, items, device):
    acts = torch.as_tensor(np.stack([it["actions"] for it in items]),
                           dtype=torch.float32, device=device)   # [N,5,2,10]
    n, c = acts.shape[0], acts.shape[1]
    z = torch.as_tensor(np.stack([it["z0"] for it in items]),
                        dtype=torch.float32, device=device)
    g = torch.as_tensor(np.stack([it["goal"] for it in items]),
                        dtype=torch.float32, device=device)
    up = torch.as_tensor(np.stack([it["goal_is_upper"] for it in items]),
                         dtype=torch.bool, device=device)
    with torch.no_grad():
        q = head(z.repeat_interleave(c, 0), g.repeat_interleave(c, 0),
                 acts.reshape(n * c, *acts.shape[2:]),
                 up.repeat_interleave(c, 0))
    return q.reshape(n, c).cpu().numpy()


def evaluate(items, q, margin):
    rescued = destroyed = switched = 0
    base_succ = chosen_succ = 0
    for i, it in enumerate(items):
        ok = it["real_success"]
        best = int(np.argmin(q[i, 1:])) + 1
        take = float(q[i, 0] - q[i, best]) > margin
        chosen = bool(ok[best]) if take else bool(ok[0])
        base_succ += int(ok[0])
        chosen_succ += int(chosen)
        switched += int(take)
        rescued += int(take and not ok[0] and chosen)
        destroyed += int(take and ok[0] and not chosen)
    return {
        "n": len(items), "base_succ": base_succ, "chosen_succ": chosen_succ,
        "switched": switched, "rescued": rescued, "destroyed": destroyed,
        "net": rescued - destroyed,
    }


def show(name, res, margin):
    print(f"  {name:<8} n={res['n']:<3} margin={margin:<5} "
          f"base={res['base_succ']} chosen={res['chosen_succ']} "
          f"switch={res['switched']} rescue={res['rescued']} "
          f"destroy={res['destroyed']} NET={res['net']:+d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="fixed decision margin (default 0.05)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    head = load_head(args.ckpt, device)
    print("=" * 78)
    print("TEST B — offline 'modify vs keep A*' decision on REAL outcomes")
    print("=" * 78)

    val_items = build_decision_set(args.val)
    val_q = score_all(head, val_items, device)
    if args.train:
        tr_items = build_decision_set(args.train)
        tr_q = score_all(head, tr_items, device)
        print("\n-- reference (train) --")
        show("train", evaluate(tr_items, tr_q, args.margin), args.margin)

    print("\n-- PRIMARY (validation, margin fixed in advance) --")
    show("val", evaluate(val_items, val_q, args.margin), args.margin)

    print("\n-- sensitivity (diagnostic only; NOT for threshold selection) --")
    for m in (0.0, 0.02, 0.05, 0.1, 0.2):
        show("val", evaluate(val_items, val_q, m), m)


if __name__ == "__main__":
    main()
