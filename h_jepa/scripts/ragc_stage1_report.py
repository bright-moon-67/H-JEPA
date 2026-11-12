"""Stage-1 label report: rebuild the NPZ with provenance and print the
statistics requested for the Stage-1C review.

Key distinctions enforced here (they are NOT the same quantity):

  * candidate-vs-A*  dC_i = C(A_i) - C(A*)       (relative realized change)
  * paired (training) dC   = C(A+) - C(A-)        (what the head is trained on)

The head only ever sees the SECOND one, so a large dC_i does not by itself
imply a usable gradient signal.

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/ragc_stage1_report.py \
      --jsonl /tmp/ragc_stage1_smoke/anchors.jsonl \
      --out-dir /tmp/ragc_stage1_smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path):
    recs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "anchor":
            recs.append(r)
    return recs


def pct(a, b):
    return f"{a}/{b} ({100.0 * a / b:.1f}%)" if b else f"{a}/{b} (n/a)"


def rebuild_npz(records, out_dir):
    """NPZ with per-pair supervision + baseline/anchor provenance."""
    made = {}
    for split in ("train", "val"):
        rows = [r for r in records
                if r["split"] == split and r.get("valid") and r.get("pairs")]
        if not rows:
            made[split] = None
            continue
        flat = []
        for r in rows:
            for p in r["pairs"]:
                flat.append({
                    "anchor_id": np.int64(r["anchor_id"]),
                    "z0": np.asarray(r["z0"], np.float32),
                    "goal": np.asarray(r["goal_emb"], np.float32),
                    "goal_is_upper": np.bool_(r["goal_is_upper"]),
                    "plus": np.asarray(p["plus"], np.float32),
                    "minus": np.asarray(p["minus"], np.float32),
                    "plus_cost": np.float32(p["plus_cost"]),
                    "minus_cost": np.float32(p["minus_cost"]),
                    "epsilon": np.float32(r["epsilon"]),
                    "source_episode_uid": np.asarray(r["source_episode_uid"]),
                    "task_uid": np.asarray(r["task_uid"]),
                    "baseline_success": np.bool_(r["baseline_success"]),
                    "astar_cost": np.float32(r["astar_cost"]),
                    "plus_success": np.bool_(r["candidate_successes"][0]
                                             if len(r["candidate_successes"]) > 0
                                             else False),
                    "minus_success": np.bool_(r["candidate_successes"][1]
                                              if len(r["candidate_successes"]) > 1
                                              else False),
                })
        keys = list(flat[0].keys())
        ds = {k: np.stack([f[k] for f in flat]) for k in keys}
        ds["label_source"] = np.asarray("baseline_continuation")
        p = Path(out_dir) / f"{split}.npz"
        np.savez_compressed(p, **ds)
        made[split] = (p, ds)
    return made


def describe(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return "n=0"
    return (f"n={x.size} mean={x.mean():+.4f} median={np.median(x):+.4f} "
            f"min={x.min():+.4f} max={x.max():+.4f} "
            f"nonzero={int((np.abs(x) > 1e-4).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--big", type=float, default=0.5,
                    help="|paired dC| threshold for 'large signal' samples")
    ap.add_argument("--normalizer", default=None,
                    help="normalizer.pt next to the checkpoint; used to check "
                         "the REAL action range after de-standardisation")
    args = ap.parse_args()

    recs = load_records(args.jsonl)
    valid = [r for r in recs if r.get("valid")]
    invalid = [r for r in recs if not r.get("valid")]

    print("=" * 78)
    print("STAGE 1C — LABEL REPORT")
    print("=" * 78)
    print(f"anchors total   : {len(recs)}")
    print(f"  valid         : {len(valid)}")
    print(f"  invalid       : {len(invalid)}")
    for r in invalid:
        print(f"    - {r['task_uid']} anchor {r['anchor_id']}: "
              f"{r.get('invalid_reason')}")

    print("\n--- per-task ---")
    print(f"{'task':<28}{'base':>6}{'cand_ok':>10}{'rescued':>9}{'destroyed':>11}")
    tasks = {}
    for r in valid:
        t = tasks.setdefault(r["task_uid"], {
            "baseline": r["baseline_success"], "ok": [], "destroyed": 0
        })
        t["ok"].append(sum(bool(s) for s in r["candidate_successes"]))
        if r["baseline_success"]:
            t["destroyed"] += sum(1 for s in r["candidate_successes"]
                                  if not bool(s))
    n_task_success = n_task_fail = 0
    for k, t in tasks.items():
        rescued = (not t["baseline"]) and any(o > 0 for o in t["ok"])
        n_task_success += bool(t["baseline"])
        n_task_fail += (not t["baseline"])
        print(f"{k:<28}{'OK' if t['baseline'] else 'FAIL':>6}"
              f"{sum(t['ok'])}/{4*len(t['ok']):>8}"
              f"{'YES' if rescued else '-':>9}{t['destroyed']:>11}")

    dcost_all, pair_dc, pair_flip = [], [], 0
    succ_anchor_dc, fail_anchor_dc = [], []
    n_flip_pairs = 0
    big_pairs = []
    for r in valid:
        c = np.asarray(r["candidate_costs"], np.float64)
        a = float(r["astar_cost"])
        d = c - a
        dcost_all.extend(d.tolist())
        (succ_anchor_dc if r["baseline_success"] else fail_anchor_dc).extend(
            d.tolist())
        ok = [bool(s) for s in r["candidate_successes"]]
        for i in range(0, len(c) - 1, 2):
            pdc = float(c[i] - c[i + 1])
            pair_dc.append(pdc)
            if ok[i] != ok[i + 1]:
                n_flip_pairs += 1
            if abs(pdc) > args.big:
                big_pairs.append(pdc)

    print("\n--- candidate vs A* (dC_i = C(A_i) - C(A*)) ---")
    print("  all      :", describe(dcost_all))
    print("  baseline OK  anchors:", describe(succ_anchor_dc))
    print("  baseline FAIL anchors:", describe(fail_anchor_dc))

    print("\n--- paired supervised target (C(A+) - C(A-)) ---")
    print("  all      :", describe(pair_dc))
    print(f"  |dC| > {args.big}: n={len(big_pairs)}")

    cand_ok = [bool(s) for r in valid for s in r["candidate_successes"]]
    destroyed = sum(1 for r in valid if r["baseline_success"]
                    for s in r["candidate_successes"] if not bool(s))
    rescued = sum(1 for r in valid if not r["baseline_success"]
                  for s in r["candidate_successes"] if bool(s))
    print("\n--- success flips ---")
    print(f"  tasks: {len(tasks)}  baseline OK {n_task_success} / "
          f"FAIL {n_task_fail}")
    print(f"  task-level rescued  : "
          f"{sum(1 for t in tasks.values() if not t['baseline'] and any(o > 0 for o in t['ok']))}"
          f" / {n_task_fail}")
    print(f"  candidate success   : {pct(sum(cand_ok), len(cand_ok))}")
    print(f"  candidates that BROKE a successful episode : "
          f"{pct(destroyed, sum(len(r['candidate_successes']) for r in valid if r['baseline_success']))}")
    print(f"  candidates that RESCUED a failed episode   : "
          f"{pct(rescued, sum(len(r['candidate_successes']) for r in valid if not r['baseline_success']))}")
    print(f"  pairs whose two candidates differ in success: "
          f"{n_flip_pairs}/{len(pair_dc)}")

    acts = np.asarray([p[k] for r in valid for p in r["pairs"]
                       for k in ("plus", "minus")], np.float32)
    if acts.size:
        print("\n--- action legality ---")
        print(f"  standardized max|a| = {np.abs(acts).max():.4f} "
              "(tokens are NOT bounded by 1; only env actions are)")
        if args.normalizer:
            import torch
            norm = torch.load(args.normalizer, map_location="cpu",
                              weights_only=False)
            mean = np.asarray(norm["stats"]["action"]["mean"]).reshape(1, 2)
            std = np.asarray(norm["stats"]["action"]["std"]).reshape(1, 2)
            raw = acts.reshape(-1, 2) * std + mean
            print(f"  raw env action max|a| = {np.abs(raw).max():.4f}   "
                  f"|a|>1 ratio = {(np.abs(raw) > 1.0).mean():.4f}   "
                  f"(Push-T action box is [-1,1])")

    made = rebuild_npz(recs, args.out_dir)
    print("\n--- NPZ ---")
    for split, v in made.items():
        if v is None:
            print(f"  {split}: EMPTY")
            continue
        p, ds = v
        print(f"  {split}: {p}  pairs={len(ds['plus_cost'])}  "
              f"anchors={len(np.unique(ds['anchor_id']))}  "
              f"episodes={sorted(set(ds['source_episode_uid'].tolist()))}")
    if made["train"] and made["val"]:
        a = set(made["train"][1]["source_episode_uid"].tolist())
        b = set(made["val"][1]["source_episode_uid"].tolist())
        print(f"  episode overlap train/val: {sorted(a & b) or 'NONE'}")


if __name__ == "__main__":
    main()
