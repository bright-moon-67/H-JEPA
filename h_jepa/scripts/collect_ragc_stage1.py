"""RAGC Stage 1A: real-return data collector (smoke-scale).

Builds task windows from the EXPERT TRAIN split (`pusht_expert_train.h5`),
splits them by ORIGINAL expert episode id (never by window/anchor id), and
collects real counterfactual labels with the frozen H-JEPA policy:

    A* (the plan actually executed)      -> reference cost
    A* +/- eps*u  (2 directions, 4 cand) -> realized continuation costs

Labels are produced only for anchors that pass the strict replay gate
(see `ragc_collect`). Invalid anchors are written with `valid=False` and are
excluded from the training NPZ, but collection continues.

Outputs (all incremental / crash-safe):
  <out>/anchors.jsonl      one line per anchor (valid or invalid)
  <out>/train.npz          paired labels for RAGC training
  <out>/val.npz            paired labels for validation (episode-disjoint)
  <out>/summary.json       aggregate statistics

Example (smoke):
  cd H-JEPA/h_jepa
  export PYTHONPATH=. HJEPA_HOME=/... SDL_VIDEODRIVER=dummy
  python -u scripts/collect_ragc_stage1.py \
      --h5 "$HJEPA_HOME/pusht_expert_train.h5" \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --train-episodes 40-45 --val-episodes 46-49 \
      --goal-offset 75 --directions 2 --epsilon 0.3 \
      --out /tmp/ragc_stage1_smoke
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers the compression filters
import numpy as np
import torch

from ragc_collect import RAGCOracleCollector, TaskSpec


def parse_episode_spec(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def build_task(h5, ep_id, split, goal_offset, start_shift, seed_base):
    ep_len = int(h5["ep_len"][ep_id])
    ep_off = int(h5["ep_offset"][ep_id])
    L = goal_offset + 1
    if ep_len < L + 10:
        return None
    start = (start_shift + 3 * ep_id) % (ep_len - L - 1)
    sl = slice(ep_off + start, ep_off + start + L)
    px = np.transpose(h5["pixels"][sl], (0, 3, 1, 2))  # HWC -> CHW
    payload = {
        "pixels": torch.from_numpy(np.ascontiguousarray(px)),
        "state": torch.from_numpy(np.asarray(h5["state"][sl], np.float32)),
        "proprio": torch.from_numpy(np.asarray(h5["proprio"][sl], np.float32)),
        "action": torch.from_numpy(np.asarray(h5["action"][sl], np.float32)),
        "step_idx": torch.from_numpy(np.asarray(h5["step_idx"][sl], np.int64)),
        "episode_idx": torch.from_numpy(
            np.asarray(h5["episode_idx"][sl], np.int64)
        ),
    }
    uid = f"{split}:ep{ep_id:04d}:off{goal_offset}:s{start}"
    return TaskSpec(
        task_uid=uid,
        source_episode_uid=f"train_ep{ep_id:04d}",
        split=split,
        payload=payload,
        seed_idx=int(seed_base + ep_id),
        goal_offset_steps=int(goal_offset),
        meta={"seed_mode": "chunk", "start_step": int(start), "ep_len": ep_len},
    )


def build_npz(records, path):
    rows = [r for r in records
            if r.get("valid") and r.get("pairs") and "z0" in r]
    if not rows:
        return None
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
            })
    ds = {k: np.stack([f[k] for f in flat]) for k in
          ("anchor_id", "z0", "goal", "goal_is_upper", "plus", "minus",
           "plus_cost", "minus_cost", "epsilon",
           "source_episode_uid", "task_uid")}
    ds["label_source"] = np.asarray("baseline_continuation")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ds)
    return ds


def summarize(records, out_dir):
    valid = [r for r in records if r.get("valid")]
    invalid = [r for r in records if not r.get("valid")]
    dc = np.asarray(
        [d for r in valid for d in r.get("realized_cost_diff", [])], np.float64
    )
    cand_ok = [s for r in valid for s in r.get("candidate_successes", [])]
    summary = {
        "n_anchors": len(records),
        "n_valid": len(valid),
        "n_invalid": len(invalid),
        "invalid_reasons": sorted(
            {r.get("invalid_reason", "?") for r in invalid}
        ),
        "n_baseline_success": int(sum(1 for r in records
                                     if r.get("baseline_success"))),
        "n_recoverable": int(sum(1 for r in valid if r.get("recoverable"))),
        "n_fragile": int(sum(1 for r in valid if r.get("fragile"))),
        "n_pairs": int(len(dc)),
        "n_nonzero_pairs": int((np.abs(dc) > 1e-4).sum()),
        "dc_mean": float(dc.mean()) if len(dc) else None,
        "dc_median": float(np.median(dc)) if len(dc) else None,
        "dc_min": float(dc.min()) if len(dc) else None,
        "dc_max": float(dc.max()) if len(dc) else None,
        "candidate_success_rate": (
            float(np.mean(cand_ok)) if cand_ok else None
        ),
        "episode_overlap": None,
    }
    if valid:
        acts = np.asarray(
            [p[k] for r in valid for p in r["pairs"] for k in ("plus", "minus")],
            np.float32,
        )
        summary["action_abs_max"] = float(np.abs(acts).max())
        summary["action_oob_ratio"] = float((np.abs(acts) > 1.0).mean())
    p = Path(out_dir) / "summary.json"
    p.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="pusht_l2")
    ap.add_argument("--train-episodes", default="40-45")
    ap.add_argument("--val-episodes", default="46-49")
    ap.add_argument("--goal-offset", type=int, default=75)
    ap.add_argument("--start-shift", type=int, default=5)
    ap.add_argument("--directions", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-decisions", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="only build and report task windows (no model, no rollout)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "anchors.jsonl"
    jsonl.write_text("")

    train_eps = parse_episode_spec(args.train_episodes)
    val_eps = parse_episode_spec(args.val_episodes)
    overlap = set(train_eps) & set(val_eps)
    if overlap:
        raise SystemExit(f"TRAIN/VAL episode leak: {sorted(overlap)}")

    with h5py.File(args.h5, "r") as h5:
        n_eps = len(h5["ep_len"])
        missing = [e for e in train_eps + val_eps if e >= n_eps]
        if missing:
            raise SystemExit(f"episode ids out of range ({n_eps} available): {missing}")
        tasks = []
        for e in train_eps:
            t = build_task(h5, e, "train", args.goal_offset,
                           args.start_shift, args.seed)
            if t is not None:
                tasks.append(t)
            else:
                print(f"skip train ep{e}: too short")
        for e in val_eps:
            t = build_task(h5, e, "val", args.goal_offset,
                           args.start_shift, args.seed)
            if t is not None:
                tasks.append(t)
            else:
                print(f"skip val ep{e}: too short")
    print(f"built {len(tasks)} tasks "
          f"({len(train_eps)} train eps / {len(val_eps)} val eps)", flush=True)

    if args.dry_run:
        for t in tasks:
            px = t.payload["pixels"]
            print(f"  {t.task_uid} split={t.split} "
                  f"pixels={tuple(px.shape)} state={tuple(t.payload['state'].shape)} "
                  f"start={t.meta['start_step']} ep_len={t.meta['ep_len']} "
                  f"seed_idx={t.seed_idx}")
        return

    records = []
    anchor_id = 0
    with RAGCOracleCollector(
        args.ckpt,
        config_name=args.config_name,
        seed=args.seed,
        epsilon=args.epsilon,
        directions=args.directions,
        jsonl_path=str(jsonl),
    ) as coll:
        for task in tasks:
            out = coll.collect_task(
                task, max_decisions=args.max_decisions,
                anchor_id_base=anchor_id,
            )
            for r in out["anchors"]:
                r["global_anchor_id"] = anchor_id
                anchor_id += 1
            records.extend(out["anchors"])

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    tr = build_npz(train_recs, out_dir / "train.npz")
    va = build_npz(val_recs, out_dir / "val.npz")

    summary = summarize(records, out_dir)
    if tr is not None and va is not None:
        a = set(np.asarray(tr["source_episode_uid"]).tolist())
        b = set(np.asarray(va["source_episode_uid"]).tolist())
        summary["episode_overlap"] = sorted(a & b)
        summary["train_episodes"] = sorted(a)
        summary["val_episodes"] = sorted(b)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("saved:", out_dir / "anchors.jsonl",
          "|", out_dir / "train.npz" if tr is not None else "no train.npz",
          "|", out_dir / "val.npz" if va is not None else "no val.npz")


if __name__ == "__main__":
    main()
