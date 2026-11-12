"""RAGC Stage 1D: ONE bounded, targeted re-collection.

Stage 1C left the splits skewed: train was dominated by anchors whose baseline
SUCCEEDED, while val contained only baseline FAILURES.  This collector fixes the
coverage with a hard budget:

    train: +N failure anchors   (episodes where the frozen policy fails)
    val  : +N success anchors   (episodes where the frozen policy succeeds)

Cheap first, expensive later: every candidate episode is first rolled out with
the frozen baseline only.  The five-branch counterfactual rollout (A* + 4
antithetic candidates) runs ONLY for episodes that match the requested class.
That avoids spending full counterfactual rollouts on easy/hard tasks we do not
need.

Constraints (fixed by the review):
  * epsilon = 0.3, directions = 2, first planning step only
  * original two-level H-JEPA, action cost weight 0.2, 90-step budget
  * train / val episodes come from DISJOINT expert episode id ranges
  * new output directory (never overwrites the Stage 1C artifacts)
  * stops after the anchor budget is met; no second round

Usage:
  cd H-JEPA/h_jepa
  export PYTHONPATH=. HJEPA_HOME=/... SDL_VIDEODRIVER=dummy
  python -u scripts/collect_ragc_stage1d.py \
      --h5 "$HJEPA_HOME/pusht_expert_train.h5" \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --train-episodes 50-99 --val-episodes 100-139 \
      --need-train-fail 6 --need-val-success 6 \
      --out /tmp/ragc_stage1d
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from ragc_collect import RAGCOracleCollector, TaskSpec, append_jsonl


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
    px = np.transpose(h5["pixels"][sl], (0, 3, 1, 2))
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
    return TaskSpec(
        task_uid=f"{split}:ep{ep_id:04d}:off{goal_offset}:s{start}",
        source_episode_uid=f"train_ep{ep_id:04d}",
        split=split,
        payload=payload,
        seed_idx=int(seed_base + ep_id),
        goal_offset_steps=int(goal_offset),
        meta={"seed_mode": "chunk", "start_step": int(start), "ep_len": ep_len},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="pusht_l2")
    ap.add_argument("--train-episodes", default="50-99")
    ap.add_argument("--val-episodes", default="100-139")
    ap.add_argument("--need-train-fail", type=int, default=6)
    ap.add_argument("--need-val-success", type=int, default=6)
    ap.add_argument("--max-train-screen", type=int, default=45)
    ap.add_argument("--max-val-screen", type=int, default=20)
    ap.add_argument("--goal-offset", type=int, default=75)
    ap.add_argument("--start-shift", type=int, default=5)
    ap.add_argument("--directions", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_eps = parse_episode_spec(args.train_episodes)
    val_eps = parse_episode_spec(args.val_episodes)
    if set(train_eps) & set(val_eps):
        raise SystemExit("train/val episode ranges overlap")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "anchors.jsonl"
    screen_jsonl = out_dir / "screening.jsonl"
    if jsonl.exists() and jsonl.stat().st_size > 0:
        raise SystemExit(f"{jsonl} already has data; use a fresh --out directory")
    jsonl.write_text("")
    screen_jsonl.write_text("")

    with h5py.File(args.h5, "r") as h5:
        n_eps = len(h5["ep_len"])
        for e in train_eps + val_eps:
            if e >= n_eps:
                raise SystemExit(f"episode {e} out of range ({n_eps})")

        def queue(spec, split, want_fail, max_screen):
            out = []
            for e in spec[:max_screen]:
                t = build_task(h5, e, split, args.goal_offset,
                               args.start_shift, args.seed)
                if t is not None:
                    out.append((t, want_fail))
            return out

        # train -> look for FAILING baselines, val -> look for SUCCEEDING ones
        plan = (queue(train_eps, "train", True, args.max_train_screen)
                + queue(val_eps, "val", False, args.max_val_screen))

    need = {"train_fail": args.need_train_fail,
            "val_success": args.need_val_success}
    got = {"train_fail": 0, "val_success": 0}
    anchor_id = 0
    collected = []

    with RAGCOracleCollector(
        args.ckpt,
        config_name=args.config_name,
        seed=args.seed,
        epsilon=args.epsilon,
        directions=args.directions,
        jsonl_path=str(jsonl),
    ) as coll:
        for task, want_fail in plan:
            key = "train_fail" if want_fail else "val_success"
            if got[key] >= need[key]:
                continue
            # ---- cheap stage: baseline only -----------------------------
            ep = task.payload
            states = np.asarray(ep["state"].numpy(), np.float64)
            pixels = ep["pixels"].numpy()
            ep_pixels = np.transpose(pixels, (0, 2, 3, 1))
            coll._reseed(task.seed_idx)
            trace = coll.trace_baseline(states[0], states[-1], ep_pixels)
            append_jsonl(screen_jsonl, {
                "task_uid": task.task_uid,
                "split": task.split,
                "baseline_success": bool(trace["success"]),
                "baseline_steps": int(trace["elapsed"]),
                "wanted": "fail" if want_fail else "success",
            })
            keep = (not trace["success"]) if want_fail else bool(trace["success"])
            print(f"[screen] {task.task_uid} baseline="
                  f"{'OK' if trace['success'] else 'FAIL'}({trace['elapsed']}) "
                  f"{'-> collect' if keep else 'skip'}", flush=True)
            if not keep:
                continue
            # ---- expensive stage: A* + 4 candidates ---------------------
            entry = trace["path"][0] if trace["path"] else None
            if entry is None:
                continue
            rec = coll.collect_anchor(
                task, entry, trace, states[0], states[-1],
                np.asarray(ep_pixels[-1], np.uint8),
                anchor_id=anchor_id,
                rng=np.random.default_rng(args.seed + task.seed_idx),
            )
            rec["kind"] = "anchor"
            append_jsonl(jsonl, rec)
            if rec.get("valid"):
                got[key] += 1
                collected.append(rec)
            anchor_id += 1
            print(f"  anchor valid={rec.get('valid')} "
                  f"{rec.get('invalid_reason','')} "
                  f"cand_ok={rec.get('candidate_successes')}", flush=True)
            if got["train_fail"] >= need["train_fail"] and \
                    got["val_success"] >= need["val_success"]:
                print("budget met", flush=True)
                break

    summary = {
        "need": need,
        "got": got,
        "n_new_valid": len(collected),
        "episodes_used": [r["source_episode_uid"] for r in collected],
        "epsilon": args.epsilon,
        "directions": args.directions,
        "goal_offset": args.goal_offset,
    }
    (out_dir / "stage1d_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
