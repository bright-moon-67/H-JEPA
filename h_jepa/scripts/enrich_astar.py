"""Recover the realized A* plan for anchors collected before Stage 1D.

Stage 1C records did not store `astar_actions`, so the offline decision test
could not score "keep the original action" for them (it would have used a zero
placeholder). This script RE-RUNS ONLY THE FROZEN BASELINE (no counterfactual
branches, no extra labels) to recover `path[0]["actions_standard"]`, and then
verifies the replayed anchor is the same one that was recorded:

    pixels_sha1 must match, goal_sha1 must match, baseline success and step
    count must match.

If any check fails the anchor is left without A* (never silently patched).

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/enrich_astar.py \
      --jsonl artifacts/stage1c_smoke/anchors.jsonl \
      --h5 "$HJEPA_HOME/pusht_expert_train.h5" \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --out /tmp/ragc_stage1c_enriched/anchors.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from ragc_collect import RAGCOracleCollector, TaskSpec

UID = re.compile(r"^(?P<split>[a-z]+):ep(?P<ep>\d+):off(?P<off>\d+):s(?P<start>\d+)$")


def build_payload(h5, ep_id, start, length):
    off = int(h5["ep_offset"][ep_id])
    sl = slice(off + start, off + start + length)
    px = np.transpose(h5["pixels"][sl], (0, 3, 1, 2))
    return {
        "pixels": torch.from_numpy(np.ascontiguousarray(px)),
        "state": torch.from_numpy(np.asarray(h5["state"][sl], np.float32)),
        "proprio": torch.from_numpy(np.asarray(h5["proprio"][sl], np.float32)),
        "action": torch.from_numpy(np.asarray(h5["action"][sl], np.float32)),
        "step_idx": torch.from_numpy(np.asarray(h5["step_idx"][sl], np.int64)),
        "episode_idx": torch.from_numpy(
            np.asarray(h5["episode_idx"][sl], np.int64)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="pusht_l2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines()]
    anchors = [r for r in recs if r.get("kind") == "anchor"]
    todo = [r for r in anchors if not r.get("astar_actions")]
    print(f"anchors={len(anchors)}  missing A*={len(todo)}")

    out_recs = []
    with h5py.File(args.h5, "r") as h5, RAGCOracleCollector(
        args.ckpt, config_name=args.config_name, seed=args.seed,
        epsilon=0.30, directions=2, jsonl_path=None,
    ) as coll:
        for r in recs:
            if r.get("kind") != "anchor" or r.get("astar_actions"):
                out_recs.append(r)
                continue
            m = UID.match(r["task_uid"])
            if m is None:
                print(f"  cannot parse {r['task_uid']} -> leave without A*")
                out_recs.append(r)
                continue
            ep = int(m["ep"]); start = int(m["start"])
            off = int(m["off"])
            payload = build_payload(h5, ep, start, off + 1)
            task = TaskSpec(
                task_uid=r["task_uid"], source_episode_uid=r["source_episode_uid"],
                split=r["split"], payload=payload, seed_idx=int(r["seed_idx"]),
                goal_offset_steps=off, meta={"seed_mode": "chunk"},
            )
            states = payload["state"].numpy().astype(np.float64)
            ep_pixels = np.transpose(payload["pixels"].numpy(), (0, 2, 3, 1))
            coll._reseed(task.seed_idx)
            trace = coll.trace_baseline(states[0], states[-1], ep_pixels)
            entry = trace["path"][0] if trace["path"] else None
            ok = (
                entry is not None
                and entry["pixels_sha1"] == r["pixels_sha1"]
                and entry["goal_sha1"] == r["goal_sha1"]
                and bool(trace["success"]) == bool(r["baseline_success"])
                and int(trace["elapsed"]) == int(r["baseline_steps"])
            )
            if not ok:
                print(f"  VERIFY FAILED {r['task_uid']} -> leave without A*")
                out_recs.append(r)
                continue
            r = dict(r)
            r["astar_actions"] = np.asarray(
                entry["actions_standard"], np.float32
            ).tolist()
            r["astar_source"] = "baseline_retrace_verified"
            print(f"  recovered A* for {r['task_uid']} "
                  f"(sha1 match, steps={trace['elapsed']})")
            out_recs.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    n_ok = sum(1 for r in out_recs
               if r.get("kind") == "anchor" and r.get("astar_actions"))
    print(f"wrote {out}  anchors with A*: {n_ok}")


if __name__ == "__main__":
    main()
