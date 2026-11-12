"""RAGC oracle gate pilot (thin wrapper over `ragc_collect`).

The counterfactual simulation logic now lives in `h_jepa/ragc_collect.py` so the
Stage-1 collector and this gate share ONE implementation. Behaviour changes vs
the previous standalone pilot:

  * Strict replay validation is KEPT (anchor RGB, full remaining raw action
    trace, success equality, cross-branch anchor equality).
  * A failing anchor is marked `valid=False` with the reason and the first
    divergence step, is excluded from training labels, and collection CONTINUES
    instead of aborting the run.
  * Every finished anchor is appended to a JSONL file immediately (fsync), so a
    crash cannot destroy already-computed data.

Usage:
  cd H-JEPA/h_jepa
  export PYTHONPATH=. HJEPA_HOME=/... SDL_VIDEODRIVER=dummy
  python -u scripts/ragc_oracle_pilot.py \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --trajs /tmp/pusht_dev/dev_fails.pt --episodes 8 \
      --max-decisions 1 --directions 2 --seed-mode chunk \
      --out /tmp/ragc_pilot.json --jsonl /tmp/ragc_pilot.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ragc_collect import RAGCOracleCollector, TaskSpec, append_jsonl


def load_tasks(path, episodes, seed_mode):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    idx_all = payload.get("episodes_idx", list(range(len(payload["data"]))))
    goal_offset = int(payload.get("goal_offset_steps", -1))
    tasks = []
    for ep_i, ep in enumerate(payload["data"][:episodes]):
        seed_idx = int(idx_all[ep_i]) if seed_mode == "chunk" else ep_i
        tasks.append(
            TaskSpec(
                task_uid=f"{Path(path).stem}:{ep_i}",
                source_episode_uid=f"{Path(path).stem}:{seed_idx}",
                split="pilot",
                payload=ep,
                seed_idx=seed_idx,
                goal_offset_steps=goal_offset,
                meta={"seed_mode": seed_mode},
            )
        )
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-name", default="pusht_l2")
    p.add_argument("--trajs", default="assets/eval_trajs/pusht/goal_offset_75_val.pt")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-decisions", type=int, default=1)
    p.add_argument("--directions", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-mode", choices=("chunk", "legacy"), default="chunk")
    p.add_argument("--out", default="/tmp/ragc_pilot.json")
    p.add_argument("--jsonl", default=None,
                   help="Incremental per-anchor JSONL (defaults to <out>.jsonl)")
    p.add_argument("--dump-trace", default=None,
                   help="Optional JSONL of baseline standardized actions per task")
    p.add_argument("--dataset", default=None,
                   help="Optional NPZ export. Must NOT use held-out eval_trajs.")
    args = p.parse_args()

    if args.dataset is not None and "eval_trajs" in Path(args.trajs).parts:
        raise ValueError(
            "Refusing to export RAGC training labels from held-out eval_trajs."
        )

    jsonl = args.jsonl or str(Path(args.out).with_suffix(".jsonl"))
    Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(jsonl).write_text("")  # fresh run

    tasks = load_tasks(args.trajs, args.episodes, args.seed_mode)
    rows, pairs = [], []

    with RAGCOracleCollector(
        args.ckpt,
        config_name=args.config_name,
        seed=args.seed,
        epsilon=args.epsilon,
        directions=args.directions,
        jsonl_path=jsonl,
    ) as coll:
        if args.dump_trace:
            def hook(task, trace):
                append_jsonl(args.dump_trace, {
                    "kind": "task",
                    "task": task.seed_idx,
                    "task_uid": task.task_uid,
                    "success": bool(trace["success"]),
                    "steps": [
                        {"steps_taken": int(e["step"]),
                         "actions": np.asarray(e["actions_standard"]).tolist(),
                         "pixels_sha1": e["pixels_sha1"],
                         "goal_sha1": e["goal_sha1"]}
                        for e in trace["path"]
                    ],
                })
            coll.trace_hook = hook

        anchor_id = 0
        for task in tasks:
            out = coll.collect_task(
                task, max_decisions=args.max_decisions, anchor_id_base=anchor_id
            )
            anchor_id += len(out["anchors"])
            rows.extend(out["anchors"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "warning": "reference gate: official-equivalence verified at commit 2a0bbc7",
        "n_tasks": len(tasks),
        "n_anchors": len(rows),
        "n_valid": sum(1 for r in rows if r.get("valid")),
        "records": rows,
    }, indent=2, default=str))

    if args.dataset is not None and rows:
        ds = [r for r in rows if r.get("valid")]
        if ds:
            np.savez_compressed(
                args.dataset,
                source_episode_uid=np.asarray([r["source_episode_uid"] for r in ds]),
                task_uid=np.asarray([r["task_uid"] for r in ds]),
                anchor_id=np.asarray([r["anchor_id"] for r in ds], np.int64),
                plus_cost=np.asarray([r["candidate_costs"][0] for r in ds]),
                minus_cost=np.asarray([r["candidate_costs"][1] for r in ds]),
                epsilon=np.asarray([r["epsilon"] for r in ds]),
                label_source=np.asarray("baseline_continuation"),
            )
            print("Saved", args.dataset, flush=True)
    print("Saved", out_path, flush=True)


if __name__ == "__main__":
    main()
