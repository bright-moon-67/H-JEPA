"""Stage 2 — RAGC-Discrete small closed-loop pilot.

Runs the REAL environment twice per task:

    Original H-JEPA        (selector detached)
    H-JEPA + Discrete RAGC (selector attached; first planning call only)

Same task window, same planner seed, same 90-step budget.  Tasks are taken from
expert episodes that were NEVER used for Stage-1C/1D collection or tuning and
are NOT filtered by baseline outcome.

Before the pilot it runs the two engineering equivalence checks:

  CHECK 1  selector=None reproduces the baseline trace exactly
  CHECK 2  selector attached but forced to never switch (huge margin)
           also reproduces it exactly  -> isolates "RNG/plumbing" from "RAGC"

Usage:
  cd H-JEPA/h_jepa
  export PYTHONPATH=. HJEPA_HOME=... SDL_VIDEODRIVER=dummy
  python -u scripts/ragc_discrete_pilot.py \
      --h5 "$HJEPA_HOME/pusht_expert_train.h5" \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --head artifacts/stage1d/head_100.pt \
      --episodes 200-211 --goal-offset 75 --out /tmp/ragc_discrete_pilot
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from ragc import RAGCConfig, RealizedAdvantageHead
from ragc_collect import RAGCOracleCollector, TaskSpec
from ragc_discrete import DiscreteConfig, RAGCDiscreteSelector


def parse_spec(spec):
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


def build_task(h5, ep_id, goal_offset, start_shift=5, seed_base=42):
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
    }
    return TaskSpec(
        task_uid=f"pilot:ep{ep_id:04d}:off{goal_offset}:s{start}",
        source_episode_uid=f"train_ep{ep_id:04d}",
        split="pilot",
        payload=payload,
        seed_idx=int(seed_base + ep_id),
        goal_offset_steps=int(goal_offset),
        meta={"seed_mode": "chunk", "start_step": int(start), "ep_len": ep_len},
    )


def run_once(coll, task, selector):
    coll.solver.set_ragc_selector(selector)
    if selector is not None:
        selector.stats = type(selector.stats)()
    states = np.asarray(task.payload["state"].numpy(), np.float64)
    px = np.transpose(task.payload["pixels"].numpy(), (0, 2, 3, 1))
    coll._reseed(task.seed_idx)
    trace = coll.trace_baseline(states[0], states[-1], px)
    return trace


def trace_key(trace):
    return [(e["step"], e["pixels_sha1"], e["goal_sha1"]) for e in trace["path"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--config-name", default="pusht_l2")
    ap.add_argument("--episodes", default="200-211")
    ap.add_argument("--goal-offset", type=int, default=75)
    ap.add_argument("--epsilon", type=float, default=0.30)
    ap.add_argument("--directions", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max-regression", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--equivalence-tasks", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with h5py.File(args.h5, "r") as h5:
        tasks = [t for t in (build_task(h5, e, args.goal_offset)
                             for e in parse_spec(args.episodes)) if t is not None]
    print(f"built {len(tasks)} pilot tasks", flush=True)

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    if payload.get("supervision") != "baseline_continuation":
        raise SystemExit("head was not trained on real continuation labels")
    head_cfg = RAGCConfig(**payload["ragc_config"])
    head = RealizedAdvantageHead(head_cfg)
    head.load_state_dict(payload["head_state_dict"])

    with RAGCOracleCollector(args.ckpt, config_name=args.config_name,
                             seed=args.seed, epsilon=args.epsilon,
                             directions=args.directions, jsonl_path=None,
                             deterministic_reset=True) as coll:
        device = coll.device
        head = head.to(device).eval()
        cfg = DiscreteConfig(
            epsilon=args.epsilon, directions=args.directions,
            margin=args.margin,
            max_original_cost_regression=args.max_regression,
            seed=args.seed,
        )
        selector = RAGCDiscreteSelector(head, cfg, coll.mean, coll.std,
                                        device=device)
        frozen = RAGCDiscreteSelector(head, replace(cfg, margin=1e9),
                                      coll.mean, coll.std, device=device)

        # ---------------- CHECK 1 / CHECK 2 ---------------------------- #
        checks = []
        for t in tasks[:args.equivalence_tasks]:
            a = run_once(coll, t, None)
            b = run_once(coll, t, None)
            c = run_once(coll, t, frozen)
            ok_off = trace_key(a) == trace_key(b) and a["success"] == b["success"]
            ok_frozen = (trace_key(a) == trace_key(c)
                         and a["success"] == c["success"]
                         and frozen.stats.n_switched == 0)
            checks.append({"task": t.task_uid, "check1_selector_off": ok_off,
                           "check2_forced_no_switch": ok_frozen})
            print(f"[equiv] {t.task_uid} off={ok_off} forced_no_switch="
                  f"{ok_frozen}", flush=True)
        if not all(c["check1_selector_off"] and c["check2_forced_no_switch"]
                   for c in checks):
            (out_dir / "equivalence.json").write_text(json.dumps(checks, indent=2))
            raise SystemExit("EQUIVALENCE CHECK FAILED - aborting pilot")

        # ---------------- paired pilot --------------------------------- #
        for t in tasks:
            base = run_once(coll, t, None)
            dis = run_once(coll, t, selector)
            st = selector.stats
            row = {
                "task": t.task_uid,
                "episode": t.source_episode_uid,
                "start_step": t.meta["start_step"],
                "baseline_success": bool(base["success"]),
                "baseline_steps": int(base["elapsed"]),
                "discrete_success": bool(dis["success"]),
                "discrete_steps": int(dis["elapsed"]),
                "switched": int(st.n_switched),
                "considered": int(st.n_considered),
                "reasons": dict(st.reasons),
                "last_gain": st.last_gain,
                # if nothing was switched the two traces MUST be identical;
                # this column audits the paired setup itself.
                "identical_when_no_switch": (
                    st.n_switched > 0 or trace_key(base) == trace_key(dis)),
            }
            rows.append(row)
            print(f"[pilot] {t.task_uid} base={row['baseline_success']}"
                  f"({row['baseline_steps']}) disc={row['discrete_success']}"
                  f"({row['discrete_steps']}) switched={row['switched']} "
                  f"reasons={row['reasons']}", flush=True)

    rescued = sum(1 for r in rows
                  if not r["baseline_success"] and r["discrete_success"])
    destroyed = sum(1 for r in rows
                    if r["baseline_success"] and not r["discrete_success"])
    summary = {
        "n_tasks": len(rows),
        "baseline_success": sum(1 for r in rows if r["baseline_success"]),
        "discrete_success": sum(1 for r in rows if r["discrete_success"]),
        "rescued": rescued,
        "destroyed": destroyed,
        "net": rescued - destroyed,
        "n_switched": sum(r["switched"] for r in rows),
        "equivalence_checks": checks,
        "config": {
            "epsilon": args.epsilon, "directions": args.directions,
            "margin": args.margin, "max_regression": args.max_regression,
            "goal_offset": args.goal_offset, "episodes": args.episodes,
        },
    }
    (out_dir / "pilot.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("saved", out_dir / "pilot.json")


if __name__ == "__main__":
    main()
