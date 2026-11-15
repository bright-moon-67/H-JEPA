#!/usr/bin/env python3
"""Validate proposed cross-episode graph links with frozen H-JEPA rollouts.

Each proposal is only a *hypothesis*.  A directed edge is written to
``validated_links.jsonl`` only when ``--trials-per-link`` independent local
rollouts all reach the memorized target physically, with the source physics
state replayed bit-exactly.

Run from H-JEPA/h_jepa with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from pahrl_local_reach import (
    LocalReachRunner,
    build_eval_stack,
    sha256_array,
)


def read_rows(dataset: Path, rows: list[int]) -> tuple[np.ndarray, ...]:
    with h5py.File(dataset, "r") as handle:
        order = np.argsort(rows)
        sorted_rows = np.asarray(rows, dtype=np.int64)[order]
        pixels = np.asarray(handle["pixels"][sorted_rows])
        qpos = np.asarray(handle["qpos"][sorted_rows], dtype=np.float64)
        qvel = np.asarray(handle["qvel"][sorted_rows], dtype=np.float64)
    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return pixels[restore], qpos[restore], qvel[restore]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--proposals", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rollout-log", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    p.add_argument("--trials-per-link", type=int, default=3)
    p.add_argument("--max-links", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--links-per-batch", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=91000)
    args = p.parse_args()

    with np.load(args.states, allow_pickle=False) as data:
        episode = np.asarray(data["episode"])
        steps = np.asarray(data["steps"])
        source_row = np.asarray(data["source_row"], dtype=np.int64)
        xy = np.asarray(data["xy_diagnostic_only"], dtype=np.float32)
    proposals = [
        json.loads(line)
        for line in args.proposals.read_text().splitlines()
        if line.strip()
    ]
    if not proposals:
        raise ValueError("No proposals to validate")
    proposals = proposals[: args.max_links]

    for record in proposals:
        a, b = int(record["from_node"]), int(record["to_node"])
        if episode[a] == episode[b]:
            raise ValueError(
                f"Proposal {a}->{b} is not cross-episode; links must join "
                "separate source trajectories"
            )
        record["from_episode"] = int(episode[a])
        record["to_episode"] = int(episode[b])
        record["from_step"] = int(steps[a])
        record["to_step"] = int(steps[b])

    cfg, policy = build_eval_stack(args.checkpoint, args.config)
    trials_per_link = max(3, int(args.trials_per_link))
    links_per_batch = max(1, int(args.links_per_batch))

    all_rows: list[dict] = []
    validated: list[dict] = []

    for start in range(0, len(proposals), links_per_batch):
        chunk = proposals[start:start + links_per_batch]
        node_ids = sorted(
            {int(r["from_node"]) for r in chunk}
            | {int(r["to_node"]) for r in chunk}
        )
        row_map = {node: int(source_row[node]) for node in node_ids}
        pixels, qpos, qvel = read_rows(args.dataset, list(row_map.values()))
        by_row = {
            row: (pixels[i], qpos[i], qvel[i])
            for i, row in enumerate(list(row_map.values()))
        }

        trials: list[dict] = []
        labels: list[dict] = []
        for record in chunk:
            a, b = int(record["from_node"]), int(record["to_node"])
            src_pix, src_qpos, src_qvel = by_row[row_map[a]]
            tgt_pix, tgt_qpos, tgt_qvel = by_row[row_map[b]]
            for trial_idx in range(trials_per_link):
                trials.append(
                    {
                        "from_node": a,
                        "to_node": b,
                        "trial": trial_idx,
                        "source_qpos": src_qpos,
                        "source_qvel": src_qvel,
                        "target_qpos": tgt_qpos,
                        "target_qvel": tgt_qvel,
                        "target_pixels": tgt_pix,
                        "target_embedding": None,  # filled below
                        "target_xy": xy[b].astype(np.float64),
                        "latent_cosine_distance": float(
                            record.get("latent_cosine_distance", float("nan"))
                        ),
                        "source_split": "train",
                    }
                )
                labels.append(record)

        # The link target latent is the graph node embedding itself.
        with np.load(args.states, allow_pickle=False) as state_npz:
            z = np.asarray(state_npz["z"], dtype=np.float32)
        for trial in trials:
            trial["target_embedding"] = z[int(trial["to_node"])]

        runner = LocalReachRunner(
            cfg,
            policy,
            max_steps=args.max_steps,
            base_seed=args.base_seed + 100 * start,
        )
        results = runner.run(trials)
        for row in results:
            all_rows.append(row)

        grouped: dict[tuple[int, int], list[dict]] = {}
        for row in results:
            grouped.setdefault((row["from_node"], row["to_node"]), []).append(row)
        for (a, b), rows in grouped.items():
            successes = sum(1 for r in rows if r["local_success"])
            ok = successes == len(rows) == trials_per_link and all(
                r["state_replay_ok"] for r in rows
            )
            mean_steps = (
                float(np.mean([r["arrival_step"] for r in rows]))
                if ok and all(r["arrival_step"] >= 0 for r in rows)
                else float("nan")
            )
            if ok:
                evidence = json.dumps(rows, sort_keys=True).encode()
                validated.append(
                    {
                        "from_node": a,
                        "to_node": b,
                        "successes": int(successes),
                        "trials": int(len(rows)),
                        "mean_steps": float(mean_steps),
                        "source_split": "train",
                        "evidence_sha256": sha256_array(
                            np.frombuffer(evidence, dtype=np.uint8)
                        ),
                    }
                )
            print(
                f"link {a}->{b}: {successes}/{len(rows)} "
                f"{'VALIDATED' if ok else 'rejected'}"
            )

    args.rollout_log.parent.mkdir(parents=True, exist_ok=True)
    with args.rollout_log.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in validated:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"validated {len(validated)}/{len(proposals)} proposed links")
    print("rollouts ->", args.rollout_log)
    print("links ->", args.output)


if __name__ == "__main__":
    main()
