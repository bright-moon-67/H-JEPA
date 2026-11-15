#!/usr/bin/env python3
"""Gate A: dev-task route coverage from ONLINE decoded positions only.

Runs the real evaluation path (`run_planning_eval`) with a 1-step budget and
captures the two genuine all-level encodings from `HierarchicalSolver.solve`.
The graph route is then produced exactly as the online adapter would:

  xy_current = readout(embed_3(REAL current observation))
  xy_goal    = readout(embed_3(REAL goal observation))
  route      = graph.suggest(xy_current, xy_goal, frozen thresholds)

True simulator XY is never used, here or online. Coverage counts a task as
routed only if the graph attaches both endpoints and returns a nontrivial
waypoint; every routed path is then audited against the offline maze map.

  python scripts/export_topology_dev_queries.py \
    --checkpoint ... --graph artifacts/hide/topology_train_live.npz \
    --readout artifacts/hide/nav_projector.pt \
    --tasks audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
    --output artifacts/hide/topology_dev_coverage.json --env-seeds 1000,2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from hierarchical_solver import HierarchicalSolver
from hide_nav import NavigationProjector
from planning_eval import run_planning_eval
from topo_graph import load_topology

CAPTURE: dict = {}


def install_capture():
    original = HierarchicalSolver._encode_all_levels

    def patched(self, info_dict, key="pixels"):
        out = original(self, info_dict, key=key)
        if key not in CAPTURE:  # first planning call = the t=0 observation
            CAPTURE[key] = {k: v.detach().float().cpu().numpy() for k, v in out.items()}
        return out

    HierarchicalSolver._encode_all_levels = patched
    return original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--readout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--env-seeds", default="1000,2000")
    ap.add_argument("--num-tasks", type=int, default=12)
    args = ap.parse_args()

    graph = load_topology(str(args.graph))
    payload = torch.load(args.readout, map_location="cpu", weights_only=True)
    head = NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    # frozen thresholds (identical to config/eval/ant_l3_topo.yaml)
    attach_radius, attach_margin = 2.0, 0.5
    lookahead_cells = 1

    cfg0 = json.loads(Path("artifacts/hide/topology_train_live.summary.json").read_text())
    records = []
    for seed in [int(s) for s in args.env_seeds.split(",") if s.strip()]:
        CAPTURE.clear()
        original = install_capture()
        try:
            cfg = OmegaConf.load(args.config)
            cfg.policy = str(args.checkpoint.resolve())
            cfg.seed = 42
            cfg.load_eval_trajs_path = str(args.tasks)
            cfg.eval.num_eval = int(args.num_tasks)
            cfg.eval.eval_budget = 1
            # Only t=0 is captured; the task payload supplies goal_steps, so the
            # dataset goal-offset is irrelevant and must not force a longer env.
            cfg.eval.goal_offset_steps = 1
            cfg.eval.chunk_size = None
            cfg.eval.env_seed = int(seed)
            for level in (1, 2, 3):
                cfg.solver.solvers[f"level{level}"].batch_size = 2
            cfg.output.dir = f"topo_gate_a_env{seed}"
            metrics = run_planning_eval(cfg)
        finally:
            HierarchicalSolver._encode_all_levels = original
        if "pixels" not in CAPTURE or "goal" not in CAPTURE:
            raise RuntimeError(f"No encodings captured for seed {seed}")
        obs = CAPTURE["pixels"]
        goal = CAPTURE["goal"]
        with torch.inference_mode():
            xy_cur = head(torch.as_tensor(obs["embed_3"])[:, 0]).numpy().astype(np.float64)
            xy_goal = head(torch.as_tensor(goal["embed_3"])[:, 0]).numpy().astype(np.float64)
        assert len(xy_cur) == len(metrics["episode_successes"])
        for task in range(len(xy_cur)):
            s = graph.locate(xy_cur[task], radius=attach_radius, margin=attach_margin)
            g = graph.locate(xy_goal[task], radius=attach_radius, margin=attach_margin)
            suggestion = graph.suggest(
                xy_cur[task], xy_goal[task],
                attachment_radius=attach_radius,
                attachment_margin=attach_margin,
                lookahead_cells=lookahead_cells,
            )
            record = {
                "env_seed": int(seed),
                "task": int(task),
                "decoded_current_xy": xy_cur[task].tolist(),
                "decoded_goal_xy": xy_goal[task].tolist(),
                "start_node": None if s is None else int(s),
                "goal_node": None if g is None else int(g),
                "routed": suggestion is not None,
                "waypoint_node": None if suggestion is None else int(suggestion.waypoint_node),
                "waypoint_xy": None if suggestion is None else list(suggestion.waypoint_xy),
                "path_len": None if suggestion is None else len(suggestion.nodes),
                "path_cost": None if suggestion is None else float(suggestion.cost),
                "distance_to_goal_before": (
                    None if s is None or g is None
                    else float(np.linalg.norm(graph.center(s) - graph.center(g)))
                ),
                "distance_to_goal_after_waypoint": (
                    None if suggestion is None or g is None
                    else float(
                        np.linalg.norm(
                            np.asarray(suggestion.waypoint_xy) - graph.center(g)
                        )
                    )
                ),
                "path_nodes": None if suggestion is None else [int(n) for n in suggestion.nodes],
            }
            records.append(record)
        print(f"seed {seed}: captured {len(xy_cur)} tasks", flush=True)

    routed = [r for r in records if r["routed"]]
    # progress check: the waypoint must be strictly closer to the goal cell
    regressions = [
        r for r in routed
        if r["distance_to_goal_after_waypoint"] >= r["distance_to_goal_before"] - 1e-9
    ]
    report = {
        "gate": "dev_route_coverage",
        "graph": str(args.graph),
        "graph_summary": cfg0,
        "thresholds": {
            "attach_radius": attach_radius,
            "attach_margin": attach_margin,
            "lookahead_cells": lookahead_cells,
        },
        "num_queries": len(records),
        "num_routed": len(routed),
        "coverage": len(routed) / len(records),
        "non_progressing_waypoints": len(regressions),
        "records": records,
        "note": "decoded XY only; true simulator XY never used for routing",
    }
    report["status"] = (
        "PASS"
        if len(routed) >= 12 and len(regressions) == 0
        else "NO-GO"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(
        {k: report[k] for k in ("num_queries", "num_routed", "coverage",
                                "non_progressing_waypoints", "status")}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("NO-GO: dev route coverage gate failed")


if __name__ == "__main__":
    main()
