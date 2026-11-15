#!/usr/bin/env python3
"""OFFLINE audit of the V1.1 observed topology (Gate 1 geometry / localization).

Uses the environment's maze map ONLY as an offline audit oracle. Nothing here
is available to the online controller: `topo_graph.py` never sees walls, BFS
or simulator XY at planning time.

Checks
  1. every grid bin maps to exactly one free maze cell, and the 26 bins are a
     bijection with the 26 free cells (no thin-wall cell merging);
  2. every directed edge joins two 4-adjacent free cells, so no edge crosses a
     wall and none is diagonal;
  3. directed edge completeness / strongly-connected structure, so a route
     exists for most ordered (start, goal) cell pairs;
  4. localization: decoded-XY error statistics measured on TRAIN live states
     against the bin geometry the graph will actually attach with.

  python scripts/audit_topology_graph.py \
    --graph artifacts/hide/topology_train_live.npz \
    --dense-train-xy artifacts/hide/dense_train_xy.npz \
    --head artifacts/hide/nav_projector.pt \
    --output artifacts/hide/topology_graph_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import OmegaConf

from topo_graph import load_topology


def maze_map(cfg) -> np.ndarray:
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    world_cfg.num_envs = 1
    world_cfg.max_episode_steps = 8
    world = swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),) * 2)
    try:
        env = world.envs.unwrapped.envs[0]
        base = env
        while not hasattr(base, "maze_map") and getattr(base, "env", None) is not None:
            base = base.env
        return np.asarray(base.maze_map), world
    finally:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--dense-train-xy", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    world_cfg.num_envs = 1
    world_cfg.max_episode_steps = 8
    world = swm.World(**world_cfg, image_shape=(int(cfg.eval.img_size),) * 2)
    try:
        env = world.envs.unwrapped.envs[0]
        base = env
        while not hasattr(base, "maze_map") and getattr(base, "env", None) is not None:
            base = base.env
        mm = np.asarray(base.maze_map)
        from stable_worldmodel.world import _find_env_method

        xy_to_ij = _find_env_method(env, "xy_to_ij")
        if xy_to_ij is None:
            raise RuntimeError("env has no xy_to_ij; cannot audit wall crossings")
    finally:
        world.close()

    graph = load_topology(str(args.graph))
    width = graph.resolution
    free = {tuple(map(int, c)) for c in np.argwhere(mm == 0)}

    # --- 1. bin -> maze cell identity --------------------------------------
    dense = np.load(args.dense_train_xy, allow_pickle=False)
    xy = np.asarray(dense["xy"], dtype=np.float64)
    bins = np.floor((xy - graph.origin[None, :]) / width).astype(np.int64)
    node_of_bin = {tuple(map(int, c)): i for i, c in enumerate(graph.cells)}

    bin_ijs: dict[int, set[tuple[int, int]]] = {}
    bin_counts: dict[int, int] = {}
    for row, b in enumerate(bins):
        node = node_of_bin.get(tuple(map(int, b)))
        if node is None:
            continue
        ij = tuple(int(x) for x in np.asarray(xy_to_ij(xy[row])).reshape(-1)[:2])
        bin_ijs.setdefault(node, set()).add(ij)
        bin_counts[node] = bin_counts.get(node, 0) + 1

    merged_nodes = {n: sorted(s) for n, s in bin_ijs.items() if len(s) != 1}
    illegal_nodes = {
        n: sorted(s)
        for n, s in bin_ijs.items()
        if any(ij not in free for ij in s)
        or len(s) > 1
    }
    node_ij = {n: next(iter(s)) for n, s in bin_ijs.items() if len(s) == 1}

    # --- 2. edge wall / adjacency test -------------------------------------
    bad_edges, diagonal_edges = [], []
    for (a, b), sup in zip(graph.edges, graph.support):
        ia, ib = node_ij.get(int(a)), node_ij.get(int(b))
        if ia is None or ib is None:
            bad_edges.append({"edge": [int(a), int(b)], "reason": "unmapped_bin"})
            continue
        manhattan = abs(ia[0] - ib[0]) + abs(ia[1] - ib[1])
        if manhattan != 1:
            diagonal_edges.append(
                {"edge": [int(a), int(b)], "ij": [list(ia), list(ib)], "manhattan": manhattan}
            )
        if ia not in free or ib not in free:
            bad_edges.append({"edge": [int(a), int(b)], "reason": "wall_cell", "ij": [list(ia), list(ib)]})

    # --- 3. extent of the observed directed graph --------------------------
    pairs = {(int(a), int(b)) for a, b in graph.edges}
    missing_reverse = [(a, b) for (a, b) in pairs if (b, a) not in pairs]
    nodes = len(graph.cells)
    reach = np.zeros((nodes, nodes), dtype=bool)
    for src in range(nodes):
        reach[src, src] = True
        stack = [src]
        while stack:
            u = stack.pop()
            for v, _, _ in graph.adj[u]:
                if not reach[src, v]:
                    reach[src, v] = True
                    stack.append(v)
    reachable_pairs = int(reach.sum())
    strongly_connected = bool(reach.all())

    # --- 4. localization error against the frozen bin geometry -------------
    import torch

    from hide_nav import NavigationProjector

    payload = torch.load(args.head, map_location="cpu", weights_only=True)
    head = NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    live = np.load(
        Path("artifacts/hide/live_train_states_v11.npz"), allow_pickle=False
    )
    z3 = np.asarray(live["embed_3"], dtype=np.float32)
    true_xy = np.asarray(live["xy"], dtype=np.float64)
    with torch.inference_mode():
        decoded = head(torch.from_numpy(z3)).cpu().numpy().astype(np.float64)
    err = np.linalg.norm(decoded - true_xy, axis=1)
    centres = np.stack(
        [graph.center(i) for i in range(nodes)]
    )
    d_true = np.linalg.norm(
        true_xy[:, None, :] - centres[None, :, :], axis=2
    ) if nodes < 64 else None
    # fraction of decoded positions whose nearest lattice centre is the true one
    d_dec = (
        np.linalg.norm(true_xy[:, None, :] - centres[None, :, :], axis=2)
        if nodes < 64
        else None
    )
    nearest_true = np.argmin(d_dec, axis=1)
    nearest_dec = np.argmin(
        np.linalg.norm(decoded[:, None, :] - centres[None, :, :], axis=2), axis=1
    )
    same_cell = float((nearest_true == nearest_dec).mean())
    # conservative attachment reproduction of ObservedTopology.locate
    radius, margin = width / 2.0, 0.0
    attach_ok = 0
    for i in range(len(decoded)):
        order = np.argsort(np.linalg.norm(centres - decoded[i][None, :], axis=1), kind="stable")
        if np.linalg.norm(centres[order[0]] - decoded[i]) <= radius:
            attach_ok += 1
    report = {
        "gate": "topology_geometry_and_localization",
        "grid_width": width,
        "origin": graph.origin.tolist(),
        "num_nodes": nodes,
        "num_free_maze_cells": len(free),
        "node_cell_bijection": bool(
            len(node_ij) == nodes
            and {v for v in node_ij.values()} == free
        ),
        "nodes_with_single_maze_cell": int(len(node_ij)),
        "merged_bins": merged_nodes,
        "illegal_bins": illegal_nodes,
        "num_directed_edges": int(len(graph.edges)),
        "edge_support_total": int(graph.support.sum()),
        "bad_edges": bad_edges,
        "diagonal_edges": diagonal_edges,
        "missing_reverse_edges": [list(e) for e in missing_reverse],
        "reachable_ordered_pairs": reachable_pairs,
        "ordered_pairs_total": int(nodes * nodes),
        "strongly_connected": strongly_connected,
        "localization": {
            "num_states": int(len(err)),
            "median_xy_error": float(np.median(err)),
            "p90_xy_error": float(np.quantile(err, 0.9)),
            "max_xy_error": float(err.max()),
            "decoded_nearest_centre_matches_true": same_cell,
            "attachable_fraction_radius_width_over_2": attach_ok / len(decoded),
        },
        "wall_free_edges": len(bad_edges) == 0 and len(diagonal_edges) == 0,
        "graph_sha256": __import__("hide_integration").sha256(args.graph),
        "note": "maze_map used offline as audit oracle only; never online",
    }
    report["status"] = (
        "PASS"
        if (
            report["node_cell_bijection"]
            and not report["merged_bins"]
            and report["wall_free_edges"]
            and report["localization"]["decoded_nearest_centre_matches_true"] >= 0.95
        )
        else "NO-GO"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("NO-GO: topology geometry audit failed")


if __name__ == "__main__":
    main()
