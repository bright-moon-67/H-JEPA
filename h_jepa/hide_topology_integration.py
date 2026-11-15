"""HiDe V1.1 topology waypoint installation (fail-closed, default OFF).

Adds `install_topology_from_cfg()` next to the V1 navigation-cost installer.
Nothing here runs unless `hide_topology.enabled=true`, and enabling requires

  * a TRAIN-only observed graph whose checkpoint/normalizer hashes match,
  * a Gate-1 geometry audit file (status PASS) hashing that exact graph,
  * a Gate-2 local-controller certification file (status PASS) hashing that
    exact graph and reporting at least `min_certified_successes` successes,
  * the frozen B2 XY readout whose metadata matches the same checkpoint.

The reference library only checks that a certification id is present; these
files are what make that id true.
"""
from __future__ import annotations

import torch

from hide_nav import NavigationProjector, prepare_frozen_projector
from goal_adapter import AdapterConfig, TopologyGoalAdapter
from topo_graph import load_topology

from hide_integration import (
    _check_same, _load_json, _require_existing_file, sha256,
)

TOPOLOGY_LEVELS = (1, 2, 3)


def install_topology_from_cfg(cfg, solver, *, checkpoint, normalizer):
    """Attach the topology adapter only when every train-only gate passed."""
    spec = cfg.get("hide_topology", None)
    if spec is None or not bool(spec.get("enabled", False)):
        return None
    if getattr(solver, "hide_topology_adapter", None) is not None:
        raise RuntimeError("Topology adapter installed twice")
    if sorted(solver.level_solvers) != [1, 2, 3]:
        raise ValueError("HiDe V1.1 only supports the 3-level AntMaze model")

    from pathlib import Path

    checkpoint = Path(checkpoint)
    normalizer = Path(normalizer)
    checkpoint_hash = sha256(checkpoint)
    normalizer_hash = sha256(normalizer)

    graph_file = _require_existing_file(spec, "graph_path")
    readout_file = _require_existing_file(spec, "readout_path")
    geometry_file = _require_existing_file(spec, "geometry_audit_path")
    certification_file = _require_existing_file(spec, "certification_path")
    graph_hash = sha256(graph_file)

    graph = load_topology(str(graph_file))
    graph.verify(checkpoint_hash, normalizer_hash)

    geometry = _load_json(geometry_file)
    if geometry.get("status") != "PASS":
        raise ValueError("Gate-1 topology geometry audit did not PASS")
    _check_same("geometry/graph", geometry.get("graph_sha256"), graph_hash)
    if not bool(geometry.get("wall_free_edges", False)):
        raise ValueError("Gate-1 audit reported wall-crossing edges")
    if not bool(geometry.get("node_cell_bijection", False)):
        raise ValueError("Gate-1 audit reported merged/illegal grid cells")

    cert = _load_json(certification_file)
    if cert.get("status") != "PASS":
        raise ValueError("Gate-2 local-controller certification did not PASS")
    _check_same("certification/graph", cert.get("graph_sha256"), graph_hash)
    _check_same("certification/checkpoint", cert.get("checkpoint_sha256"), checkpoint_hash)
    _check_same("certification/normalizer", cert.get("normalizer_sha256"), normalizer_hash)
    evidence_id = str(spec.get("certification_evidence_id", "") or "")
    if not evidence_id:
        raise ValueError("hide_topology.certification_evidence_id is required")
    _check_same("certification/evidence_id", cert.get("evidence_id"), evidence_id)
    min_successes = int(spec.get("min_certified_successes", 9))
    if int(cert.get("successes", -1)) < min_successes:
        raise ValueError(
            f"Gate-2 certified {cert.get('successes')} local waypoints, "
            f"need >= {min_successes}"
        )
    if int(cert.get("false_arrivals", 1)) != 0:
        raise ValueError("Gate-2 certified a non-zero false-arrival count")

    payload = torch.load(readout_file, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    if meta.get("source_split") != "train" or meta.get("render_domain") != "live_eval_pipeline":
        raise ValueError("Topology readout must come from LIVE TRAIN states")
    if meta.get("real_state_gate") is not True:
        raise ValueError("Topology readout failed its real-state gate")
    _check_same("readout/checkpoint", meta.get("checkpoint_sha256"), checkpoint_hash)
    _check_same("readout/normalizer", meta.get("normalizer_sha256"), normalizer_hash)
    readout = NavigationProjector(**payload["architecture"])
    readout.load_state_dict(payload["state_dict"], strict=True)
    dev = next(solver.level_models[2].parameters()).device
    prepare_frozen_projector(readout, dev)

    config = AdapterConfig(
        enabled=True,
        certification_evidence_id=evidence_id,
        checkpoint_sha256=checkpoint_hash,
        normalizer_sha256=normalizer_hash,
        attach_radius=float(spec.get("attach_radius", 2.0)),
        attach_margin=float(spec.get("attach_margin", 0.5)),
        lookahead_cells=int(spec.get("lookahead_cells", 1)),
        arrival_radius=float(spec.get("arrival_radius", 2.0)),
        max_waypoint_calls=int(spec.get("max_waypoint_calls", 12)),
    )
    adapter = TopologyGoalAdapter(graph, config)
    adapter.reset(int(getattr(solver, "n_envs", 1) or 1))
    solver.hide_topology_adapter = adapter
    solver.hide_readout = readout
    return {
        "checkpoint_sha256": checkpoint_hash,
        "normalizer_sha256": normalizer_hash,
        "graph_sha256": graph_hash,
        "readout_sha256": sha256(readout_file),
        "geometry_audit_sha256": sha256(geometry_file),
        "certification_sha256": sha256(certification_file),
        "certification_evidence_id": evidence_id,
        "grid_width": graph.resolution,
        "num_nodes": int(len(graph.cells)),
        "num_edges": int(len(graph.edges)),
        "config": {
            "attach_radius": config.attach_radius,
            "attach_margin": config.attach_margin,
            "lookahead_cells": config.lookahead_cells,
            "arrival_radius": config.arrival_radius,
            "max_waypoint_calls": config.max_waypoint_calls,
        },
    }
