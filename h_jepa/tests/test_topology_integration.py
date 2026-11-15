"""CPU contract tests for the V1.1 topology waypoint integration.

They pin the properties the pilot depends on:
  * absent / disabled / L3-horizon-1 paths return the ORIGINAL goal dict object;
  * an active waypoint replaces ALL three level goals for exactly the masked
    envs, and never mutates the caller's tensors;
  * the installer refuses to enable without a PASSing geometry audit and a
    PASSing local-controller certification for the same graph.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from goal_adapter import AdapterConfig
from hide_nav import NavigationProjector
from hide_integration import sha256
from hide_topology_integration import install_topology_from_cfg
from hierarchical_solver import HierarchicalSolver
from topo_graph import build_observed_topology, save_topology


def make_graph(tmp_path):
    cells = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    xy = np.asarray(cells, dtype=float) + 0.5
    g = build_observed_topology(
        xy, np.zeros(len(xy), dtype=int), np.arange(len(xy), dtype=int),
        resolution=1.0, evidence_id="train", checkpoint_sha256="ckpt",
        normalizer_sha256="norm",
    )
    g.attach_live_representatives(
        xy,
        {
            "embed_1": np.stack([xy[:, 0], xy[:, 1], xy[:, 0] + 1], axis=1),
            "embed_2": np.stack([xy[:, 0], xy[:, 1], xy[:, 1] + 1], axis=1),
            "embed_3": np.stack([xy[:, 0], xy[:, 1], xy[:, 0] - xy[:, 1]], axis=1),
        },
    )
    path = tmp_path / "graph.npz"
    save_topology(g, str(path))
    return g, xy, path


def solver_stub(adapter=None, readout=None, levels=3):
    s = object.__new__(HierarchicalSolver)
    s.level_models = [torch.nn.Linear(1, 1) for _ in range(levels)]
    s.level_solvers = {i: SimpleNamespace(model=torch.nn.Module()) for i in (1, 2, 3)}
    s.hide_nav_enabled = False
    s.hide_topology_adapter = adapter
    s.hide_readout = readout
    return s


def goals(b=1, d=3):
    return {
        "embed_1": torch.arange(b * d, dtype=torch.float32).reshape(b, 1, d) + 1,
        "embed_2": torch.arange(b * d, dtype=torch.float32).reshape(b, 1, d) + 11,
        "embed_3": torch.arange(b * d, dtype=torch.float32).reshape(b, 1, d) + 21,
    }


def obs(b=1, d=3):
    return {f"embed_{i}": torch.zeros(b, 1, d) for i in (1, 2, 3)}


def identity_readout():
    """L3 latent -> XY: takes the first two coordinates."""

    class Readout(torch.nn.Module):
        def forward(self, z):
            return z[..., :2]

    return Readout()


def active_solver(graph, **overrides):
    cfg = dict(
        enabled=True,
        certification_evidence_id="gate2",
        checkpoint_sha256="ckpt",
        normalizer_sha256="norm",
        attach_radius=0.3,
        attach_margin=0.0,
        lookahead_cells=2,
        arrival_radius=0.3,
        max_waypoint_calls=3,
    )
    cfg.update(overrides)
    from goal_adapter import TopologyGoalAdapter

    adapter = TopologyGoalAdapter(graph, AdapterConfig(**cfg))
    adapter.reset(1)
    return solver_stub(adapter=adapter, readout=identity_readout())


def test_absent_adapter_returns_same_object():
    s = solver_stub()
    g = goals()
    out = s._apply_topology_waypoint(
        obs(), g, horizon_schedules={3: [5]}, call_idx=0
    )
    assert out is g


def test_disabled_adapter_returns_same_object(tmp_path):
    from goal_adapter import TopologyGoalAdapter

    g, _, _ = make_graph(tmp_path)
    s = solver_stub(
        adapter=TopologyGoalAdapter(g, AdapterConfig(enabled=False)),
        readout=identity_readout(),
    )
    goal = goals()
    out = s._apply_topology_waypoint(
        obs(), goal, horizon_schedules={3: [5]}, call_idx=0
    )
    assert out is goal


def test_l3_horizon_one_keeps_real_goal(tmp_path):
    g, _, _ = make_graph(tmp_path)
    s = active_solver(g)
    goal = goals()
    out = s._apply_topology_waypoint(
        obs(), goal, horizon_schedules={3: [1]}, call_idx=0
    )
    assert out is goal


def test_only_three_level_models_are_supported(tmp_path):
    g, _, _ = make_graph(tmp_path)
    s = active_solver(g)
    s.level_models = [torch.nn.Linear(1, 1) for _ in range(2)]
    goal = goals()
    assert s._apply_topology_waypoint(
        obs(), goal, horizon_schedules={3: [5]}, call_idx=0
    ) is goal


def test_active_waypoint_replaces_all_levels_without_mutating(tmp_path):
    g, xy, _ = make_graph(tmp_path)
    s = active_solver(g)
    goal = goals()
    before = {k: v.clone() for k, v in goal.items()}
    live = xy[0:1]
    task_goal = xy[-1:]
    s.hide_readout = lambda z: torch.as_tensor(
        np.asarray(live if float(z.sum()) == 0 else task_goal, dtype=np.float32)
    )
    out = s._apply_topology_waypoint(
        obs(), goal, horizon_schedules={3: [5]}, call_idx=0
    )
    assert out is not goal
    for k in ("embed_1", "embed_2", "embed_3"):
        torch.testing.assert_close(goal[k], before[k])  # caller data intact
    changed = sum(
        not torch.equal(out[k], goal[k]) for k in ("embed_1", "embed_2", "embed_3")
    )
    assert changed == 3  # every level moves to the SAME waypoint
    rep = g.reps[int(s.hide_topology_adapter.records[-1]["waypoint_node"])]
    for k in ("embed_1", "embed_2", "embed_3"):
        torch.testing.assert_close(
            out[k][0, 0], torch.as_tensor(rep[k], dtype=torch.float32)
        )


def test_per_env_mask_isolation(tmp_path):
    g, xy, _ = make_graph(tmp_path)
    s = active_solver(g)
    s.hide_topology_adapter.reset(2)
    goal = goals(b=2)

    class Readout(torch.nn.Module):
        def forward(self, z):
            # obs embeddings are all-zero -> live position; goal is non-zero.
            is_live = float(z.abs().sum()) == 0.0
            row = torch.as_tensor(
                np.array([0.5, 0.5] if is_live else [0.5, 2.5], dtype=np.float32)
            )
            out = row[None, :].repeat(z.shape[0], 1)
            # env 1 lives far outside the graph in both cases
            out[1] = torch.tensor([999.0, 999.0])
            return out

    s.hide_readout = Readout()
    out = s._apply_topology_waypoint(
        obs(b=2), goal, horizon_schedules={3: [5]}, call_idx=0
    )
    assert out is not goal
    assert changed_all(out, goal, 0)
    for k in ("embed_1", "embed_2", "embed_3"):
        torch.testing.assert_close(out[k][1], goal[k][1])  # untouched env


def changed_all(out, goal, env):
    return all(
        not torch.equal(out[k][env], goal[k][env])
        for k in ("embed_1", "embed_2", "embed_3")
    )


def assets(tmp_path, monkeypatch):
    ckpt = tmp_path / "model.ckpt"
    norm = tmp_path / "normalizer.pt"
    ckpt.write_bytes(b"frozen checkpoint")
    norm.write_bytes(b"frozen normalizer")
    readout = tmp_path / "readout.pt"
    torch.save(
        {
            "architecture": {"latent_dim": 384, "hidden_dim": 128},
            "state_dict": NavigationProjector().state_dict(),
            "metadata": {
                "source_split": "train",
                "render_domain": "live_eval_pipeline",
                "checkpoint_sha256": sha256(ckpt),
                "normalizer_sha256": sha256(norm),
                "real_state_gate": True,
            },
        },
        readout,
    )
    g = build_observed_topology(
        np.array([[0.5, 0.5], [1.5, 0.5]]), np.array([0, 0]), np.array([0, 1]),
        resolution=1.0, checkpoint_sha256=sha256(ckpt),
        normalizer_sha256=sha256(norm),
    )
    g.attach_live_representatives(
        np.array([[0.5, 0.5], [1.5, 0.5]]),
        {k: np.zeros((2, 384), dtype=np.float32) for k in ("embed_1", "embed_2", "embed_3")},
    )
    graph = tmp_path / "graph.npz"
    save_topology(g, str(graph))
    geom = tmp_path / "geometry.json"
    geom.write_text(
        json.dumps(
            {
                "status": "PASS",
                "graph_sha256": sha256(graph),
                "wall_free_edges": True,
                "node_cell_bijection": True,
            }
        )
    )
    cert = tmp_path / "cert.json"
    cert.write_text(
        json.dumps(
            {
                "status": "PASS",
                "graph_sha256": sha256(graph),
                "checkpoint_sha256": sha256(ckpt),
                "normalizer_sha256": sha256(norm),
                "evidence_id": "gate2_local_readiness_v1",
                "successes": 12,
                "false_arrivals": 0,
            }
        )
    )
    cfg = {
        "hide_topology": {
            "enabled": True,
            "graph_path": str(graph),
            "readout_path": str(readout),
            "geometry_audit_path": str(geom),
            "certification_path": str(cert),
            "certification_evidence_id": "gate2_local_readiness_v1",
        }
    }
    return ckpt, norm, cfg, geom, cert


def test_installer_default_off_is_inert(tmp_path):
    s = solver_stub()
    install_topology_from_cfg(
        {}, s, checkpoint=tmp_path / "absent", normalizer=tmp_path / "absent"
    )
    assert s.hide_topology_adapter is None and s.hide_readout is None


def test_installer_requires_certification(tmp_path):
    ckpt, norm, cfg, _, cert = assets(tmp_path, None)
    cert.write_text(json.dumps({"status": "NO-GO"}))
    s = solver_stub()
    with pytest.raises(ValueError, match="Gate-2"):
        install_topology_from_cfg(cfg, s, checkpoint=ckpt, normalizer=norm)
    assert s.hide_topology_adapter is None


def test_installer_rejects_geometry_failure(tmp_path):
    ckpt, norm, cfg, geom, _ = assets(tmp_path, None)
    geom.write_text(
        json.dumps(
            {
                "status": "PASS",
                "graph_sha256": sha256(tmp_path / "graph.npz"),
                "wall_free_edges": False,
                "node_cell_bijection": True,
            }
        )
    )
    with pytest.raises(ValueError, match="wall-crossing"):
        install_topology_from_cfg(
            cfg, solver_stub(), checkpoint=ckpt, normalizer=norm
        )


def test_installer_rejects_too_few_certified_waypoints(tmp_path):
    ckpt, norm, cfg, _, cert = assets(tmp_path, None)
    payload = json.loads(cert.read_text())
    payload["successes"] = 5
    cert.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="need >= 9"):
        install_topology_from_cfg(
            cfg, solver_stub(), checkpoint=ckpt, normalizer=norm
        )


def test_installer_attaches_frozen_adapter(tmp_path):
    ckpt, norm, cfg, _, _ = assets(tmp_path, None)
    s = solver_stub()
    s.level_models = [torch.nn.Linear(1, 1) for _ in range(3)]
    s._n_envs = 2
    manifest = install_topology_from_cfg(cfg, s, checkpoint=ckpt, normalizer=norm)
    assert manifest["certification_evidence_id"] == "gate2_local_readiness_v1"
    assert s.hide_topology_adapter.config.enabled
    assert not any(p.requires_grad for p in s.hide_readout.parameters())
    assert len(s.hide_topology_adapter._env_state) == 2
    with pytest.raises(RuntimeError, match="twice"):
        install_topology_from_cfg(cfg, s, checkpoint=ckpt, normalizer=norm)
