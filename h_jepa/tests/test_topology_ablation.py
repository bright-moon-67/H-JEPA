"""CPU tests for the V1.1 ablation switches (audit-only controls).

`oracle_localization` is privileged and must only ever change the xy feeding the
adapter; `anti_route` must keep the same graph/readout but pick a waypoint that
moves AWAY from the goal. Both default to off.
"""
import numpy as np
import torch

from goal_adapter import AdapterConfig, TopologyGoalAdapter
from hide_topology_ablation import install_anti_route
from topo_graph import build_observed_topology


def make_graph():
    # corridor 0-1-2-3-4-5 with an observed side branch 2 -> (2,1)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0)]
    xy = np.asarray(path, dtype=float) + 0.5
    g = build_observed_topology(
        xy, np.zeros(len(xy), dtype=int), np.arange(len(xy), dtype=int),
        resolution=1.0, checkpoint_sha256="abc", normalizer_sha256="def",
    )
    g.attach_live_representatives(
        xy,
        {
            k: np.zeros((len(xy), 3), dtype=np.float32)
            for k in ("embed_1", "embed_2", "embed_3")
        },
    )
    return g, xy


def test_anti_route_picks_the_farthest_neighbour():
    g, _ = make_graph()
    start = np.array([2.5, 0.5])
    goal = np.array([5.5, 0.5])
    goal_centre = g.center(g.locate(goal, radius=0.3, margin=0.0))
    normal = g.suggest(start, goal, attachment_radius=0.3,
                       attachment_margin=0.0, lookahead_cells=1)
    assert normal is not None
    install_anti_route(g)
    wrong = g.suggest(start, goal, attachment_radius=0.3,
                      attachment_margin=0.0, lookahead_cells=1)
    assert wrong is not None
    assert getattr(g, "ablation") == "anti_route"
    d_normal = np.linalg.norm(np.asarray(normal.waypoint_xy) - goal_centre)
    d_wrong = np.linalg.norm(np.asarray(wrong.waypoint_xy) - goal_centre)
    assert d_wrong > d_normal


def test_anti_route_keeps_default_behaviour_when_unattachable():
    g, xy = make_graph()
    install_anti_route(g)
    assert g.suggest(np.array([99.0, 99.0]), xy[-1],
                     attachment_radius=0.3, attachment_margin=0.0) is None


def test_oracle_flag_defaults_absent(monkeypatch):
    from hierarchical_solver import HierarchicalSolver
    s = object.__new__(HierarchicalSolver)
    assert not getattr(s, "hide_topology_oracle_localization", False)


def test_oracle_xy_coercion():
    from hierarchical_solver import _as_xy
    assert _as_xy(None) is None
    t = torch.tensor([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])
    out = _as_xy(t)
    assert out.shape == (2, 2) and out[1].tolist() == [3.0, 4.0]
    assert _as_xy(np.array([1.0, 2.0])).shape == (1, 2)
