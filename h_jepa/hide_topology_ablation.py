"""Ablation switches for HiDe V1.1 (audit only, default OFF).

The reference asks for a comparison against "a planner with the same offline
supervised XY data". A literal coordinate-only planner cannot be built here:
the ant is torque-controlled and there is no second low-level controller to
drive it, so the two feasible controls are implemented instead, both on the
SAME graph and the SAME frozen checkpoint:

1. ``oracle_localization`` (UPPER BOUND, deliberately PRIVILEGED)
   Replaces the learned 2-D readout with the simulator's true ``xy`` /
   ``goal_xy`` for attachment and arrival. This is privileged information and
   exists only to measure what the readout costs. It must never be reported as
   the method, and the installer records it in the run manifest.

2. ``anti_route`` (INFORMATION-MATCHED CONTROL)
   Keeps the learned readout and the graph but deliberately selects the
   neighbour that is FARTHEST from the goal. If the development gain came from
   "any goal perturbation" rather than from correct routing, this control would
   gain as well.

Neither switch changes the checkpoint, the solver, the sample counts, the
iteration count or the execution cadence.
"""
from __future__ import annotations

import numpy as np

from topo_graph import RouteSuggestion


def install_anti_route(graph) -> None:
    """Shadow `graph.suggest` with a deliberately wrong-route chooser."""

    def suggest(
        xy_current,
        xy_goal,
        *,
        attachment_radius: float,
        attachment_margin: float,
        lookahead_cells: int = 1,
    ):
        start = graph.locate(
            xy_current, radius=attachment_radius, margin=attachment_margin
        )
        goal = graph.locate(xy_goal, radius=attachment_radius, margin=attachment_margin)
        if start is None or goal is None or start == goal:
            return None
        goal_xy = graph.center(goal)
        best, best_d = None, -1.0
        for node, _, _ in graph.adj[start]:
            if node not in graph.reps:
                continue
            d = float(np.linalg.norm(graph.center(node) - goal_xy))
            if d > best_d:
                best, best_d = node, d
        if best is None:
            return None
        return RouteSuggestion(
            nodes=(start, best, goal),
            waypoint_node=int(best),
            waypoint_xy=tuple(float(v) for v in graph.center(best)),
            cost=float("inf"),
        )

    graph.suggest = suggest  # instance-level shadow; audit runs only
    graph.ablation = "anti_route"


def install_oracle_localization(solver, normalizer_path) -> None:
    """Flag the solver to use privileged simulator xy for attachment/arrival.

    `info_dict['xy']` and `['goal_xy']` are NORMALIZED by the policy processor
    (the frozen normalizer has an `xy` entry), so the statistics are loaded
    here and inverted before the values reach the adapter.
    """
    import torch

    payload = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    stats = payload["stats"]["xy"]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)[:2]
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)[:2]
    if not (np.all(np.isfinite(mean)) and np.all(std > 0)):
        raise ValueError("Invalid xy normalization statistics")
    solver.hide_topology_xy_stats = (mean, std)
    solver.hide_topology_oracle_localization = True
