"""Certified memory-graph waypoints for the EXISTING frozen H-JEPA L3 planner.

IMPORTANT:
  - graph goals are supplied to the original L3 Goal Cost, not to an oracle;
  - L3's goal embedding is replaced consistently with downstream anchoring;
  - skipped L3 and no-graph cases are strict no-ops;
  - online controller never accepts XY or ground-truth maze maps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pahrl_memory import GraphMemory, Route, _unit


@dataclass(frozen=True)
class WaypointConfig:
    enabled: bool = False
    reach_cosine_distance: float = 0.035
    k_attach: int = 3
    max_waypoint_calls: int = 6

    def __post_init__(self):
        if self.k_attach < 1 or self.max_waypoint_calls < 1:
            raise ValueError("k_attach/max_waypoint_calls must be positive")
        if not 0 < self.reach_cosine_distance < 2:
            raise ValueError("Invalid reach_cosine_distance")


class GraphWaypointController:
    def __init__(self, graph: GraphMemory, cfg: WaypointConfig):
        self.graph = graph
        self.cfg = cfg
        if cfg.enabled and not graph.certified:
            raise ValueError(
                "PAHRL online mode is blocked until graph attachments are "
                "validated using frozen H-JEPA local reach tests."
            )
        self._state: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []

    def reset(self, n_envs: int) -> None:
        if n_envs < 1:
            raise ValueError("Invalid env count")
        self._state = [dict() for _ in range(n_envs)]
        self.records = []

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = _unit(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
        b = _unit(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
        return float(max(0.0, 1.0 - a @ b))

    def target_embeddings(
        self,
        current: torch.Tensor,
        true_goal: torch.Tensor,
        *,
        steps_taken: int,
        call_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (replacement L3 goals [B,1,D], bool mask [B]).

        Replanning is only at ORIGINAL policy cadence. A waypoint only advances
        after being reached according to a *calibrated* latent threshold.
        No waypoint route => mask false, caller retains original task goal.
        """
        if current.ndim != 3 or true_goal.ndim != 3:
            raise ValueError("Expected H-JEPA L3 goals [B,1,D]")
        if current.shape != true_goal.shape or current.shape[1] != 1:
            raise ValueError("Bad H-JEPA L3 goal/current shape")
        B, _, D = current.shape
        if len(self._state) != B or steps_taken == 0:
            self.reset(B)
        replacement = true_goal.detach().clone()
        mask = torch.zeros(B, dtype=torch.bool, device=true_goal.device)
        if not self.cfg.enabled:
            return replacement, mask

        cur_cpu = current.detach().cpu().float().numpy()[:, 0]
        goal_cpu = true_goal.detach().cpu().float().numpy()[:, 0]
        for b in range(B):
            cur, goal = cur_cpu[b], goal_cpu[b]
            state = self._state[b]
            reason = "continue"
            if (
                not state
                or not np.allclose(state["true_goal"], goal, rtol=0, atol=1e-6)
            ):
                route: Route | None = self.graph.route(
                    cur, goal, k=self.cfg.k_attach
                )
                # Exclude the local start attachment and the final near-goal node:
                # once all graph intermediates are reached, original goal resumes.
                intermediates = () if route is None else route.nodes[1:-1]
                state = {
                    "true_goal": goal.copy(),
                    "route": route,
                    "waypoints": intermediates,
                    "cursor": 0,
                    "wait_calls": 0,
                    "disabled": False,
                }
                self._state[b] = state
                reason = "new_route" if intermediates else "no_usable_route"

            if not state["disabled"] and state["waypoints"]:
                waypoints = state["waypoints"]
                pos = int(state["cursor"])
                # Advance only when actual current embedding reached waypoint.
                while pos < len(waypoints):
                    node = int(waypoints[pos])
                    d = self._cosine_distance(cur, self.graph.z[node])
                    if d > self.cfg.reach_cosine_distance:
                        break
                    pos += 1
                    state["wait_calls"] = 0
                state["cursor"] = pos
                if pos < len(waypoints):
                    if state["wait_calls"] >= self.cfg.max_waypoint_calls:
                        state["disabled"] = True
                        reason = "waypoint_timeout_fallback"
                    else:
                        node = int(waypoints[pos])
                        replacement[b, 0] = torch.as_tensor(
                            self.graph.z[node],
                            dtype=true_goal.dtype,
                            device=true_goal.device,
                        )
                        mask[b] = True
                        state["wait_calls"] += 1
                        reason = f"waypoint_{node}"
                else:
                    reason = "route_complete_use_task_goal"

            self.records.append({
                "env_idx": b,
                "steps_taken": int(steps_taken),
                "call_idx": int(call_idx),
                "goal_overridden": bool(mask[b]),
                "reason": reason,
                "route": (
                    list(state["route"].nodes)
                    if state.get("route") is not None else []
                ),
                "cursor": int(state.get("cursor", 0)),
                "wait_calls": int(state.get("wait_calls", 0)),
            })
        return replacement, mask
