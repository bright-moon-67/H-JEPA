"""Live-state topology subgoal adapter; local controller remains UNCHANGED.

Use after `obs_embeddings = _encode_all_levels(raw obs)` and
`task_goal_embeddings = _encode_all_levels(raw goal)`.

NO online env XY is accepted here. Pass `xy_current = readout(E3(real_obs))`
and `xy_goal = readout(E3(real_goal))`. The caller must enforce this invariant.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from topo_graph import ObservedTopology, RouteSuggestion


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool = False
    # Must be provided by a held-out real-controller/localization audit.
    certification_evidence_id: str = ""
    checkpoint_sha256: str = ""
    normalizer_sha256: str = ""
    attach_radius: float = 1.0
    attach_margin: float = 0.0
    lookahead_cells: int = 2
    arrival_radius: float = 0.8
    max_waypoint_calls: int = 6

    def __post_init__(self):
        if (
            self.attach_radius<=0 or self.attach_margin<0
            or self.lookahead_cells<1 or self.arrival_radius<=0
            or self.max_waypoint_calls<1
        ):
            raise ValueError("Invalid calibration fields")


class TopologyGoalAdapter:
    def __init__(self, graph: ObservedTopology, config: AdapterConfig):
        self.graph=graph
        self.config=config
        if config.enabled:
            if not config.certification_evidence_id:
                raise ValueError("Uncertified online graph is blocked")
            graph.verify(config.checkpoint_sha256,config.normalizer_sha256)
        self._env_state:list[dict]=[]
        self.records:list[dict]=[]

    def reset(self, num_envs: int):
        if num_envs<1:
            raise ValueError("Invalid number of environments")
        self._env_state=[{} for _ in range(num_envs)]
        self.records=[]

    def choose(
        self,
        xy_current: np.ndarray,
        xy_task_goal: np.ndarray,
        original_goals: dict[str,torch.Tensor],
        *,
        planning_call: int,
    ) -> tuple[dict[str,torch.Tensor],torch.Tensor]:
        """Return same-shape goals and override mask.

        Fallback returns EXACT original tensor references, zero decoder graph,
        and no seeded RNG effects. Do not call when disabled.
        """
        if not self.config.enabled:
            b=original_goals["embed_3"].shape[0]
            return original_goals,torch.zeros(b,dtype=torch.bool,
                                            device=original_goals["embed_3"].device)
        b=original_goals["embed_3"].shape[0]
        cur=np.asarray(xy_current,dtype=float)
        goal=np.asarray(xy_task_goal,dtype=float)
        if cur.shape!=(b,2) or goal.shape!=(b,2):
            raise ValueError("Only real current/goal decoded [B,2] positions")
        for key in ("embed_1","embed_2","embed_3"):
            if key not in original_goals or original_goals[key].ndim!=3:
                raise ValueError(f"Missing [B,1,D] goal for {key}")
            if original_goals[key].shape[:2]!=(b,1):
                raise ValueError(f"Invalid original goal shape: {key}")
        if len(self._env_state)!=b:
            self.reset(b)
        changed={k:v.detach().clone() for k,v in original_goals.items()}
        masks=torch.zeros(b,dtype=torch.bool,
                          device=original_goals["embed_3"].device)
        for i in range(b):
            mem=self._env_state[i]
            # Never allow a previously assigned task's route to leak into
            # the next task. More robust task reset is caller.reset() at env reset.
            if mem and np.linalg.norm(mem["goal_xy"]-goal[i])>1e-4:
                mem={}
                self._env_state[i]=mem
            if mem.get("disabled",False):
                why="previous_timeout_or_no_route"
                node=None
            else:
                node=mem.get("node")
                completed_node=None
                if node is not None:
                    dist=float(np.linalg.norm(cur[i]-self.graph.center(node)))
                    if dist<=self.config.arrival_radius:
                        completed_node=node
                        node=None
                        mem.pop("node",None)
                        mem["calls"]=0
                        why="waypoint_reached"
                    elif mem["calls"]>=self.config.max_waypoint_calls:
                        mem["disabled"]=True
                        node=None
                        why="waypoint_timeout"
                    else:
                        why="continue_waypoint"
                else:
                    why="new_waypoint"

                if node is None and not mem.get("disabled",False):
                    suggestion=self.graph.suggest(
                        cur[i],goal[i],
                        attachment_radius=self.config.attach_radius,
                        attachment_margin=self.config.attach_margin,
                        lookahead_cells=self.config.lookahead_cells,
                    )
                    if suggestion is None:
                        mem["disabled"]=True
                        why="no_usable_route"
                    elif suggestion.waypoint_node==completed_node:
                        mem["disabled"]=True
                        why="same_waypoint_reselected_fallback"
                    else:
                        node=suggestion.waypoint_node
                        mem["node"]=node
                        mem["calls"]=0
                        mem["route_nodes"]=suggestion.nodes
                        why="new_waypoint"

                mem["goal_xy"]=goal[i].copy()
            if node is not None and not mem.get("disabled",False):
                rep=self.graph.reps[node]
                for key in ("embed_1","embed_2","embed_3"):
                    template=changed[key][i,0]
                    vec=torch.as_tensor(rep[key],dtype=template.dtype,
                                        device=template.device)
                    if vec.shape!=template.shape:
                        raise ValueError(f"{key} waypoint mismatch")
                    changed[key][i,0]=vec
                masks[i]=True
                mem["calls"]+=1
            self.records.append({
                "env":i,
                "planning_call":int(planning_call),
                "active":bool(masks[i]),
                "reason":why,
                "waypoint_node":None if node is None else int(node),
                "wait_calls":int(mem.get("calls",0))
            })
        return changed,masks
