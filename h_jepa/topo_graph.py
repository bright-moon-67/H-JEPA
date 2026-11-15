"""HiDe-style TASK-SPACE route planner; H-JEPA controller remains frozen.

Offline supervised map evidence: dense TRAIN XY observations only.
Online localization: decoded XY from real pixels/proprio -> frozen E3 -> P(xy).
Online action selection never reads privileged env xy, maze BFS, or future rollout.

This is a reference core, NOT integrated with a specific H-JEPA branch.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
from typing import Any
import numpy as np


def _array(x, shape_tail=None, dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if shape_tail is not None and (a.ndim != 2 or a.shape[1] != shape_tail):
        raise ValueError(f"Expected [N,{shape_tail}], got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite inputs")
    return a


@dataclass(frozen=True)
class RouteSuggestion:
    nodes: tuple[int, ...]
    waypoint_node: int
    waypoint_xy: tuple[float, float]
    cost: float


class ObservedTopology:
    """Grid memory with only *actually observed* adjacent transitions.

    Grid-cell merging of distinct episodes is a modeling assumption and must
    be independently audited for thin-wall/cell-straddling false connections.
    """
    def __init__(
        self, cells: np.ndarray, edges: np.ndarray, support: np.ndarray,
        *, resolution: float, origin=(0.0, 0.0), source_split="train",
        evidence_id="", checkpoint_sha256="", normalizer_sha256="",
    ):
        self.cells = np.asarray(cells,dtype=np.int64)
        self.edges = np.asarray(edges,dtype=np.int64).reshape(-1,2)
        self.support = np.asarray(support,dtype=np.int64)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin,dtype=np.float64)
        self.source_split = source_split
        self.evidence_id = evidence_id
        self.checkpoint_sha256=checkpoint_sha256
        self.normalizer_sha256=normalizer_sha256
        if (
            self.cells.ndim != 2 or self.cells.shape[1] != 2
            or self.resolution <= 0 or not np.isfinite(self.resolution)
            or self.origin.shape != (2,)
            or self.source_split != "train"
            or self.edges.ndim != 2 or self.edges.shape[1] != 2
            or len(self.support) != len(self.edges)
            or np.any(self.support <= 0)
        ):
            raise ValueError("Invalid TRAIN-only graph")
        if len(set(map(tuple,self.cells))) != len(self.cells):
            raise ValueError("Duplicate grid cells")
        if (len(self.edges) and (
            np.any(self.edges < 0) or np.any(self.edges >= len(self.cells))
        )):
            raise ValueError("Out-of-range graph edges")
        for a,b in self.edges:
            if np.sum(np.abs(self.cells[b]-self.cells[a])) != 1:
                raise ValueError("Forbidden unobserved/nonlocal edge")
        self.cell_to_id={tuple(map(int,x)):i for i,x in enumerate(self.cells)}
        self.adj=[[] for _ in range(len(self.cells))]
        for (a,b),count in zip(self.edges,self.support):
            # one grid-neighbor traversal cost = 1. Edges remain DIRECTED.
            self.adj[int(a)].append((int(b),1.0,int(count)))
        for row in self.adj:
            row.sort()
        self.reps: dict[int,dict[str,np.ndarray]] = {}

    def verify(self, checkpoint_sha256: str, normalizer_sha256: str):
        if (
            not self.checkpoint_sha256 or not self.normalizer_sha256 or
            self.checkpoint_sha256 != checkpoint_sha256 or
            self.normalizer_sha256 != normalizer_sha256
        ):
            raise ValueError("Checkpoint/normalizer mismatch")

    def center(self, idx: int) -> np.ndarray:
        return self.origin + (self.cells[idx].astype(np.float64)+0.5)*self.resolution

    def locate(
        self, xy: np.ndarray, *, radius: float, margin: float=0.0
    ) -> int | None:
        """Conservative query attachment; query XY must be DECODED online."""
        p=_array(np.asarray(xy).reshape(1,2),2)[0]
        if radius <= 0 or margin < 0:
            raise ValueError("Invalid calibrated attachment thresholds")
        if not len(self.cells):
            return None
        centers=self.origin[None,:]+(
            self.cells.astype(np.float64)+0.5
        )*self.resolution
        d=np.linalg.norm(centers-p[None,:],axis=1)
        order=np.argsort(d,kind="stable")
        if d[order[0]]>radius:
            return None
        if len(order)>1 and d[order[1]]-d[order[0]] < margin:
            return None
        return int(order[0])

    def shortest(self, source: int, target: int) -> tuple[tuple[int,...],float]:
        n=len(self.cells)
        if not 0 <= source < n or not 0 <= target < n:
            raise ValueError("Invalid node")
        dist=np.full(n,np.inf)
        prev=np.full(n,-1,dtype=np.int64)
        dist[source]=0.
        q=[(0.,source)]
        while q:
            d,u=heapq.heappop(q)
            if d>dist[u]: continue
            if u==target: break
            for v,w,_ in self.adj[u]:
                nd=d+w
                if nd<dist[v]:
                    dist[v]=nd
                    prev[v]=u
                    heapq.heappush(q,(nd,v))
        if not np.isfinite(dist[target]):
            return (),float("inf")
        path=[target]
        while path[-1]!=source:
            path.append(int(prev[path[-1]]))
        return tuple(reversed(path)),float(dist[target])

    def attach_live_representatives(
        self, xy_train_live: np.ndarray, embeddings: dict[str,np.ndarray]
    ) -> dict[str,int]:
        """TRAIN-only rendered states. Goal encoding must match state encoding.

        `embeddings` keys: embed_1, embed_2, embed_3; each [N,D].
        Representatives are chosen nearest to each cell center among live
        training observations lying IN that cell, avoiding cross-cell teleport.
        """
        xy=_array(xy_train_live,2)
        dims={}
        for k in ("embed_1","embed_2","embed_3"):
            if k not in embeddings:
                raise ValueError(f"Missing {k}")
            z=_array(embeddings[k])
            if z.ndim!=2 or len(z)!=len(xy):
                raise ValueError(f"Invalid {k} embeddings")
            dims[k]=z.shape[1]
        self.reps.clear()
        for r,point in enumerate(xy):
            cell=tuple(np.floor((point-self.origin)/self.resolution).astype(int))
            node=self.cell_to_id.get(cell)
            if node is None: continue
            dist=float(np.linalg.norm(point-self.center(node)))
            old=self.reps.get(node)
            if old is None or dist<old["_distance"]:
                self.reps[node]={
                    "_distance":dist,
                    "_train_index":r,
                    **{k:np.asarray(embeddings[k][r]).copy()
                       for k in ("embed_1","embed_2","embed_3")},
                }
        return dims

    def suggest(
        self, xy_current: np.ndarray, xy_goal: np.ndarray,
        *, attachment_radius: float, attachment_margin: float,
        lookahead_cells: int,
    ) -> RouteSuggestion | None:
        if not self.reps:
            return None
        if lookahead_cells<1:
            raise ValueError("Invalid lookahead_cells")
        s=self.locate(xy_current,radius=attachment_radius,
                      margin=attachment_margin)
        g=self.locate(xy_goal,radius=attachment_radius,
                      margin=attachment_margin)
        if s is None or g is None or s==g:
            return None
        path,cost=self.shortest(s,g)
        if len(path)<3:  # nearby target -> let original H-JEPA finish
            return None
        # Prefer a representative at declared LOCAL lookahead, never past goal.
        max_index=min(lookahead_cells,len(path)-2)
        for i in range(max_index,0,-1):
            node=path[i]
            if node in self.reps:
                center=self.center(node)
                return RouteSuggestion(
                    path,int(node),(float(center[0]),float(center[1])),cost
                )
        return None

    def diagnostic(self) -> dict[str,Any]:
        return {
            "num_cells":len(self.cells),
            "directed_observed_edges":len(self.edges),
            "cells_with_live_waypoint":len(self.reps),
            "train_only":self.source_split=="train",
            "warning":"Cell merging may create false shortcuts; audit walls."
        }


def build_observed_topology(
    xy: np.ndarray,
    episode: np.ndarray,
    step: np.ndarray,
    *,
    resolution: float,
    min_edge_support: int=1,
    origin=(0.,0.),
    max_displacement_per_step: float | None=None,
    source_split="train",
    evidence_id="",
    checkpoint_sha256="",
    normalizer_sha256="",
) -> ObservedTopology:
    """Only raw CONSECUTIVE observations (step diff=1) support cell edges.

    Reject diagonal and nonlocal transitions instead of interpolating.
    Never create kNN edges or infer reversed edges.
    """
    xy=_array(xy,2)
    ep=np.asarray(episode,dtype=np.int64)
    step=np.asarray(step,dtype=np.int64)
    if source_split!="train":
        raise ValueError("TRAIN only; no dev/official task data in map")
    if len(ep)!=len(xy) or len(step)!=len(xy):
        raise ValueError("Mismatched episode and time axes")
    if resolution<=0 or min_edge_support<1:
        raise ValueError("Invalid grid resolution/support")
    if max_displacement_per_step is None:
        max_displacement_per_step=float(resolution)*1.1
    if max_displacement_per_step<=0:
        raise ValueError("Invalid max displacement")
    if any(
        ep[i]>ep[i+1] or (ep[i]==ep[i+1] and step[i]>=step[i+1])
        for i in range(len(ep)-1)
    ):
        raise ValueError("Sort by (episode, step), no repeated timestamps")

    cell=np.floor((xy-np.asarray(origin)[None,:])/resolution).astype(np.int64)
    all_cells=sorted(set(map(tuple,cell)))
    ids={tuple(c):i for i,c in enumerate(all_cells)}
    counts:dict[tuple[int,int],int]={}
    discarded_jump=0
    for i in range(len(xy)-1):
        if ep[i]!=ep[i+1] or step[i+1]-step[i]!=1:
            continue
        if (
            np.linalg.norm(xy[i+1]-xy[i])>max_displacement_per_step or
            np.sum(np.abs(cell[i+1]-cell[i])) > 1
        ):
            discarded_jump+=1
            continue
        if np.array_equal(cell[i],cell[i+1]):
            continue
        a=ids[tuple(cell[i])]
        b=ids[tuple(cell[i+1])]
        counts[(a,b)]=counts.get((a,b),0)+1

    pairs=sorted((k,v) for k,v in counts.items()
                 if v>=min_edge_support)
    graph=ObservedTopology(
        np.asarray(all_cells,dtype=np.int64).reshape(-1,2),
        np.asarray([k for k,_ in pairs],dtype=np.int64).reshape(-1,2),
        np.asarray([v for _,v in pairs],dtype=np.int64),
        resolution=resolution,origin=origin,source_split=source_split,
        evidence_id=evidence_id,checkpoint_sha256=checkpoint_sha256,
        normalizer_sha256=normalizer_sha256
    )
    graph.discarded_nonlocal_transitions=discarded_jump
    return graph


def save_topology(graph: ObservedTopology, path: str):
    """No pickle. Checkpoint hashes and train provenance travel with artifact."""
    import json
    from pathlib import Path
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    rep_ids=np.array(sorted(graph.reps),dtype=np.int64)
    data={
        "cells":graph.cells,
        "edges":graph.edges,
        "support":graph.support,
        "rep_ids":rep_ids,
        "meta":np.asarray(json.dumps({
            "format":"hide_topology_v11",
            "resolution":graph.resolution,
            "origin":graph.origin.tolist(),
            "source_split":graph.source_split,
            "evidence_id":graph.evidence_id,
            "checkpoint_sha256":graph.checkpoint_sha256,
            "normalizer_sha256":graph.normalizer_sha256,
        },sort_keys=True))
    }
    for key in ("embed_1","embed_2","embed_3"):
        if len(rep_ids):
            data[key]=np.stack(
                [graph.reps[int(i)][key] for i in rep_ids]
            ).astype(np.float32)
        else:
            data[key]=np.empty((0,0),dtype=np.float32)
    np.savez_compressed(dest,**data)


def load_topology(path: str) -> ObservedTopology:
    """No pickle. Re-verify SHA before enabling real online controller."""
    import json
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f["meta"].item()))
        if meta.get("format")!="hide_topology_v11":
            raise ValueError("Unknown map version")
        graph=ObservedTopology(
            f["cells"],f["edges"],f["support"],
            resolution=meta["resolution"],origin=meta["origin"],
            source_split=meta["source_split"],
            evidence_id=meta["evidence_id"],
            checkpoint_sha256=meta["checkpoint_sha256"],
            normalizer_sha256=meta["normalizer_sha256"]
        )
        rep_ids=f["rep_ids"]
        for pos,idx in enumerate(rep_ids):
            node=int(idx)
            if not 0<=node<len(graph.cells):
                raise ValueError("Out of range waypoint representative")
            graph.reps[node]={
                key:f[key][pos].copy() for key in (
                    "embed_1","embed_2","embed_3"
                )
            }
    return graph
