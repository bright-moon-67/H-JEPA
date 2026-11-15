"""PAHRL-inspired, proof-carrying *directed* episodic state graph.

Observed trajectory transitions are edges; latent neighbors are NOT edges.
Online graph search reads only frozen L3 embeddings and certified graph edges.
Source training-set xy can be kept in a separate diagnostics file, never queried.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-8):
        raise ValueError("Nonfinite or zero-length latent embedding")
    return x / norms


@dataclass(frozen=True)
class Route:
    nodes: tuple[int, ...]
    observed_time_cost: float
    start_attach_distance: float
    goal_attach_distance: float

    @property
    def has_intermediate(self) -> bool:
        return len(self.nodes) >= 3


class GraphMemory:
    """Memory graph on PRE-ENCODED real states; CSR is directed.

    `metadata['attachment_certified']` is false until held-out local
    controller tests demonstrate that current/goal queries may safely attach
    to nearby memorized states. Never auto-certify based on latent distance.
    """

    def __init__(
        self,
        z: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        edge_time: np.ndarray,
        episode: np.ndarray,
        steps: np.ndarray,
        metadata: Mapping[str, Any],
    ):
        self.z = np.asarray(z, dtype=np.float32)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.edge_time = np.asarray(edge_time, dtype=np.float32)
        self.episode = np.asarray(episode, dtype=np.int64)
        self.steps = np.asarray(steps, dtype=np.int64)
        self.metadata = dict(metadata)
        n = self.z.shape[0]
        if self.z.ndim != 2 or n < 2:
            raise ValueError("Need at least two [N,D] graph embeddings")
        if not (self.episode.shape == self.steps.shape == (n,)):
            raise ValueError("Episode/time indices must be [N]")
        if self.indptr.shape != (n + 1,) or self.indptr[0] != 0:
            raise ValueError("Invalid CSR indptr")
        if (
            self.indptr[-1] != len(self.indices)
            or len(self.indices) != len(self.edge_time)
            or np.any(np.diff(self.indptr) < 0)
            or np.any(self.indices < 0)
            or np.any(self.indices >= n)
            or np.any(self.edge_time <= 0)
            or not np.isfinite(self.edge_time).all()
        ):
            raise ValueError("Malformed directed graph edges")
        if self.metadata.get("source_split") != "train":
            raise ValueError("Graph may only be constructed from training episodes")
        self.z_norm = _unit(self.z)

    def verify_checkpoint(self, checkpoint_sha256: str) -> None:
        old = self.metadata.get("checkpoint_sha256", "")
        if not old or old != checkpoint_sha256:
            raise ValueError("Graph and live L3 checkpoint do not match")

    def verify_artifacts(
        self, checkpoint_sha256: str, normalizer_sha256: str
    ) -> None:
        self.verify_checkpoint(checkpoint_sha256)
        old = self.metadata.get("normalizer_sha256")
        if not old or old != normalizer_sha256:
            raise ValueError("Graph and live action/proprio normalizer mismatch")

    @property
    def certified(self) -> bool:
        return bool(self.metadata.get("attachment_certified", False))

    @property
    def attach_max(self) -> float:
        if not self.certified:
            raise ValueError("Online use requires a validated attachment threshold")
        d = self.metadata.get("attachment_max_cosine_distance")
        if d is None or not 0 < float(d) < 2:
            raise ValueError("Invalid attachment certification")
        return float(d)

    def nearest(
        self, query: np.ndarray, *, max_dist: float, k: int = 3
    ) -> list[tuple[int, float]]:
        """Retrieve candidate attachment nodes; does not create graph edges."""
        vec = _unit(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if vec.shape[1] != self.z.shape[1]:
            raise ValueError("Encoder dimension mismatch")
        d = np.maximum(0.0, 1.0 - self.z_norm @ vec[0])
        nearest = np.argsort(d, kind="stable")[:max(1, k)]
        return [
            (int(i), float(d[i])) for i in nearest if d[i] <= max_dist
        ]

    def shortest_path(self, start: int, goal: int) -> tuple[float, tuple[int, ...]]:
        """Dijkstra, only following observed/verified directed edges."""
        if start == goal:
            return 0.0, (int(start),)
        n = len(self.z)
        best = np.full(n, np.inf, dtype=np.float64)
        prev = np.full(n, -1, dtype=np.int64)
        best[start] = 0.0
        heap: list[tuple[float, int]] = [(0.0, start)]
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > best[u]:
                continue
            if u == goal:
                break
            for i in range(self.indptr[u], self.indptr[u + 1]):
                v = int(self.indices[i])
                nxt = cost + float(self.edge_time[i])
                if nxt < best[v]:
                    best[v], prev[v] = nxt, u
                    heapq.heappush(heap, (nxt, v))
        if not np.isfinite(best[goal]):
            return np.inf, ()
        path = [goal]
        while path[-1] != start:
            p = int(prev[path[-1]])
            if p < 0:
                raise AssertionError("Broken shortest-path predecessor")
            path.append(p)
        path.reverse()
        return float(best[goal]), tuple(path)

    def route(
        self,
        current: np.ndarray,
        task_goal: np.ndarray,
        *,
        k: int = 3,
    ) -> Route | None:
        """Plan from current and task GOAL embeddings, with certified attachments.

        Current/goal nearest nodes are PROXY attachments; Stage 1 must certify
        that the local H-JEPA controller can reach near nodes reliably.
        """
        if not self.certified:
            return None
        start_nodes = self.nearest(current, max_dist=self.attach_max, k=k)
        goal_nodes = self.nearest(task_goal, max_dist=self.attach_max, k=k)
        best: Route | None = None
        for a, da in start_nodes:
            for b, db in goal_nodes:
                cost, path = self.shortest_path(a, b)
                # A zero-hop or one-hop result offers no intermediate
                # waypoint; do not let it suppress a useful longer route.
                if len(path) < 3:
                    continue
                # Do not mix unitless latent distance with real elapsed steps.
                candidate = Route(path, cost, da, db)
                if best is None or (
                    candidate.observed_time_cost, da + db, candidate.nodes
                ) < (
                    best.observed_time_cost,
                    best.start_attach_distance + best.goal_attach_distance,
                    best.nodes,
                ):
                    best = candidate
        return best

    def save(self, filename: str | Path) -> None:
        np.savez_compressed(
            filename, z=self.z, indptr=self.indptr, indices=self.indices,
            edge_time=self.edge_time, episode=self.episode, steps=self.steps,
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, filename: str | Path) -> "GraphMemory":
        with np.load(filename, allow_pickle=False) as f:
            return cls(
                f["z"], f["indptr"], f["indices"], f["edge_time"],
                f["episode"], f["steps"], json.loads(str(f["metadata"].item())),
            )


def build_graph(
    embeddings: np.ndarray,
    episode: np.ndarray,
    steps: np.ndarray,
    *,
    checkpoint_sha256: str,
    dataset_name: str,
    normalizer_sha256: str = "",
    max_observed_gap: int = 10,
    validated_links: list[dict] | None = None,
    certification: dict | None = None,
) -> GraphMemory:
    """Build only observed transitions and separately certified local edges.

    `validated_links` records must contain from/to indices and independent
    local rollout evidence (>=3 successes out of >=3 attempts).
    No inverse edges are inferred. No same-latent or same-XY shortcuts.
    """
    z = np.asarray(embeddings, dtype=np.float32)
    ep = np.asarray(episode, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    if len(z) != len(ep) or len(z) != len(steps):
        raise ValueError("Mismatched node array lengths")
    if max_observed_gap < 1:
        raise ValueError("max_observed_gap must be positive")
    if not dataset_name.endswith("_train"):
        raise ValueError("Use train-only data: prohibit validation or official test episodes")
    # Export must be lexicographically sorted. This is intentional: stable IDs
    # are used in the rollout-based validated_links records.
    if any(
        (ep[i] > ep[i+1]) or (
            ep[i] == ep[i+1] and steps[i] >= steps[i+1]
        ) for i in range(len(ep)-1)
    ):
        raise ValueError("Node indices must be sorted by (episode, step)")
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(ep) - 1):
        gap = int(steps[i+1] - steps[i])
        if ep[i] == ep[i+1] and 0 < gap <= max_observed_gap:
            edges[(i, i+1)] = float(gap)

    for record in validated_links or []:
        a, b = int(record["from_node"]), int(record["to_node"])
        succ, trials = int(record["successes"]), int(record["trials"])
        mean_steps = float(record["mean_steps"])
        if not (
            0 <= a < len(ep) and 0 <= b < len(ep) and a != b
            and trials >= 3 and succ >= 3 and succ == trials
            and np.isfinite(mean_steps) and mean_steps > 0
            and record.get("source_split") == "train"
            and bool(record.get("evidence_sha256"))
        ):
            raise ValueError(f"Unproven or malformed extra edge: {record}")
        edges[(a, b)] = min(edges.get((a, b), np.inf), mean_steps)

    buckets: list[list[tuple[int, float]]] = [[] for _ in range(len(ep))]
    for (a, b), cost in sorted(edges.items()):
        buckets[a].append((b, cost))
    indptr = [0]
    indices: list[int] = []
    weights: list[float] = []
    for row in buckets:
        indices.extend(v for v, _ in row)
        weights.extend(w for _, w in row)
        indptr.append(len(indices))

    cert = certification or {}
    certified = bool(cert.get("calibration_passed", False))
    # Stage 1 must provide external held-out local controller tests before on.
    if certified and (
        int(cert.get("num_trials", 0)) < 8
        or int(cert.get("positive_trials", 0)) < 3
        or int(cert.get("negative_trials", 0)) < 3
        or not str(cert.get("evidence_sha256", ""))
        or cert.get("source_split") != "train"
    ):
        raise ValueError(
            "Attachment calibration needs >=8 train-only real rollouts, "
            "including >=3 positives and >=3 negatives, with evidence hash"
        )
    meta = {
        "format": "pahrl_hjepa_verified_graph_v1",
        "source_split": "train",
        "dataset_name": dataset_name,
        "checkpoint_sha256": checkpoint_sha256,
        "normalizer_sha256": normalizer_sha256,
        "max_observed_gap": max_observed_gap,
        "observed_edges": sum(1 for i in range(len(ep)-1)
                              if ep[i] == ep[i+1]
                              and 0 < steps[i+1]-steps[i] <= max_observed_gap),
        "validated_links": len(validated_links or []),
        "attachment_certified": certified,
        "attachment_max_cosine_distance": (
            float(cert["max_cosine_distance"]) if certified else None
        ),
        "attachment_evidence_id": cert.get("evidence_id", ""),
    }
    return GraphMemory(
        z, np.asarray(indptr), np.asarray(indices),
        np.asarray(weights), ep, steps, meta,
    )
