#!/usr/bin/env python3
"""Build an auditable, strictly directed PAHRL-inspired state graph.

No kNN 'latent shortcuts' are created as graph edges. Across-episode edges
MUST carry independent local-controller success evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pahrl_memory import build_graph, GraphMemory


def jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def diagnostics(graph: GraphMemory) -> dict:
    n = len(graph.z)
    # Undirected components for quick analysis, but original graph
    # remains DIRECTED for search (reverse reachability is not inferred).
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for u in range(n):
        for index in range(graph.indptr[u], graph.indptr[u + 1]):
            v = int(graph.indices[index])
            parent[find(u)] = find(v)
    comps = {}
    for i in range(n):
        r = find(i)
        comps[r] = comps.get(r, 0) + 1
    return {
        "nodes": n,
        "directed_edges": len(graph.indices),
        "episodes": len(np.unique(graph.episode)),
        "weak_components": len(comps),
        "largest_weak_component": max(comps.values()),
        "certified_online": graph.certified,
        "attachments_enabled": bool(graph.metadata["attachment_certified"]),
        "validation_links": int(graph.metadata.get("validated_links", 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-observed-gap", type=int, default=10)
    p.add_argument("--validated-links-jsonl", type=Path, default=None)
    p.add_argument("--attachment-certificate", type=Path, default=None)
    args = p.parse_args()

    with np.load(args.states, allow_pickle=False) as d:
        z, ep, steps = d["z"], d["episode"], d["steps"]
        meta = json.loads(str(d["metadata"].item()))
    if meta.get("source_split") != "train":
        raise ValueError("Train-only graph invariant violated")
    cert = (
        json.loads(args.attachment_certificate.read_text())
        if args.attachment_certificate else None
    )
    links = jsonl(args.validated_links_jsonl)
    graph = build_graph(
        z, ep, steps,
        checkpoint_sha256=meta["checkpoint_sha256"],
        normalizer_sha256=meta["normalizer_sha256"],
        dataset_name=meta["dataset_name"],
        max_observed_gap=args.max_observed_gap,
        validated_links=links,
        certification=cert,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph.save(args.output)
    status = diagnostics(graph)
    status["source_states"] = str(args.states)
    status["graph"] = str(args.output)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2)
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
