#!/usr/bin/env python3
"""Stage-0 OFFLINE reachability diagnostic without trusting online attachments.

Input `queries.npz` holds start_z[N,D], goal_z[N,D] exported from original
dev-task t=0 by the EXACT live encoder. This diagnostic only tests potential
graph connectivity at one declared candidate attachment cutoff; it does NOT
certify that H-JEPA can reach the attached nodes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from pahrl_memory import GraphMemory


def possible_path(
    graph: GraphMemory, start: np.ndarray, goal: np.ndarray,
    max_distance: float, k: int = 3
) -> tuple[int, float] | None:
    a=graph.nearest(start,max_dist=max_distance,k=k)
    b=graph.nearest(goal,max_dist=max_distance,k=k)
    candidates=[]
    for start_i,_ in a:
        for goal_i,_ in b:
            d,path=graph.shortest_path(start_i,goal_i)
            if len(path)>=3:
                candidates.append((len(path),d))
    return min(candidates,key=lambda pair:(pair[1],pair[0])) if candidates else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph",type=Path,required=True)
    p.add_argument("--queries",type=Path,required=True)
    p.add_argument("--diagnostic-cosine-cutoff",type=float,required=True)
    p.add_argument("--k",type=int,default=3)
    args=p.parse_args()
    graph=GraphMemory.load(args.graph)
    with np.load(args.queries,allow_pickle=False) as d:
        cur=np.asarray(d["start_z"],dtype=np.float32)
        goal=np.asarray(d["goal_z"],dtype=np.float32)
    if cur.shape!=goal.shape or cur.ndim!=2:
        raise ValueError("Expected matching [N,D] start/goal embeddings")
    paths=[
        possible_path(graph,a,b,args.diagnostic_cosine_cutoff,k=args.k)
        for a,b in zip(cur,goal)
    ]
    report={
        "queries":len(paths),
        "possible_route_count":sum(p is not None for p in paths),
        "possible_route_rate":sum(p is not None for p in paths)/len(paths),
        "path_lengths":[None if p is None else p[0] for p in paths],
        "warning":"NOT ONLINE REACHABILITY; requires real local-controller validation",
    }
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
