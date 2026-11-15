#!/usr/bin/env python3
"""Strict source-anchored, fail-closed HiDe V1 patcher.

Patches ONLY H-JEPA/h_jepa/hierarchical_solver.py and planning_eval.py.
All exact-string anchors must occur once. Preflights syntax before writing.
Run on a clean branch based on research/ant-trace-audit-v1, never on PAHRL.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

def main(repo: Path) -> None:
    repo=repo.resolve()
    edits=json.loads((Path(__file__).parent/"patch_spec.json").read_text())
    stage={}
    for edit in edits:
        target=repo/edit["file"]
        if target not in stage:
            stage[target]=target.read_text(encoding="utf-8")
        n=stage[target].count(edit["find"])
        if n!=1:
            raise RuntimeError(
                f"Refusing to patch {edit['file']}: matched {n} exact source "
                f"anchors, expected 1; no files written. Check your base commit."
            )
        stage[target]=stage[target].replace(edit["find"],edit["replace"],1)
    for f,content in stage.items():
        compile(content,str(f),"exec")
    for f,content in stage.items():
        f.write_text(content,encoding="utf-8")
        print("PATCHED",f.relative_to(repo))

if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Usage: python apply_hide.py /path/to/H-JEPA")
    main(Path(sys.argv[1]))
