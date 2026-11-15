#!/usr/bin/env python3
"""Same strict, fail-closed applier as apply_hide.py, but with a spec path.

Why this file exists: the reference `patch_spec.json` has three anchors that
carry one extra blank line in the real `research/ant-trace-audit-v1` source
(hierarchical_solver.py x2, planning_eval.py x1). Per the reference README the
fix is to adapt ONLY the smallest semantically equivalent anchors - never to
widen them. `patch_spec_adapted.json` differs from the reference spec solely in
those three blank lines; all inserted code is byte-identical.

Usage:
  python apply_hide_adapted.py /path/to/H-JEPA [--spec patch_spec_adapted.json]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(repo: Path, spec: Path) -> None:
    repo = repo.resolve()
    edits = json.loads(spec.read_text())
    stage = {}
    for edit in edits:
        target = repo / edit["file"]
        if target not in stage:
            stage[target] = target.read_text(encoding="utf-8")
        n = stage[target].count(edit["find"])
        if n != 1:
            raise RuntimeError(
                f"Refusing to patch {edit['file']}: matched {n} exact source "
                f"anchors, expected 1; no files written. Check your base commit."
            )
        stage[target] = stage[target].replace(edit["find"], edit["replace"], 1)
    for f, content in stage.items():
        compile(content, str(f), "exec")
    for f, content in stage.items():
        f.write_text(content, encoding="utf-8")
        print("PATCHED", f.relative_to(repo))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python apply_hide_adapted.py /path/to/H-JEPA [--spec FILE]"
        )
    spec_arg = HERE / "patch_spec_adapted.json"
    if "--spec" in sys.argv:
        spec_arg = Path(sys.argv[sys.argv.index("--spec") + 1])
    main(Path(sys.argv[1]), spec_arg)
