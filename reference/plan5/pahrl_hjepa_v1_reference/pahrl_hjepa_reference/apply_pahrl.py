#!/usr/bin/env python3
"""Strict source-anchored PAHRL patcher for H-JEPA at research/ant-trace-audit-v1.

Preflight all changes and Python syntax before writing any file.
Refuse to patch an unexpected code version. Run exactly once on a clean branch.
"""
from pathlib import Path
import json
import sys


def main(repo: Path):
    repo = repo.resolve()
    entries = json.loads((Path(__file__).parent / "patch_spec.json").read_text())
    staged = {}
    for entry in entries:
        f = repo / entry["path"]
        if f not in staged:
            staged[f] = f.read_text()
        matches = staged[f].count(entry["old"])
        if matches != 1:
            raise RuntimeError(
                f"{entry['path']}: expected 1 patch anchor, found {matches}. "
                "Branch or commit mismatch; no files changed."
            )
        staged[f] = staged[f].replace(entry["old"], entry["new"], 1)
    for f, src in staged.items():
        compile(src, str(f), "exec")
    for f, src in staged.items():
        f.write_text(src)
        print("PATCHED", f.relative_to(repo))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python apply_pahrl.py /path/to/H-JEPA")
    main(Path(sys.argv[1]))
