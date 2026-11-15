#!/usr/bin/env python3
"""Install CLFB on the audited source layout (strict, atomic preflight)."""
import json
import sys
from pathlib import Path

def main(repo: Path) -> None:
    repo = repo.resolve()
    spec = json.loads((Path(__file__).parent / "patch_spec.json").read_text())
    staged = {}
    for item in spec:
        path = repo / item["target"]
        if path not in staged:
            staged[path] = path.read_text()
        count = staged[path].count(item["old"])
        if count != 1:
            raise RuntimeError(
                f"{item['target']}: expected one exact source anchor; "
                f"found {count}. Verify branch / commit. NOTHING WRITTEN."
            )
        staged[path] = staged[path].replace(item["old"], item["new"], 1)

    for path, text in staged.items():
        compile(text, str(path), "exec")  # no partial file writes on syntax errors
    for path, text in staged.items():
        path.write_text(text)
        print("PATCHED", path.relative_to(repo))
    print("CLFB integrated. Run tests before enabling it on the GPU.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python apply_clfb.py /path/to/H-JEPA")
    main(Path(sys.argv[1]))
