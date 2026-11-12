"""Gate a traced replay against the original evaluation artifacts.

The audit rule: adding logging must not change task outcomes.  This script
compares the original per-worker ``metrics.yaml`` / ``planning_compute.json``
with the replay's files and writes a machine-readable ``gate.json``.

Usage::

    python scripts/compare_trace_replay.py \
        --original-dir .../eval_ant_l3_par/w1 \
        --replay-dir   .../eval_ant_l3_trace_w1

Exit code 0 means the gate passed **exactly** (success pattern, success
steps, seeds and per-solve GD iteration counts).  ``--allow-iters-drift``
relaxes only the last check (GPU kernels may be timing-dependent), which
must be recorded explicitly in the audit report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def _nan_equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def compare(original_dir: Path, replay_dir: Path, allow_iters_drift: bool = False) -> dict:
    result = {
        "original_dir": str(original_dir),
        "replay_dir": str(replay_dir),
        "checks": {},
        "passed": True,
    }

    def check(name, ok, detail=None):
        result["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            result["passed"] = False

    orig = _load(original_dir / "metrics.yaml")
    repl = _load(replay_dir / "metrics.yaml")
    check("metrics_present", orig is not None and repl is not None)
    if orig is None or repl is None:
        return result

    o = [bool(x) for x in orig["episode_successes"]]
    r = [bool(x) for x in repl["episode_successes"]]
    check("episode_successes", o == r, {"original": o, "replay": r})

    o_steps = list(orig.get("steps_to_success") or [])
    r_steps = list(repl.get("steps_to_success") or [])
    same_steps = len(o_steps) == len(r_steps) and all(
        _nan_equal(a, b) for a, b in zip(o_steps, r_steps)
    )
    check("steps_to_success", same_steps, {"original": o_steps, "replay": r_steps})

    check(
        "seeds",
        orig.get("seeds") == repl.get("seeds"),
        {"original": orig.get("seeds"), "replay": repl.get("seeds")},
    )

    o_comp = _load(original_dir / "planning_compute.json")
    r_comp = _load(replay_dir / "planning_compute.json")
    if o_comp is not None and r_comp is not None:
        check(
            "planning_compute_n_records",
            len(o_comp) == len(r_comp),
            {"original": len(o_comp), "replay": len(r_comp)},
        )
        same_meta = all(
            (a.get("level"), a.get("horizon"), a.get("num_samples"), len(a.get("n_iters") or []))
            == (b.get("level"), b.get("horizon"), b.get("num_samples"), len(b.get("n_iters") or []))
            for a, b in zip(o_comp, r_comp)
        )
        check("planning_compute_levels_horizons", same_meta)
        same_iters = all(
            (a.get("n_iters") or []) == (b.get("n_iters") or []) for a, b in zip(o_comp, r_comp)
        )
        check(
            "planning_compute_n_iters",
            same_iters,
            None if same_iters else "GD iteration counts differ (timing/GPU drift)",
        )
        if not same_iters and allow_iters_drift:
            result["checks"]["planning_compute_n_iters"]["ok"] = True
            result["checks"]["planning_compute_n_iters"]["waived"] = True
            result["passed"] = all(c["ok"] for c in result["checks"].values())
    else:
        check("planning_compute_present", False)

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-dir", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--allow-iters-drift", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = compare(
        Path(args.original_dir), Path(args.replay_dir), args.allow_iters_drift
    )
    out = Path(args.out) if args.out else Path(args.replay_dir) / "gate.json"
    out.write_text(json.dumps(result, indent=1))
    for name, info in result["checks"].items():
        print(f"[{'OK' if info['ok'] else 'FAIL'}] {name}")
        if not info["ok"] and info.get("detail") is not None:
            print(f"       {info['detail']}")
    print(f"gate {'PASSED' if result['passed'] else 'FAILED'} -> {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
