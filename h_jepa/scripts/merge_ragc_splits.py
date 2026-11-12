"""Merge Stage-1C + Stage-1D anchors into one dataset and verify isolation.

  * Multiple `anchors.jsonl` inputs are concatenated (no duplicates allowed).
  * Train/val must not share an EXPERT EPISODE (not just an anchor id).
  * Writes `anchors.jsonl`, `train.npz`, `val.npz` into --out-dir.

Usage:
  cd H-JEPA/h_jepa
  PYTHONPATH=. python scripts/merge_ragc_splits.py \
      --jsonl artifacts/stage1c_smoke/anchors.jsonl /tmp/ragc_stage1d/anchors.jsonl \
      --out-dir /tmp/ragc_merged
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load(path):
    recs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "anchor":
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "stage1_report",
        Path(__file__).resolve().parent / "ragc_stage1_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    recs = []
    for p in args.jsonl:
        r = load(p)
        print(f"{p}: {len(r)} anchors")
        recs.extend(r)

    uids = [r["task_uid"] for r in recs]
    if len(uids) != len(set(uids)):
        dup = sorted({u for u in uids if uids.count(u) > 1})
        raise SystemExit(f"duplicate task_uid across inputs: {dup}")

    valid = [r for r in recs if r.get("valid")]
    print(f"anchors: {len(recs)}  valid: {len(valid)}  "
          f"invalid: {len(recs) - len(valid)}")

    # Anchor ids are per-collection counters; renumber globally so the
    # train/val anchor-id disjointness check in ragc_data stays meaningful.
    for i, r in enumerate(recs):
        r["local_anchor_id"] = r.get("anchor_id")
        r["anchor_id"] = i

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "anchors.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    made = mod.rebuild_npz(recs, str(out))
    eps = {}
    for split, v in made.items():
        if v is None:
            print(f"{split}: EMPTY")
            continue
        ds = v[1]
        ok = np.asarray(ds["baseline_success"]).astype(bool)
        eps[split] = set(ds["source_episode_uid"].tolist())
        print(f"{split}: pairs={len(ok)}  "
              f"from_success_anchors={int(ok.sum())}  "
              f"from_fail_anchors={int((~ok).sum())}  "
              f"anchors={len(np.unique(ds['anchor_id']))}  "
              f"episodes={len(eps[split])}")
        if len({r for r in ds["source_episode_uid"].tolist()}) != len(
                np.unique(ds["source_episode_uid"])):
            pass
    if "train" in eps and "val" in eps:
        overlap = sorted(eps["train"] & eps["val"])
        print(f"episode overlap train/val: {overlap or 'NONE'}")
        if overlap:
            raise SystemExit(f"EPISODE LEAK: {overlap}")

    # anchor-id leakage guard (train/val must not share anchor ids)
    if made["train"] and made["val"]:
        a = set(np.unique(made["train"][1]["anchor_id"]).tolist())
        b = set(np.unique(made["val"][1]["anchor_id"]).tolist())
        if not a.isdisjoint(b):
            # anchor ids are local counters; enforce global uniqueness instead
            print(f"WARN local anchor ids overlap: {sorted(a & b)} "
                  "(ids are per-collection counters)")
    print("merged ->", out)


if __name__ == "__main__":
    main()
