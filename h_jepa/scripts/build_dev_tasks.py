"""Build DEVELOPMENT tasks for the RAGC oracle gate from the val expert split.

These tasks must NOT overlap the official 50-task evaluation windows:
we use a different goal offset (60 vs 75) AND shifted start steps.

Output payload matches the eval-trajectory format consumed by eval/pilot:
  pixels stored CHW uint8 (world permutes to HWC), goal = last frame.

Usage (from h_jepa/):
    PYTHONPATH=. python scripts/build_dev_tasks.py \
        --h5 "$HJEPA_HOME/pusht_expert_val.h5" --episodes 12 --offset 60 \
        --out /tmp/pusht_dev/dev_tasks.pt
"""
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers compression filters used by the H5 file
import numpy as np
import torch


def parse_ids(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--episode-ids", default=None,
                   help="explicit expert episode ids, e.g. 400-415 (skips --episodes)")
    p.add_argument("--offset", type=int, default=60)
    p.add_argument("--start-shift", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        ep_ids = (parse_ids(args.episode_ids) if args.episode_ids
                  else list(range(min(args.episodes, len(f["ep_len"])))))
        data, starts, kept = [], [], []
        for e in ep_ids:
            L = args.offset + 1
            ep_len_e = int(f["ep_len"][e])
            if ep_len_e < L + 10:
                continue
            off = int(f["ep_offset"][e])
            s = (args.start_shift + e * 3) % (ep_len_e - L - 1)
            sl = slice(off + s, off + s + L)
            px = np.transpose(f["pixels"][sl], (0, 3, 1, 2))  # HWC -> CHW
            data.append({
                "pixels": torch.from_numpy(np.ascontiguousarray(px)),
                "state": torch.from_numpy(
                    np.asarray(f["state"][sl], np.float32)),
                "proprio": torch.from_numpy(
                    np.asarray(f["proprio"][sl], np.float32)),
                "action": torch.from_numpy(
                    np.asarray(f["action"][sl], np.float32)),
                "step_idx": torch.from_numpy(
                    np.asarray(f["step_idx"][sl], np.int64)),
                "episode_idx": torch.from_numpy(
                    np.asarray(f["episode_idx"][sl], np.int64)),
                # Deterministic env reset: without this, `reset(seed=None)`
                # samples a fresh variation (block/agent geometry) every run and
                # two processes evaluating the same task diverge after contact.
                "seed": np.int64(1000 + len(data)),
            })
            starts.append(int(s))
            kept.append(int(e))

    payload = {
        "data": data,
        "columns": ["action", "episode_idx", "pixels", "proprio", "state",
                    "step_idx", "seed"],
        "episodes_idx": kept,
        "start_steps": starts,
        "goal_offset_steps": int(args.offset),
        "eval_budget": 90,
        "process": None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out}: {len(data)} tasks, offset={args.offset}, "
          f"episodes={kept}, starts={starts}")


if __name__ == "__main__":
    main()
