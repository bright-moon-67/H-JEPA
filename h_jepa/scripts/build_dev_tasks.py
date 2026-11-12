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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--offset", type=int, default=60)
    p.add_argument("--start-shift", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        pixels = f["pixels"][()]      # [N,224,224,3] uint8 HWC
        states = f["state"][()]
        proprio = f["proprio"][()]
        actions = f["action"][()]
        step_idx = f["step_idx"][()]
        ep_len = f["ep_len"][()]
        ep_off = f["ep_offset"][()]
        episode_idx = f["episode_idx"][()]

    data, starts = [], []
    for e in range(min(args.episodes, len(ep_len))):
        L = args.offset + 1
        if ep_len[e] < L + 10:
            continue
        off = int(ep_off[e])
        # Shift the start so windows differ from the official (75-offset) ones.
        s = (args.start_shift + e * 3) % (int(ep_len[e]) - L - 1)
        sl = slice(off + s, off + s + L)
        px = np.transpose(pixels[sl], (0, 3, 1, 2))  # CHW
        data.append({
            "pixels": torch.from_numpy(np.ascontiguousarray(px)),
            "state": torch.from_numpy(np.asarray(states[sl], np.float32)),
            "proprio": torch.from_numpy(np.asarray(proprio[sl], np.float32)),
            "action": torch.from_numpy(np.asarray(actions[sl], np.float32)),
            "step_idx": torch.from_numpy(np.asarray(step_idx[sl], np.int64)),
            "episode_idx": torch.from_numpy(np.asarray(episode_idx[sl], np.int64)),
        })
        starts.append(s)

    payload = {
        "data": data,
        "columns": ["action", "episode_idx", "pixels", "proprio", "state", "step_idx"],
        "episodes_idx": list(range(len(data))),
        "start_steps": starts,
        "goal_offset_steps": int(args.offset),
        "eval_budget": 90,
        "process": None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out}: {len(data)} dev tasks, offset={args.offset}, starts={starts}")


if __name__ == "__main__":
    main()
