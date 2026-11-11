"""Render-distribution sanity check (plan2 §3.4).

The counterfactual collector renders with `with_target=False`, while the
official evaluation pipeline may render the goal overlay. Before scaling data
collection, compare:
  1) collector images vs official eval-trajectory images (shape/dtype/RGB stats)
  2) env render with_target=True vs with_target=False (what the flag changes)

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/check_render_consistency.py \
        --cf-data /tmp/pusht_cf/train.h5 \
        --eval-trajs assets/eval_trajs/pusht/goal_offset_75_val.pt
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from action_response_data import load_anchor_h5


def rgb_stats(images_hwc, name):
    """images_hwc: uint8 [N,H,W,3] -> prints channel stats."""
    x = images_hwc.astype(np.float32)
    means = x.mean(axis=(0, 1, 2))
    stds = x.std(axis=(0, 1, 2))
    print(f"{name:<28s} n={len(images_hwc):5d} shape={images_hwc.shape[1:]} "
          f"dtype={images_hwc.dtype} mean(RGB)={np.round(means, 2)} "
          f"std(RGB)={np.round(stds, 2)}")
    return means, stds


def load_eval_images(path, max_episodes=20):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    frames = []
    for ep in payload["data"][:max_episodes]:
        px = ep["pixels"]
        if isinstance(px, torch.Tensor):
            px = px.numpy()
        frames.append(np.transpose(px, (0, 2, 3, 1)))  # NCHW -> NHWC
    return np.concatenate(frames, axis=0)


def render_pair(seed):
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    imgs = {}
    for flag in (True, False):
        env = gym.make(
            "swm/PushT-v1", render_mode="rgb_array", resolution=224,
            with_target=flag, relative=True,
        )
        env.reset(seed=int(seed))
        imgs[flag] = np.asarray(env.render(), dtype=np.uint8)
        env.close()
    return imgs[True], imgs[False]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cf-data", required=True)
    p.add_argument("--eval-trajs", default="assets/eval_trajs/pusht/goal_offset_75_val.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.cf_data.endswith(".h5"):
        data = load_anchor_h5(args.cf_data)
        cf_rgb0 = data["rgb0"]
        cf_end = data["end_rgb"].reshape(-1, *data["end_rgb"].shape[2:])
    else:
        d = np.load(args.cf_data, allow_pickle=False)
        cf_rgb0, cf_end = d["rgb0"], d["rgb_a"]

    cf_means, _ = rgb_stats(cf_rgb0[:500], "collector anchors (rgb0)")
    rgb_stats(cf_end[:500], "collector branch ends")

    if os.path.exists(args.eval_trajs):
        eval_imgs = load_eval_images(args.eval_trajs)
        eval_means, _ = rgb_stats(eval_imgs[:500], "official eval trajs")

        d_mean = float(np.abs(cf_means - eval_means).mean())
        print(f"\nchannel-mean |collector - eval| = {d_mean:.2f} (0-255 scale)")
        if d_mean > 5.0:
            print("WARNING: collector images differ noticeably from eval images; "
                  "check with_target / render pipeline before scaling collection.")
        else:
            print("OK: collector and eval image statistics are close.")
    else:
        print(f"warning: eval traj file not found: {args.eval_trajs}")

    with_target, without_target = render_pair(args.seed)
    diff = np.abs(with_target.astype(np.int16) - without_target.astype(np.int16))
    frac = float((diff.max(axis=-1) > 0).mean())
    print(f"\nwith_target=True vs False: mean|diff|={diff.mean():.2f}, "
          f"pixels changed={frac:.2%}")
    print("(goal overlay only appears with with_target=True and "
          "variation_space['rendering']['render_goal'] enabled)")


if __name__ == "__main__":
    main()
