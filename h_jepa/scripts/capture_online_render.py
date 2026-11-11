"""Capture REAL online encoder inputs and compare render distributions (P2.5).

The counterfactual collector renders with `with_target=False`; the official
planning pipeline renders through MegaWrapper (world.py) with the env default
(with_target=True, white background). Offline eval-trajectory pixels come from
the original gray-background dataset. This script samples the THREE
distributions and compares raw RGB statistics and frozen-encoder latents.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/capture_online_render.py \
        --cf-data /tmp/pusht_cf_full/train.h5 \
        --eval-trajs assets/eval_trajs/pusht/goal_offset_75_val.pt \
        --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt"
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.wrapper import MegaWrapper

from action_response_data import load_anchor_h5
from action_response_metrics import compute_response_metrics  # noqa: F401 (kept for parity)
from train_action_response import encode_rgb, load_checkpoint


def make_online_env(seed):
    """Replicates World's env construction: gym env + MegaWrapper(224)."""
    base = gym.make(
        "swm/PushT-v1",
        render_mode="rgb_array",
        resolution=224,
        with_target=True,   # World does not override this env default
        relative=True,
        max_episode_steps=90,
    )
    env = MegaWrapper(base, image_shape=(224, 224), history_size=1, frame_skip=1)
    env.reset(seed=seed)
    return env


def collect_online_frames(seeds, steps_per_seed):
    frames = []
    rng = np.random.default_rng(0)
    for seed in seeds:
        env = make_online_env(seed)
        obs, info = env.reset(seed=seed)
        frames.append(np.asarray(info["pixels"], dtype=np.uint8))
        for _ in range(steps_per_seed - 1):
            action = rng.uniform(-0.75, 0.75, size=(2,)).astype(np.float32)
            obs, reward, term, trunc, info = env.step(action)
            frames.append(np.asarray(info["pixels"], dtype=np.uint8))
        env.close()
    frames = np.stack(frames)
    # StackedWrapper adds a history dimension: [N,1,H,W,3] -> [N,H,W,3]
    return frames.reshape(-1, *frames.shape[-3:])


def rgb_stats(images, name):
    x = images.astype(np.float32)
    means = x.mean(axis=(0, 1, 2))
    stds = x.std(axis=(0, 1, 2))
    print(f"{name:<30s} n={len(images):4d} shape={images.shape[1:]} "
          f"mean(RGB)={np.round(means, 1)} std(RGB)={np.round(stds, 1)}")
    return means, stds


def latent_stats(model, images, device, name):
    z = encode_rgb(model, images, device, 16).numpy()
    print(f"  latent {name:<22s} mean|z|={np.linalg.norm(z.mean(0)):.3f} "
          f"mean||z||={np.linalg.norm(z, axis=1).mean():.3f} "
          f"std={z.std():.4f}")
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cf-data", required=True)
    p.add_argument("--eval-trajs", default="assets/eval_trajs/pusht/goal_offset_75_val.pt")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda" if torch_available() else "cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()

    online = collect_online_frames(args.seeds, args.steps)
    print(f"online frames: {online.shape}\n")

    cf = load_anchor_h5(args.cf_data)
    cf_frames = cf["rgb0"][:500]

    eval_frames = None
    if os.path.exists(args.eval_trajs):
        import torch
        payload = torch.load(args.eval_trajs, map_location="cpu", weights_only=False)
        frames = []
        for ep in payload["data"][:20]:
            px = ep["pixels"]
            px = px.numpy() if hasattr(px, "numpy") else np.asarray(px)
            frames.append(np.transpose(px, (0, 2, 3, 1)))
        eval_frames = np.concatenate(frames, axis=0)[:500]

    print("=== raw RGB statistics ===")
    online_m, _ = rgb_stats(online, "ONLINE (MegaWrapper, World)")
    cf_m, _ = rgb_stats(cf_frames, "counterfactual collector")
    if eval_frames is not None:
        off_m, _ = rgb_stats(eval_frames, "offline eval-traj pixels")
        print(f"\nchannel-mean |online - collector| = {np.abs(online_m - cf_m).mean():.2f}")
        print(f"channel-mean |online - offline|   = {np.abs(online_m - off_m).mean():.2f}")
        print(f"channel-mean |collector - offline| = {np.abs(cf_m - off_m).mean():.2f}")

    # ---- same-state comparison: goal overlay on/off ----
    import json

    def render_same_state(with_target, seed, actions):
        env = gym.make(
            "swm/PushT-v1", render_mode="rgb_array", resolution=224,
            with_target=with_target, relative=True,
        )
        env.reset(seed=seed)
        out = [np.asarray(env.render(), dtype=np.uint8)]
        for a in actions:
            env.step(a)
            out.append(np.asarray(env.render(), dtype=np.uint8))
        env.close()
        return np.stack(out)

    rng = np.random.default_rng(0)
    same_seeds = args.seeds
    same_acts = {s: rng.uniform(-0.75, 0.75, (args.steps - 1, 2)).astype(np.float32)
                 for s in same_seeds}
    same_on = np.concatenate([render_same_state(True, s, same_acts[s]) for s in same_seeds])
    same_off = np.concatenate([render_same_state(False, s, same_acts[s]) for s in same_seeds])
    px_diff = float(np.abs(same_on.astype(np.float64) - same_off.astype(np.float64)).mean())
    print(f"\nsame-state goal overlay: pixel mean|diff|={px_diff:.2f} (0-255 scale)")

    if args.ckpt:
        import torch
        device = torch.device(args.device)
        model = load_checkpoint(args.ckpt, device)
        z_on = encode_rgb(model, same_on, device, 16)
        z_off = encode_rgb(model, same_off, device, 16)
        z_cf = encode_rgb(model, cf_frames[:len(same_on)], device, 16)

        def rel_dist(a, b):
            return float((a - b).norm(dim=1).mean() / b.norm(dim=1).mean())

        print("=== frozen-encoder latent statistics ===")
        print("  latent mean|z|: " + json.dumps({
            "online(goal on)": round(float(z_on.norm(dim=1).mean()), 2),
            "online(goal off)": round(float(z_off.norm(dim=1).mean()), 2),
            "collector": round(float(z_cf.norm(dim=1).mean()), 2),
        }))
        print("  relative latent distance:")
        print(f"    goal-on vs goal-off (same state) = {rel_dist(z_on, z_off):.4f}")
        print(f"    collector vs goal-off (same cfg)  = {rel_dist(z_cf, z_off):.4f}")
        if eval_frames is not None:
            z_ev = encode_rgb(model, eval_frames[:len(same_on)], device, 16)
            print(f"    offline vs goal-off               = {rel_dist(z_ev, z_off):.4f}")
            print(f"    offline mean|z| = {float(z_ev.norm(dim=1).mean()):.2f}")


def torch_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
