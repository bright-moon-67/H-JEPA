"""Probe-based physical effect diagnostics (P2.5).

Question: is the latent response (WM or ARN) actually about the BLOCK, or only
about the agent's own displacement?

Linear probes decode agent position / block position / block angle
(sin,cos) from frozen Level-1 latents. Because probes are linear, the bias
cancels for latent DIFFERENCES, so

    predicted delta_effect = W_probe @ (predicted latent response)

can be compared against the true physical deltas of the two branches.

State layout: [agent_xy(2), block_xy(2), block_angle(1), agent_vel(2)].
Physical targets are used ONLY for this diagnosis, never as model input.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/physical_effect_probes.py \
        --train /tmp/pusht_cf_full/train.h5 \
        --test /tmp/pusht_cf_full/test.h5 \
        --head /tmp/action_response_head_full3k.pt \
        --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt"
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from action_response import ActionResponseNet, ResponseConfig
from action_response_data import build_pair_tensors, encode_anchors, load_anchor_h5
from action_response_wm import get_action_stats, wm_response
from train_action_response import load_checkpoint


def fit_probe(z, y):
    """Least-squares linear probe with bias. z [N,D], y [N,T] -> (W [T,D], b [T])."""
    X = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[:-1].T, coef[-1]


def probe_r2(z, y, W, b):
    pred = z @ W.T + b
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def state_targets(state):
    """[N,7] -> dict of physical targets used by probes."""
    ang = state[:, 4]
    return {
        "agent_xy": state[:, 0:2],
        "block_xy": state[:, 2:4],
        "block_sincos": np.stack([np.sin(ang), np.cos(ang)], axis=1),
    }


def angle_error_deg(pred_sincos, true_sincos):
    """Rotation error from differences of unit-circle vectors (degrees)."""
    d = np.asarray(pred_sincos) - np.asarray(true_sincos)
    norm = np.clip(np.linalg.norm(d, axis=-1) / 2.0, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(norm))


@torch.no_grad()
def batched_wm(model, z0, a, b, mean, std, device, batch_size=64):
    outs = []
    for k in range(0, len(z0), batch_size):
        outs.append(wm_response(
            model, z0[k:k + batch_size].to(device),
            a[k:k + batch_size].to(device), b[k:k + batch_size].to(device),
            mean, std,
        ).float().cpu())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device)
    mean, std = get_action_stats(model, args.ckpt, device)

    # ---------- fit probes on train anchors ----------
    tr = load_anchor_h5(args.train)
    tr_z0, tr_zend = encode_anchors(model, tr, device, 16)
    tr_targets = state_targets(tr["end_state"].reshape(-1, 7))

    # ---------- test anchors ----------
    te = load_anchor_h5(args.test)
    te_z0, te_zend = encode_anchors(model, te, device, 16)
    te_targets = state_targets(te["end_state"].reshape(-1, 7))

    probes = {}
    print("probe quality on TEST anchors (linear probe fitted on TRAIN only):")
    for name in tr_targets:
        W, b = fit_probe(tr_zend.reshape(-1, tr_zend.shape[-1]).numpy(),
                         tr_targets[name])
        r2 = probe_r2(te_zend.reshape(-1, te_zend.shape[-1]).numpy(),
                      te_targets[name], W, b)
        probes[name] = (W, b)
        print(f"  {name:<12s} R2 per dim = {np.round(r2, 3)}")

    # ---------- pair tensors ----------
    pairs = build_pair_tensors(te_z0, te_zend, te["actions"])
    z0p, a, b, target = pairs["z0"], pairs["a"], pairs["b"], pairs["target"]
    ai, bi, bj = (pairs["anchor_idx"].numpy(), pairs["branch_i"].numpy(),
                  pairs["branch_j"].numpy())

    s_a = te["end_state"][ai, bi]
    s_b = te["end_state"][ai, bj]
    contacts = (te["contacts"][ai, bi] + te["contacts"][ai, bj]) > 0

    true_delta = {
        "agent_xy": s_a[:, 0:2] - s_b[:, 0:2],
        "block_xy": s_a[:, 2:4] - s_b[:, 2:4],
        "block_sincos": np.stack([
            np.sin(s_a[:, 4]) - np.sin(s_b[:, 4]),
            np.cos(s_a[:, 4]) - np.cos(s_b[:, 4]),
        ], axis=1),
    }

    wm = batched_wm(model, z0p, a, b, mean, std, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    net = ActionResponseNet(ResponseConfig(**payload["config"])).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    with torch.no_grad():
        arn = net(z0p.to(device), a.to(device), b.to(device)).float().cpu()

    methods = {
        "zero": np.zeros_like(wm.numpy()),
        "frozen-WM": wm.numpy(),
        "ARN": arn.numpy(),
    }

    def report(mask, tag):
        print(f"\n--- physical effect errors [{tag}] (n={int(mask.sum())} pairs) ---")
        for mname, resp in methods.items():
            pred_delta = {
                key: resp[mask] @ W.T + b - b  # bias cancels for differences
                for key, (W, b) in probes.items()
            }
            # latent-space response magnitude (context)
            resp_norm = np.linalg.norm(resp[mask], axis=1).mean()

            agent_mse = ((pred_delta["agent_xy"] - true_delta["agent_xy"][mask]) ** 2).sum(1).mean()
            block_mse = ((pred_delta["block_xy"] - true_delta["block_xy"][mask]) ** 2).sum(1).mean()
            ang_err = angle_error_deg(
                pred_delta["block_sincos"], true_delta["block_sincos"][mask]
            ).mean()
            true_block = np.linalg.norm(true_delta["block_xy"][mask], axis=1).mean()
            true_agent = np.linalg.norm(true_delta["agent_xy"][mask], axis=1).mean()
            print(f"  {mname:<10s} |r|={resp_norm:5.2f}  "
                  f"agent MSE={agent_mse:8.2f}px^2  block MSE={block_mse:8.2f}px^2  "
                  f"angle MAE={ang_err:6.2f}deg   "
                  f"(true |d_agent|={true_agent:.1f}px, |d_block|={true_block:.1f}px)")

    report(np.ones(len(target), dtype=bool), "ALL")
    report(contacts, "contact")
    report(~contacts, "no-contact")


if __name__ == "__main__":
    main()
