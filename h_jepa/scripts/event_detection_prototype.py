"""P3 prototype: action-response signature change as a dynamics-event signal.

Compares, on continuous Push-T trajectories:
  - b_t^WM      : change of FROZEN Level-1 WM response signature
  - b_t^ARN     : change of trained ActionResponseNet signature
  - b_t^latent  : plain latent change (control)
  - random      : chance baseline

Event labels come ONLY from real contact transitions (start/end), never from
sampling mode. All scores are evaluated against those transitions.

Usage (from h_jepa/):
    PYTHONPATH=. python -u scripts/event_detection_prototype.py \
        --data /tmp/pusht_events/events.h5 \
        --head /tmp/action_response_head_full3k.pt \
        --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
        --stride 5 --out-fig /tmp/event_detection.png
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from action_response import ActionResponseNet, ResponseConfig
from action_response_wm import get_action_stats, predict_level1_once
from train_action_response import encode_rgb, load_checkpoint

PROBE_ACTIONS = np.array([
    [0.60, 0.00],
    [0.60, 0.35],
    [0.60, -0.35],
    [0.25, 0.00],
], dtype=np.float32)  # [K,2]
REFERENCE = np.array([0.60, 0.00], dtype=np.float32)  # [2]


def fixed_probe_sequences(horizon=5):
    probes = np.repeat(PROBE_ACTIONS[:, None, :], horizon, axis=1)   # [K,5,2]
    reference = np.repeat(REFERENCE[None, None, :], horizon, axis=1)  # [1,5,2]
    return probes, reference[0]


@torch.no_grad()
def wm_signatures(model, z, probes, reference, mean, std, device, batch_size=128):
    """z [N,D] -> signature [N,K,D] = F1(z,A_k) - F1(z,A_ref)."""
    level1 = model.get_level(1)
    z = z.to(device)
    n, d = z.shape
    k = len(probes)

    z_rep = z[:, None, :].expand(n, k, d).reshape(n * k, d)
    a_rep = torch.as_tensor(np.repeat(probes[None], n, axis=0)).reshape(n * k, 5, 2)

    outs = []
    for s in range(0, n * k, batch_size):
        outs.append(predict_level1_once(
            level1, z_rep[s:s + batch_size], a_rep[s:s + batch_size], mean, std
        ).float().cpu())
    pred = torch.cat(outs).reshape(n, k, d)

    # reference prediction per state (same batched path)
    a_ref = torch.as_tensor(np.repeat(reference[None, None], n, axis=0)).reshape(n, 1, 5, 2)
    ref_outs = []
    for s in range(0, n, batch_size):
        ref_outs.append(predict_level1_once(
            level1, z[s:s + batch_size], a_ref[s:s + batch_size, 0], mean, std
        ).float().cpu())
    ref = torch.cat(ref_outs)[:, None, :]
    return (pred - ref).numpy()


@torch.no_grad()
def arn_signatures(net, z, probes, reference, device, batch_size=256):
    n, d = z.shape
    k = len(probes)
    z_rep = z[:, None, :].expand(n, k, d).reshape(n * k, d).to(device)
    a_rep = torch.as_tensor(np.repeat(probes[None], n, axis=0)).reshape(n * k, 5, 2).to(device)
    ref_rep = torch.as_tensor(np.repeat(reference[None, None], n * k, axis=0)).reshape(n * k, 5, 2).to(device)
    outs = []
    for s in range(0, n * k, batch_size):
        outs.append(net(z_rep[s:s + batch_size], a_rep[s:s + batch_size],
                        ref_rep[s:s + batch_size]).float().cpu())
    return torch.cat(outs).reshape(n, k, d).numpy()


def signature_change(sig):
    """[T,K,D] -> [T-1] RMS change."""
    d = np.diff(sig, axis=0)
    return np.sqrt((d ** 2).mean(axis=(1, 2)))


def latent_change(z):
    d = np.diff(z, axis=0)
    return np.sqrt((d ** 2).mean(axis=1))


def evaluate_scores(scores, labels, episode_ids, tol=1):
    """AUPRC + best-F1 with +/-tol sample tolerance + FP/episode at that threshold."""
    ap = average_precision_score(labels, scores) if labels.any() else float("nan")

    thresholds = np.unique(np.quantile(scores, np.linspace(0.5, 0.995, 120)))
    best = {"f1": -1.0}
    for thr in thresholds:
        pred = scores >= thr
        tp = fp = fn = 0
        for ep in np.unique(episode_ids):
            m = episode_ids == ep
            p = np.flatnonzero(pred[m])
            t = np.flatnonzero(labels[m])
            matched_t = set()
            for i in p:
                hit = [j for j in t if abs(j - i) <= tol and j not in matched_t]
                if hit:
                    matched_t.add(hit[0])
                    tp += 1
                else:
                    fp += 1
            fn += len(t) - len(matched_t)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best["f1"]:
            n_ep = len(np.unique(episode_ids))
            best = {"f1": f1, "precision": precision, "recall": recall,
                    "threshold": float(thr), "fp_per_episode": fp / n_ep}
    best["auprc"] = float(ap)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-fig", default=None)
    p.add_argument("--tol", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device)
    mean, std = get_action_stats(model, args.ckpt, device)

    with h5py.File(args.data, "r") as f:
        rgb = f["rgb"][()]            # [E,T,224,224,3]
        contacts = f["contacts"][()]  # [E,T]
        states = f["states"][()]
        seeds = f["seeds"][()]

    n_ep, n_steps = contacts.shape
    times = np.arange(0, n_steps, args.stride)
    frames = rgb[:, times]                              # [E,S,H,W,3]
    contact_s = (contacts[:, times] > 0)                # [E,S]

    flat_frames = frames.reshape(-1, *frames.shape[2:])
    z = encode_rgb(model, flat_frames, device, 16)
    z = z.reshape(n_ep, len(times), -1)

    probes, reference = fixed_probe_sequences()
    probes_t = torch.as_tensor(np.repeat(probes[None], 1, axis=0))[0]

    print(f"episodes={n_ep}, steps={n_steps}, sampled={len(times)} per episode, "
          f"stride={args.stride}, probes={len(probes)}")
    print(f"contact step fraction (sampled)={contact_s.mean():.1%}, "
          f"transition events={int((np.diff(contact_s.astype(np.int8), axis=1) != 0).sum())}")

    z_flat = z.reshape(-1, z.shape[-1])
    sig_wm = wm_signatures(model, z_flat, probes, reference, mean, std, device)
    sig_wm = sig_wm.reshape(n_ep, len(times), len(probes), -1)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    net = ActionResponseNet(ResponseConfig(**payload["config"])).to(device)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    sig_arn = arn_signatures(net, z_flat, probes, reference, device)
    sig_arn = sig_arn.reshape(n_ep, len(times), len(probes), -1)

    # ---- per-episode scores / labels ----
    scores = {"WM-signature-change": [], "ARN-signature-change": [],
              "latent-change": [], "random": []}
    labels, episode_ids = [], []
    rng = np.random.default_rng(0)
    per_ep_scores = {}
    for e in range(n_ep):
        events = (np.diff(contact_s[e].astype(np.int8)) != 0).astype(np.int64)
        labels.append(events)
        episode_ids.append(np.full(len(events), e))
        s_wm = signature_change(sig_wm[e])
        s_arn = signature_change(sig_arn[e])
        s_lat = latent_change(z[e].numpy())
        scores["WM-signature-change"].append(s_wm)
        scores["ARN-signature-change"].append(s_arn)
        scores["latent-change"].append(s_lat)
        scores["random"].append(rng.random(len(events)))
        per_ep_scores[e] = {"wm": s_wm, "arn": s_arn, "latent": s_lat,
                            "events": events, "contact": contact_s[e]}

    labels = np.concatenate(labels)
    episode_ids = np.concatenate(episode_ids)
    scores = {k: np.concatenate(v) for k, v in scores.items()}

    print(f"\nprevalence={labels.mean():.3f} (event samples / total)\n")
    print(f"{'signal':<24s} {'AUPRC':>7s} {'bestF1':>7s} {'prec':>6s} {'rec':>6s} "
          f"{'FP/ep':>6s}")
    results = {}
    for name, s in scores.items():
        r = evaluate_scores(s, labels, episode_ids, tol=args.tol)
        results[name] = r
        print(f"{name:<24s} {r['auprc']:7.3f} {r['f1']:7.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['fp_per_episode']:6.2f}")

    # ---- figure: episode with most events ----
    if args.out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        e = int(np.argmax([v["events"].sum() for v in per_ep_scores.values()]))
        d = per_ep_scores[e]
        x = times[1:]
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        for key, name, color in (
            ("wm", "WM signature change", "tab:blue"),
            ("arn", "ARN signature change", "tab:orange"),
            ("latent", "latent change", "tab:green"),
        ):
            y = d[key]
            y = y / (y.max() + 1e-12)
            ax[0].plot(x, y, label=name, color=color, marker="o", ms=3)
        for ev_t in x[d["events"] > 0]:
            ax[0].axvline(ev_t, color="gray", ls="--", alpha=0.5)
        ax[0].set_ylabel("normalized score")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_title(f"episode seed={int(seeds[e])}: scores vs contact transitions (dashed)")
        ax[1].step(times, d["contact"].astype(int), where="post", color="black")
        ax[1].set_ylabel("contact")
        ax[1].set_xlabel("env step")
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=130)
        print(f"\nfigure saved: {args.out_fig}")


if __name__ == "__main__":
    main()
