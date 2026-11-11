"""Frozen H-JEPA Level-1 encoder + independently trained ActionResponseNet.

Run from h_jepa/:
  python -u scripts/train_action_response.py \
    --train /tmp/pusht_cf_train.npz \
    --val /tmp/pusht_cf_val.npz \
    --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
    --out /tmp/action_response_head.pt --steps 20

No H-JEPA world-model weights are changed.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from action_response import ActionResponseNet, ResponseConfig, response_loss
from action_response_data import (
    build_pair_tensors,
    encode_anchors,
    load_anchor_h5,
)
from action_response_metrics import compute_response_metrics, format_metrics_row
from data import IMAGENET_STATS


def load_checkpoint(path, device):
    # Only load checkpoints you created or trust. weights_only=False
    # can execute code from an untrusted pickle.
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "get_level"):
        raise TypeError("Expected an HJEPA model with get_level(1)")
    model = model.to(device).eval()
    # Parity with planning_eval.load_model: the planner enables interpolated
    # positional encodings before encoding observations.
    model.interpolate_pos_encoding = True
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_rgb(model, images, device, batch_size=16):
    """Matches planning_eval.img_transform and JEPA.encode input format.

    images: uint8 [N,224,224,3] -> [N,192].
    No action is needed to encode a single observation.
    """
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 NHWC RGB, got {images.shape}/{images.dtype}")
    means = torch.tensor(IMAGENET_STATS["mean"], device=device).view(1, 3, 1, 1)
    stds = torch.tensor(IMAGENET_STATS["std"], device=device).view(1, 3, 1, 1)
    outs = []
    for start in range(0, len(images), batch_size):
        batch_np = np.ascontiguousarray(images[start:start + batch_size])
        x = torch.from_numpy(batch_np).to(device).permute(0, 3, 1, 2).float()
        x = (x.div(255.0) - means) / stds
        # Existing repo API:
        # model.get_level(1).encode({"pixels": [B,T,C,H,W]})["embed_0"]
        z = model.get_level(1).encode(
            {"pixels": x.unsqueeze(1)}, key="pixels"
        )["embed_0"]
        if z.ndim != 3 or z.shape[1] != 1:
            raise RuntimeError(f"Unexpected JEPA latent shape: {tuple(z.shape)}")
        outs.append(z[:, 0].float().cpu())
    return torch.cat(outs, dim=0)


def prepare_tensors(npz_path, model, device, encode_batch_size):
    with np.load(npz_path, allow_pickle=False) as data:
        rgb0 = data["rgb0"]
        rgb_a = data["rgb_a"]
        rgb_b = data["rgb_b"]
        acts_a = data["acts_a"].astype(np.float32)
        acts_b = data["acts_b"].astype(np.float32)

    if acts_a.ndim != 3 or acts_a.shape[1:] != (5, 2):
        raise ValueError(f"Expected [N,5,2] raw Push-T actions, got {acts_a.shape}")
    z0 = encode_rgb(model, rgb0, device, encode_batch_size)
    z_a = encode_rgb(model, rgb_a, device, encode_batch_size)
    z_b = encode_rgb(model, rgb_b, device, encode_batch_size)
    if z0.shape != z_a.shape or z0.shape != z_b.shape:
        raise RuntimeError("Encoded shapes do not match")
    return {
        "z0": z0,
        "a": torch.from_numpy(acts_a),
        "b": torch.from_numpy(acts_b),
        "target": (z_a - z_b).detach(),
    }


def prepare_anchor_tensors(h5_path, model, device, encode_batch_size):
    """Anchor H5 -> expanded pair tensors (all C(K,2) pairs per anchor)."""
    data = load_anchor_h5(h5_path)
    z0, z_end = encode_anchors(model, data, device, encode_batch_size)
    pairs = build_pair_tensors(z0, z_end, data["actions"])
    pairs["anchor_idx"] = torch.as_tensor(pairs["anchor_idx"])
    return pairs


@torch.no_grad()
def evaluate(net, tensors, device, batch_size=128):
    net.eval()
    n = len(tensors["z0"])
    preds, targets = [], []
    for k in range(0, n, batch_size):
        items = {key: v[k:k+batch_size].to(device) for key, v in tensors.items()
                 if key in ("z0", "a", "b", "target")}
        prediction = net(items["z0"], items["a"], items["b"])
        preds.append(prediction.float().cpu())
        targets.append(items["target"].float().cpu())
    net.train()
    return compute_response_metrics(torch.cat(preds), torch.cat(targets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=20)  # smoke, NOT a full training run
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = load_checkpoint(args.ckpt, device)

    print("Encoding images once with frozen H-JEPA...", flush=True)
    loader = prepare_anchor_tensors if args.train.endswith(".h5") else prepare_tensors
    val_loader = prepare_anchor_tensors if args.val.endswith(".h5") else prepare_tensors
    train = loader(args.train, model, device, args.encode_batch_size)
    val = val_loader(args.val, model, device, args.encode_batch_size)
    if len(train["z0"]) == 0 or len(val["z0"]) == 0:
        raise ValueError("Empty train/val dataset")
    latent_dim = int(train["z0"].shape[-1])
    if latent_dim != 192:
        raise ValueError(f"Push-T L1 was expected to be 192-D, got {latent_dim}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cfg = ResponseConfig(state_dim=latent_dim, action_dim=2, horizon=5)
    net = ActionResponseNet(cfg).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    initial_val = evaluate(net, val, device)
    print("initial " + format_metrics_row("val", initial_val), flush=True)
    swap_enabled = "anchor_idx" in train
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(
            len(train["z0"]), (args.batch_size,), generator=gen
        )
        batch = {
            key: value[idx].to(device)
            for key, value in train.items()
            if key in ("z0", "a", "b", "target")
        }
        if swap_enabled:
            # Random A/B order augmentation; network is antisymmetric by design.
            swap = torch.rand(len(idx), generator=gen) < 0.5
            swap = swap.to(device)
            a, b = batch["a"], batch["b"]
            batch["a"] = torch.where(swap[:, None, None], b, a)
            batch["b"] = torch.where(swap[:, None, None], a, b)
            target = torch.where(swap[:, None], -batch["target"], batch["target"])
        else:
            target = batch["target"]

        net.train()
        pred = net(batch["z0"], batch["a"], batch["b"])
        loss = response_loss(pred, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step == 1 or step % max(1, args.steps // 5) == 0:
            metrics = evaluate(net, val, device)
            print(
                f"step={step:5d} train_mse={loss.item():.5f} "
                + format_metrics_row("val", metrics),
                flush=True,
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(cfg),
        "state_dict": net.state_dict(),
        "world_model_checkpoint": str(args.ckpt),
        "train_dataset": str(args.train),
        "val_dataset": str(args.val),
        "steps": args.steps,
        "validation_metrics": evaluate(net, val, device),
    }, args.out)
    print(f"saved head to {args.out}", flush=True)


if __name__ == "__main__":
    main()
