"""Shared data utilities for counterfactual anchor datasets.

Anchor-centric layout (one physical state -> K real action branches):
    rgb0        [N,224,224,3] uint8    shared anchor image
    end_rgb     [N,K,224,224,3] uint8  end image per branch
    actions     [N,K,5,2] float32      raw Push-T env actions per branch
    state0      [N,7] float32
    end_state   [N,K,7] float32
    contacts    [N,K] int32            post-hoc audit only, never model input
    modes       [N]                    sampling intent label (free/approach/push)
    anchor_seed [N] int64

Pairs are derived from branches (i,j) -> R target = z_end[i] - z_end[j].
Train/val/test must be split BY ANCHOR, never by pair.
"""
from itertools import combinations

import h5py
import numpy as np
import torch

DATASET_KEYS = (
    "rgb0", "end_rgb", "actions", "state0", "end_state",
    "contacts", "modes", "anchor_seed", "prefix_len",
)


def save_anchor_h5(path, arrays, compression="gzip"):
    with h5py.File(path, "w") as f:
        for key in DATASET_KEYS:
            if key not in arrays:
                continue
            data = np.asarray(arrays[key])
            if data.dtype.kind == "U":  # h5py cannot store numpy unicode
                data = data.astype("S")
            kwargs = {}
            if data.dtype.kind in "iuf":
                kwargs["compression"] = compression
                kwargs["compression_opts"] = 4
            f.create_dataset(key, data=data, **kwargs)
        f.attrs["format"] = "hjepa_action_response_anchors_v1"


def load_anchor_h5(path):
    with h5py.File(path, "r") as f:
        out = {key: f[key][()] for key in f.keys() if key in DATASET_KEYS}
    return out


def anchor_pair_index(n_anchors, n_branches):
    """All C(K,2) branch pairs per anchor -> (anchor_idx, i, j) arrays."""
    combos = list(combinations(range(n_branches), 2))
    anchor_idx = np.repeat(np.arange(n_anchors), len(combos))
    i_idx = np.tile(np.array([c[0] for c in combos]), n_anchors)
    j_idx = np.tile(np.array([c[1] for c in combos]), n_anchors)
    return anchor_idx, i_idx, j_idx


@torch.no_grad()
def encode_rgb(model, images, device, batch_size=16):
    """uint8 NHWC RGB [N,224,224,3] -> frozen Level-1 latent [N,192].

    Matches planning_eval.img_transform: ToImage -> float/255 -> ImageNet
    Normalize -> Resize(img_size=224); JEPA.encode expects [B,T,3,H,W].
    """
    from data import IMAGENET_STATS

    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError(f"Expected uint8 NHWC RGB, got {images.shape}/{images.dtype}")
    means = torch.tensor(IMAGENET_STATS["mean"], device=device).view(1, 3, 1, 1)
    stds = torch.tensor(IMAGENET_STATS["std"], device=device).view(1, 3, 1, 1)
    outs = []
    for start in range(0, len(images), batch_size):
        batch_np = np.ascontiguousarray(images[start:start + batch_size])
        x = torch.from_numpy(batch_np).to(device).permute(0, 3, 1, 2).float()
        x = (x.div(255.0) - means) / stds
        z = model.get_level(1).encode(
            {"pixels": x.unsqueeze(1)}, key="pixels"
        )["embed_0"]
        if z.ndim != 3 or z.shape[1] != 1:
            raise RuntimeError(f"Unexpected JEPA latent shape: {tuple(z.shape)}")
        outs.append(z[:, 0].float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def encode_anchors(model, data, device, batch_size=16):
    """Encode rgb0 once and every branch end image once.

    Returns z0 [N,D], z_end [N,K,D].
    """
    rgb0 = data["rgb0"]
    end_rgb = data["end_rgb"]  # [N,K,224,224,3]
    n_anchors, n_branches = end_rgb.shape[:2]

    z0 = encode_rgb(model, rgb0, device, batch_size)
    flat = end_rgb.reshape(n_anchors * n_branches, *end_rgb.shape[2:])
    z_end = encode_rgb(model, flat, device, batch_size)
    z_end = z_end.reshape(n_anchors, n_branches, -1)
    return z0, z_end


def build_pair_tensors(z0, z_end, actions, swap=None):
    """Expand anchors into branch pairs.

    Returns dict with z0 [P,D], a [P,5,2], b [P,5,2], target [P,D].
    `swap` optionally is a bool array [P]; where True, a/b (and target sign)
    are swapped, so pair ordering augmentation stays label-consistent.
    """
    n_anchors, n_branches = z_end.shape[:2]
    ai, bi, bj = anchor_pair_index(n_anchors, n_branches)

    z0p = z0[ai]
    za = z_end[ai, bi]
    zb = z_end[ai, bj]
    a = torch.as_tensor(actions[ai, bi], dtype=torch.float32)
    b = torch.as_tensor(actions[ai, bj], dtype=torch.float32)

    target = za - zb
    if swap is not None:
        swap = torch.as_tensor(swap, dtype=torch.bool)
        a, b = torch.where(swap[:, None, None], b, a), torch.where(swap[:, None, None], a, b)
        target = torch.where(swap[:, None], -target, target)

    return {"z0": z0p.float(), "a": a, "b": b, "target": target.float(),
            "anchor_idx": torch.as_tensor(ai), "branch_i": torch.as_tensor(bi),
            "branch_j": torch.as_tensor(bj)}
