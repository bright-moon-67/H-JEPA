"""Frozen H-JEPA Level-1 counterfactual baseline (plan2 P2).

Predicts with the ORIGINAL world model rather than a new head:

    z_hat_A = F1(z0, A),  z_hat_B = F1(z0, B)  ->  r_wm = z_hat_A - z_hat_B

Actions are normalized with the checkpoint's normalizer.pt (per-dim over the
raw 2-D env action), then flattened into one 10-D Level-1 action token
(frameskip 5 x action_dim 2), matching training and `JEPA.rollout`.

`numeric_consistency_check` must pass (manual call == rollout first-step
prediction) before any comparison numbers are trusted.
"""
import numpy as np
import torch

from data import NORMALIZER_ARTIFACT_FILENAME, load_normalizer_artifact


def get_action_stats(model, ckpt_path, device):
    artifact = getattr(model, "normalizer_artifact", None)
    if artifact is None:
        from pathlib import Path
        artifact = load_normalizer_artifact(
            Path(ckpt_path).parent / NORMALIZER_ARTIFACT_FILENAME
        )
    stats = artifact["stats"]["action"]
    mean = torch.as_tensor(np.asarray(stats["mean"]).reshape(-1),
                           dtype=torch.float32, device=device)
    std = torch.as_tensor(np.asarray(stats["std"]).reshape(-1),
                          dtype=torch.float32, device=device)
    if mean.numel() != 2 or std.numel() != 2:
        raise ValueError(f"expected 2-D action stats, got {mean.numel()}/{std.numel()}")
    if not bool((std > 0).all()):
        raise ValueError("non-positive action std")
    return mean, std


def normalize_actions(raw_actions, mean, std):
    """[B,5,2] raw env actions (tensor or ndarray) -> standardized tensor."""
    if not isinstance(raw_actions, torch.Tensor):
        raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32)
    raw_actions = raw_actions.to(device=mean.device, dtype=torch.float32)
    if raw_actions.ndim != 3 or raw_actions.shape[1:] != (5, 2):
        raise ValueError(f"expected [B,5,2], got {tuple(raw_actions.shape)}")
    return (raw_actions - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def action_tokens(raw_actions, mean, std):
    """[B,5,2] raw actions -> [B,1,10] Level-1 action token."""
    norm = normalize_actions(raw_actions, mean, std)
    return norm.reshape(norm.shape[0], 1, -1)


@torch.no_grad()
def predict_level1_once(level1, z0, raw_actions, mean, std):
    """z0 [B,192], raw_actions [B,5,2] -> predicted latent after 5 env steps."""
    if z0.ndim != 2:
        raise ValueError(f"expected z0 [B,D], got {tuple(z0.shape)}")
    tokens = action_tokens(raw_actions, mean, std)
    latent = level1.action_encoder(tokens)  # Identity for Push-T level 1
    act_emb = level1.action_embed(latent)
    pred = level1.predict(z0.unsqueeze(1), act_emb)
    return pred[:, -1, :]


@torch.no_grad()
def wm_response(model, z0, actions_a, actions_b, mean, std):
    level1 = model.get_level(1)
    za = predict_level1_once(level1, z0, actions_a, mean, std)
    zb = predict_level1_once(level1, z0, actions_b, mean, std)
    return za - zb


@torch.no_grad()
def numeric_consistency_check(model, images, raw_actions, mean, std,
                              device, atol=1e-4):
    """Manual single-step prediction must equal rollout's first prediction.

    images: uint8 NHWC [B,224,224,3]; raw_actions: [B,5,2].
    Returns (max_abs_diff, manual_pred, rollout_pred).
    """
    from data import IMAGENET_STATS

    level1 = model.get_level(1)
    means = torch.tensor(IMAGENET_STATS["mean"], device=device).view(1, 3, 1, 1)
    stds = torch.tensor(IMAGENET_STATS["std"], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    x = x.permute(0, 3, 1, 2).float().div(255.0)
    x = ((x - means) / stds).unsqueeze(1)  # [B,1,3,H,W]

    info = level1.encode({"pixels": x}, key="pixels")
    z0 = info["embed_0"][:, 0]

    token = action_tokens(raw_actions, mean, std)  # [B,1,10]
    # JEPA.rollout expects pixels [B,S,T,C,H,W] and actions [B,S,T,token_dim].
    pixels_bs = x.unsqueeze(1)                      # [B,1,1,3,H,W]
    action_seq = torch.stack([token, token], dim=2)  # [B,1,2,10]
    out = level1.rollout(
        {"pixels": pixels_bs},
        action_seq,
        history_size=1,
    )
    rollout_pred = out["predicted_embed_0"][:, 0, 1, :]

    manual_pred = predict_level1_once(level1, z0, raw_actions, mean, std)

    diff = (manual_pred - rollout_pred).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"manual predictor != JEPA.rollout first step (max|diff|={diff:.3e}, "
            f"atol={atol:g}); fix normalizer/token layout before trusting results"
        )
    return diff, manual_pred, rollout_pred
