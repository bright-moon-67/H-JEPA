"""Response-prediction metrics: R^2, cosine, magnitude ratio, RMS.

These answer different questions than plain MSE:
  - r2              : does the model beat always predicting zero?
  - cosine          : does it get the DIRECTION of the action effect right?
  - magnitude_ratio : does it under/over-state the effect size? (near 0 means
                      the head collapsed to near-zero responses)
  - target_rms      : is there enough action-response signal in the data?
  - pred_rms        : is the head degenerating toward zero output?
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_response_metrics(pred, target):
    """pred, target: [N, D] -> dict of python floats."""
    assert pred.ndim == target.ndim == 2, (pred.shape, target.shape)
    assert pred.shape == target.shape, (pred.shape, target.shape)

    pred = pred.float()
    target = target.float()

    err = (pred - target).square().mean(dim=-1)
    energy = target.square().mean(dim=-1)

    mse = err.mean()
    zero_mse = energy.mean()
    r2 = 1.0 - mse / zero_mse.clamp_min(1e-12)

    target_norm = target.norm(dim=-1)
    pred_norm = pred.norm(dim=-1)

    valid = target_norm > 1e-3
    if valid.any():
        cosine = F.cosine_similarity(pred[valid], target[valid], dim=-1).mean()
        magnitude_ratio = (
            pred_norm[valid].mean() / target_norm[valid].mean().clamp_min(1e-12)
        )
    else:
        cosine = torch.tensor(float("nan"))
        magnitude_ratio = torch.tensor(float("nan"))

    return {
        "mse": mse.item(),
        "zero_mse": zero_mse.item(),
        "r2": r2.item(),
        "cosine": cosine.item(),
        "magnitude_ratio": magnitude_ratio.item(),
        "target_rms": energy.mean().sqrt().item(),
        "pred_rms": pred.square().mean().sqrt().item(),
        "n": int(pred.shape[0]),
    }


def format_metrics_row(name, metrics):
    return (
        f"{name:<22s} n={metrics['n']:5d} "
        f"mse={metrics['mse']:.5f} zero={metrics['zero_mse']:.5f} "
        f"R2={metrics['r2']:+.4f} cos={metrics['cosine']:+.4f} "
        f"mag={metrics['magnitude_ratio']:.3f} "
        f"|pred|={metrics['pred_rms']:.3f} |tgt|={metrics['target_rms']:.3f}"
    )
