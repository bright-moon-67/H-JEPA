"""Minimal pre-encoded, paired RAGC dataset format.

Each row is one ANCHOR + pair of action candidates and REALIZED task costs.
Features z0/goal are frozen H-JEPA embeddings from the actual baseline planner.
Different rows from the same anchor must stay in the SAME data split.

NPZ arrays (float32 unless stated):
  anchor_id [N] int64
  z0, goal [N,192]
  goal_is_upper [N] bool
  plus, minus [N,2,10] standardized Level-1 tokens (NOT raw actions)
  plus_cost, minus_cost [N] REAL task costs AFTER baseline continuation.
  epsilon [N] half the L2 distance of symmetric action perturbations.
"""
import numpy as np
import torch


REQUIRED = (
    "anchor_id", "z0", "goal", "goal_is_upper",
    "plus", "minus", "plus_cost", "minus_cost", "epsilon",
)


def load_ragc_npz(path):
    with np.load(path, allow_pickle=False) as f:
        if "label_source" not in f:
            raise ValueError(
                "Missing label_source provenance: refuse to guess whether "
                "targets are true frozen-policy continuation results."
            )
        label_source = str(f["label_source"].item())
        if label_source != "baseline_continuation":
            raise ValueError(
                f"Expected labels from baseline_continuation, got {label_source!r}"
            )
        d = {k: f[k] for k in REQUIRED}
        # Optional provenance / audit fields (never used as training targets).
        for extra in ("source_episode_uid", "task_uid", "baseline_success",
                      "astar_cost", "plus_success", "minus_success"):
            if extra in f:
                d[extra] = f[extra]
    n = len(d["anchor_id"])
    for k in REQUIRED:
        if len(d[k]) != n:
            raise ValueError(f"{k}: expected first dim {n}, got {d[k].shape}")
    if d["z0"].shape != (n, 192) or d["goal"].shape != (n, 192):
        raise ValueError("Expected [N,192] state and goal embeddings")
    for k in ("plus", "minus"):
        if d[k].ndim != 3 or d[k].shape[0] != n or d[k].shape[-1] != 10:
            raise ValueError(f"{k}: expected [N,H,10], got {d[k].shape}")
    if not np.isfinite(d["plus_cost"]).all() or not np.isfinite(d["minus_cost"]).all():
        raise ValueError("Nonfinite realized task costs")
    eps = np.asarray(d["epsilon"], np.float64).reshape(-1)
    if (eps <= 1e-4).any():
        raise ValueError("Invalid finite-difference epsilon")
    norm_half_delta = np.linalg.norm(
        (d["plus"] - d["minus"]).reshape(n, -1) / 2.0, axis=1
    )
    if not np.allclose(norm_half_delta, eps, rtol=1e-3, atol=1e-4):
        raise ValueError(
            "Plus/minus probes are not symmetric with provided epsilon. "
            "Do not gradient-supervise clipped or arbitrary pairs."
        )
    return d


def assert_anchor_splits(train, val):
    a = set(np.asarray(train["anchor_id"]).astype(np.int64).tolist())
    b = set(np.asarray(val["anchor_id"]).astype(np.int64).tolist())
    if not a.isdisjoint(b):
        raise ValueError(f"TRAIN/VAL anchor leakage: {len(a & b)} overlapping anchors")


def assert_episode_splits(train, val):
    """Episode-level isolation.

    Anchor-id disjointness is NOT enough: several windows can be cut from the
    same expert episode, so a train window and a val window of one episode would
    still leak. When `source_episode_uid` is present, require disjoint episode
    sets.
    """
    if "source_episode_uid" not in train or "source_episode_uid" not in val:
        return None
    a = set(np.asarray(train["source_episode_uid"]).astype(str).tolist())
    b = set(np.asarray(val["source_episode_uid"]).astype(str).tolist())
    if not a.isdisjoint(b):
        raise ValueError(
            "TRAIN/VAL source-episode leakage: "
            f"{sorted(a & b)}"
        )
    return {"train_episodes": sorted(a), "val_episodes": sorted(b)}


def to_tensors(data):
    numeric = ("z0", "goal", "goal_is_upper", "plus", "minus",
               "plus_cost", "minus_cost", "epsilon")
    return {
        k: torch.from_numpy(np.ascontiguousarray(v)).float()
        for k, v in data.items()
        if k in numeric
    }


def paired_preference_accuracy(pred_plus, pred_minus, real_plus, real_minus):
    pred = np.asarray(pred_plus) - np.asarray(pred_minus)
    real = np.asarray(real_plus) - np.asarray(real_minus)
    mask = np.abs(real) > 1e-4
    if not mask.any():
        return float("nan")
    return float(np.mean((pred[mask] * real[mask]) > 0))
