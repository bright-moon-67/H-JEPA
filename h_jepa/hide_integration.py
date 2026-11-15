"""Opt-in HiDe projector loading, provenance gates and solver installation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from hide_nav import NavigationProjector, NavigationCostAdapter, prepare_frozen_projector


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_same(name: str, expected: str, actual: str) -> None:
    if not expected or expected != actual:
        raise ValueError(f"HiDe provenance mismatch ({name}): {expected} != {actual}")


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_existing_file(spec, key: str) -> Path:
    """Fail closed with an actionable message instead of ``Path(None)``.

    Behaviour-neutral hardening: the reference installer already refuses to run
    without these artifacts, so this changes only the error text. A null or
    missing path previously surfaced as ``TypeError: expected str, bytes or
    os.PathLike object, not NoneType`` far from the real cause.
    """
    value = spec.get(key, None)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"hide_nav.{key} is required when hide_nav.enabled=true; a null "
            "path would skip a required HiDe provenance gate"
        )
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"hide_nav.{key} not found: {path}")
    return path


def install_hide_from_cfg(cfg, solver, *, checkpoint: Path, normalizer: Path):
    """Install once, only when explicitly enabled and all train-only gates pass.

    This function performs no work at all for default-off configuration.
    """
    spec = cfg.get("hide_nav", None)
    if spec is None or not bool(spec.get("enabled", False)):
        return None
    if getattr(solver, "hide_nav_enabled", False):
        raise RuntimeError("HiDe projector installed twice")
    if sorted(solver.level_solvers) != [1, 2, 3]:
        raise ValueError("HiDe V1 only supports the 3-level AntMaze model")

    checkpoint = Path(checkpoint)
    normalizer = Path(normalizer)
    checkpoint_hash = sha256(checkpoint)
    normalizer_hash = sha256(normalizer)

    head_file = _require_existing_file(spec, "projector_path")
    scale_file = _require_existing_file(spec, "scale_stats_path")
    audit_file = _require_existing_file(spec, "prediction_audit_path")
    # weights_only prevents arbitrary Python object construction from model file.
    payload = torch.load(head_file, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    if meta.get("source_split") != "train" or meta.get("render_domain") != "live_eval_pipeline":
        raise ValueError("HiDe projector requires TRAIN states encoded from LIVE evaluation render domain")
    if not meta.get("source_data_sha256"):
        raise ValueError("Missing training data fingerprint")
    if meta.get("real_state_gate") is not True or meta.get("goal_encoder_parity_passed") is not True:
        raise ValueError("Required real-state and goal-encoder gates failed")
    _check_same("checkpoint", meta["checkpoint_sha256"], checkpoint_hash)
    _check_same("normalizer", meta["normalizer_sha256"], normalizer_hash)

    projector = NavigationProjector(**payload["architecture"])
    projector.load_state_dict(payload["state_dict"], strict=True)
    d = int(projector.latent_dim)
    if d != 384:
        raise ValueError(f"AntMaze L3 expected 384-D head, got {d}")

    audit = _load_json(audit_file)
    if audit.get("status") != "PASS" or audit.get("source_split") != "train_heldout":
        raise ValueError("Prediction audit on frozen L3 latent must PASS before enabling HiDe")
    _check_same("head audit", audit.get("head_sha256"), sha256(head_file))
    _check_same("checkpoint audit", audit.get("checkpoint_sha256"), checkpoint_hash)
    if int(audit.get("num_pairs", 0)) < 32:
        raise ValueError("Too few independent prediction pairs for offline audit")

    stats = _load_json(scale_file)
    if stats.get("source_split") != "train" or stats.get("calibration") != "frozen_rollouts":
        raise ValueError("Cost scales must be derived from frozen TRAIN rollouts")
    _check_same("scale/head", stats.get("head_sha256"), sha256(head_file))
    _check_same("scale/checkpoint", stats.get("checkpoint_sha256"), checkpoint_hash)
    _check_same("scale/normalizer", stats.get("normalizer_sha256"), normalizer_hash)
    if min(int(stats.get("num_l2", 0)), int(stats.get("num_l3", 0))) < 32:
        raise ValueError("Need >=32 training rollout costs per level")

    alpha_l3 = float(spec.get("alpha_l3", 0.25))
    alpha_l2 = float(spec.get("alpha_l2", 0.25))
    if not all(math.isfinite(x) and 0 < x <= 1. for x in (alpha_l3, alpha_l2)):
        raise ValueError("HiDe α weights must be fixed positive scalars <= 1")
    # Solver may be nested in shared policy model, use explicit original device.
    dev = next(solver.level_models[2].parameters()).device
    prepare_frozen_projector(projector, dev)
    for level, alpha in ((3, alpha_l3), (2, alpha_l2)):
        weight = float(stats[f"scale_l{level}"])
        adapter = NavigationCostAdapter(
            projector, level=level, alpha=alpha, scale=weight
        )
        solver.level_solvers[level].model.hide_nav_cost = adapter

    solver.hide_nav_enabled = True
    return {
        "checkpoint_sha256": checkpoint_hash,
        "normalizer_sha256": normalizer_hash,
        "projector_sha256": sha256(head_file),
        "scale_sha256": sha256(scale_file),
        "prediction_audit_sha256": sha256(audit_file),
        "alpha_l2": alpha_l2,
        "alpha_l3": alpha_l3,
    }
