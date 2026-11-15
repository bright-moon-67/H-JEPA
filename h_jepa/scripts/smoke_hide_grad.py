#!/usr/bin/env python3
"""GPU smoke: HiDe navigation cost wiring, shapes, gradients and overhead.

This is an IMPLEMENTATION check on the real frozen checkpoint, NOT a success
experiment. It attaches the reference `NavigationCostAdapter` directly to the
real `LevelCostModel`s (bypassing the installer, which stays fail-closed until
the prediction audit PASSes) and verifies the contract from AGENT_TASK IV.16:

  - head in eval mode, every parameter frozen
  - L3 cost [B,400], L2 cost [B,100] with the ORIGINAL sample counts
  - L2 predictions and L3 targets both shaped [*,384] in L3 space
  - cost_last_n=1 and the original criterion shape/time index untouched
  - finite, nonzero action gradients through the frozen head
  - disabled adapter is a bitwise no-op
  - measured extra cost of the navigation term

  python scripts/smoke_hide_grad.py \
    --checkpoint "$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt" \
    --head artifacts/hide/nav_projector.pt --output artifacts/hide/gpu_smoke.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data import NORMALIZER_ARTIFACT_FILENAME
from hide_integration import sha256
from hide_nav import (
    NavigationCostAdapter, NavigationProjector, navigation_match_cost,
    prepare_frozen_projector,
)
from hierarchical_solver import build_hierarchical_solver
from planning_eval import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("config/eval/ant_l3.yaml"))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU smoke requires CUDA")
    ckpt = args.checkpoint.resolve()
    norm = ckpt.parent / NORMALIZER_ARTIFACT_FILENAME
    cfg = OmegaConf.load(args.config)
    cfg.policy = str(ckpt)
    model = load_model(cfg)
    solver = build_hierarchical_solver(cfg, model)
    device = next(solver.level_models[2].parameters()).device

    payload = torch.load(args.head, map_location="cpu", weights_only=True)
    head = NavigationProjector(**payload["architecture"])
    head.load_state_dict(payload["state_dict"], strict=True)
    prepare_frozen_projector(head, device)
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise RuntimeError("Head must stay frozen in eval mode")

    plan = cfg.hierarchical_plan_config
    report = {
        "implementation_smoke_only": True,
        "checkpoint_sha256": sha256(ckpt),
        "normalizer_sha256": sha256(norm),
        "head_sha256": sha256(args.head),
        "device": str(device),
        "head_requires_grad": [bool(p.requires_grad) for p in head.parameters()],
        "levels": {},
    }

    B = args.batch
    for level in (3, 2):
        cost_model = solver.level_solvers[level].model
        samples = int(cfg.solver.solvers[f"level{level}"].num_samples)
        horizon = int(plan[f"level{level}"].horizon)
        cost_last_n = int(plan[f"level{level}"].cost_last_n)
        if cost_last_n != 1:
            raise ValueError("smoke expects cost_last_n=1")
        adapter = NavigationCostAdapter(head, level=level, alpha=0.25, scale=1.0)
        info = {
            "embed_0": torch.randn(B, samples, 1, 384, device=device),
            "goal_embed_0": torch.randn(B, samples, 1, 384, device=device),
            "goal_embed_0_level": 3,
            "goal_embed_0_in_upper_space": level == 2,
            "dense_upper_goal_projection": False,
            "cost_last_n": cost_last_n,
            "intermediate_cost_weight": 0.0,
            "action_cost_weight": 0.0,
            "pixels": torch.zeros(B, samples, 1, 1, 1, 3, device=device),
            "_hide_nav_active": True,
        }
        actions = torch.randn(B, samples, horizon, 4, device=device)
        actions.requires_grad_(True)

        # --- base cost (adapter absent) ---
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            base = cost_model.get_cost(dict(info), actions)
        torch.cuda.synchronize()
        base_s = (time.time() - t0) / args.iters

        # --- base cost + navigation cost ---
        cost_model.hide_nav_cost = adapter
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            total = cost_model.get_cost(dict(info), actions)
        torch.cuda.synchronize()
        nav_s = (time.time() - t0) / args.iters

        total.sum().backward()
        grad = actions.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"L{level}: action gradient missing or non-finite")
        if grad.abs().sum().item() <= 0:
            raise RuntimeError(f"L{level}: zero action gradient")

        # navigation term magnitude, measured on the same candidates
        with torch.no_grad():
            rolled = cost_model.rollout(dict(info), actions.detach())
            pred = rolled["predicted_embed_0"]
            proj, _ = cost_model._project_to_goal_space(
                pred, None, use_upper_space=level == 2, dense=False, goal_level=3
            )
            nav = navigation_match_cost(
                head, proj, info["goal_embed_0"], cost_last_n=cost_last_n
            )
        if bool(info["goal_embed_0"].requires_grad):
            raise RuntimeError("goal embedding must never be modified in place")

        # disabled adapter must return the identical tensor
        off = NavigationCostAdapter(head, level=level, alpha=0.25, scale=0.5)
        base_t = cost_model.model.criterion(
            {**info, "predicted_embed_0": proj}
        )
        same = off(base_t, proj, {**info, "_hide_nav_active": False}) is base_t

        if tuple(base.shape) != (B, samples):
            raise RuntimeError(f"L{level}: base cost shape {tuple(base.shape)}")
        if level == 2 and proj.shape[-1] != 384:
            raise RuntimeError("L2 projected predictions must be L3 384-D")

        report["levels"][level] = {
            "cost_shape": list(base.shape),
            "cost_shape_with_nav": list(total.shape),
            "num_samples": samples,
            "horizon": horizon,
            "projected_prediction_shape": list(proj.shape),
            "predicted_rollout_shape": list(pred.shape),
            "base_cost_median": float(base.median()),
            "nav_cost_median": float(nav.median()),
            "alpha_scaled_nav_median": float(0.25 * nav.median()),
            "gradient_abs_sum": float(grad.abs().sum()),
            "gradient_finite": bool(torch.isfinite(grad).all()),
            "disabled_adapter_identity": bool(same),
            "get_cost_seconds_base": base_s,
            "get_cost_seconds_with_nav": nav_s,
            "cost_overhead_ratio": nav_s / base_s,
        }
        cost_model.hide_nav_cost = None
        print(json.dumps(report["levels"][level], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
