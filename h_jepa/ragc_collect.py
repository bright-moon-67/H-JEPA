"""Reusable RAGC counterfactual-oracle collector.

Extracted from `scripts/ragc_oracle_pilot.py` so the pilot and the Stage-1
collector share ONE implementation of:

  * the original frozen H-JEPA planning loop (unchanged L1/L2 schedules),
  * physical anchor restore + prefix replay,
  * antithetic normalized action perturbations around the realized plan A*,
  * frozen-policy continuation to the episode budget,
  * STRICT replay validation of the original-action (A*) branch.

Design rules (Stage 1 requirements):
  - A strict replay failure NEVER deletes the check: the anchor is marked
    `valid=False`, the reason and the first divergence position are recorded,
    and the anchor is excluded from training labels. Collection continues.
  - Every finished anchor is appended to a JSONL file immediately (fsync), so a
    later crash cannot destroy already-computed (expensive) data.
  - Realized cost difference (candidate - A*) is stored per candidate.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import torch
import stable_worldmodel.envs  # noqa: F401
from torch.nn.attention import SDPBackend, sdpa_kernel

from action_response_wm import get_action_stats
from data import IMAGENET_STATS
from eval_config_utils import _build_policy_plan_config
from hierarchical_solver import build_hierarchical_solver
from planning_eval import load_model, load_eval_config
from ragc_oracle import (
    antithetic_candidates,
    counterfactual_fingerprint,
    raw_actions_from_tokens,
    restore_eval_anchor,
    score_continuation,
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def make_env():
    return gym.make(
        "swm/PushT-v1",
        render_mode="rgb_array",
        resolution=224,
        with_target=True,
        relative=True,
    )


def image_tensor(image, device):
    x = torch.as_tensor(np.ascontiguousarray(image), device=device)
    x = x.permute(2, 0, 1).float().div(255.0)
    mean = torch.as_tensor(IMAGENET_STATS["mean"], device=device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STATS["std"], device=device).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).unsqueeze(0)


def model_info(env, device, goal_pixels, first_pixels=None):
    frame = (
        first_pixels
        if first_pixels is not None
        else np.asarray(env.render(), dtype=np.uint8)
    )
    return {
        "pixels": image_tensor(frame, device),
        "goal": image_tensor(goal_pixels, device),
    }


def as_numpy_state(ep):
    s = ep["state"]
    return s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)


def copy_info(info):
    return {
        k: v.detach().clone() if torch.is_tensor(v) else v
        for k, v in info.items()
    }


def append_jsonl(path, record):
    """Append one record and fsync so a crash keeps the finished work."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def first_divergence(a, b, atol=1e-6):
    """Index of the first raw-action step where two traces differ."""
    a = np.asarray(a, np.float64).reshape(-1, 2)
    b = np.asarray(b, np.float64).reshape(-1, 2)
    n = min(len(a), len(b))
    for i in range(n):
        if np.abs(a[i] - b[i]).max() > atol:
            return i
    if len(a) != len(b):
        return n
    return -1


@dataclass
class TaskSpec:
    """One evaluation-style task window."""

    task_uid: str
    source_episode_uid: str
    split: str
    payload: dict
    seed_idx: int = 0
    goal_offset_steps: int = 60
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# collector
# --------------------------------------------------------------------------- #
class RAGCOracleCollector:
    def __init__(
        self,
        ckpt: str,
        *,
        config_name: str = "pusht_l2",
        seed: int = 42,
        epsilon: float = 0.30,
        directions: int = 1,
        jsonl_path: str | None = None,
        verbose: bool = True,
    ):
        if directions < 1:
            raise ValueError("directions must be >= 1")
        self.ckpt = ckpt
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.directions = int(directions)
        self.jsonl_path = jsonl_path
        self.verbose = verbose
        self._log = print if verbose else (lambda *a, **k: None)

        cfg = load_eval_config(config_name)
        cfg.policy = ckpt
        self.cfg = cfg
        self.model = load_model(cfg)
        self.device = next(self.model.parameters()).device
        mean_t, std_t = get_action_stats(self.model, ckpt, self.device)
        self.mean = mean_t.detach().cpu().numpy()
        self.std = std_t.detach().cpu().numpy()

        self.solver = build_hierarchical_solver(cfg, self.model)
        plan_cfg = _build_policy_plan_config(cfg)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
        self.solver.configure(action_space=act_space, n_envs=1, config=plan_cfg)
        self.solver.capture_level1 = {}

        l1solver = self.solver.level_solvers[1]
        orig_l1solve = l1solver.solve

        def record_level1(*a, **kw):
            self.solver.capture_level1["info"] = copy_info(kw["info_dict"])
            return orig_l1solve(*a, **kw)

        l1solver.solve = record_level1
        self.planner_gens = [
            (child.torch_gen, child.torch_gen.initial_seed())
            for child in self.solver.level_solvers.values()
            if hasattr(child, "torch_gen")
        ]
        self.budget = int(cfg.eval.eval_budget)
        self.env = make_env()
        # Optional hook: called as trace_hook(task, trace) after each baseline
        # rollout. Used by the pilot to keep its --dump-trace feature.
        self.trace_hook = None

    # -- lifecycle ---------------------------------------------------------- #
    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- primitives --------------------------------------------------------- #
    def _reseed(self, seed_idx: int):
        for gen, base in self.planner_gens:
            gen.manual_seed(int(base) + int(seed_idx))

    def _solver_rng(self):
        return {
            level: child.torch_gen.get_state().clone()
            for level, child in self.solver.level_solvers.items()
        }

    def _reset_solver_rng(self, state):
        for level, rng_state in state.items():
            self.solver.level_solvers[level].torch_gen.set_state(rng_state.clone())

    def _execute_tokens(self, std_actions, limit):
        raw = raw_actions_from_tokens(std_actions, self.mean, self.std)
        n = 0
        succeeded = False
        for a in raw:
            if n >= limit:
                break
            _, _, done, truncated, _ = self.env.step(a.astype(np.float32))
            n += 1
            succeeded |= bool(done)
            if done or truncated:
                break
        return raw[:n].copy(), bool(succeeded), n

    # -- baseline trace ----------------------------------------------------- #
    def trace_baseline(self, init_state, goal_state, ep_pixels):
        """Official-style rollout of the frozen policy, with per-step capture."""
        env = self.env
        env.reset()
        env.unwrapped._set_state(np.asarray(init_state, dtype=np.float64))
        env.unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
        goal_pixels = np.asarray(ep_pixels[-1], dtype=np.uint8)
        first_pixels = np.asarray(ep_pixels[0], dtype=np.uint8)

        path, elapsed, success, recorded = [], 0, False, []
        first_call = True
        while elapsed < self.budget and not success:
            info = model_info(
                env,
                self.device,
                goal_pixels,
                first_pixels=first_pixels if first_call else None,
            )
            first_call = False
            self.solver.capture_level1.clear()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                result = self.solver.solve(
                    info, steps_taken=elapsed, eval_budget=self.budget
                )
            std_action = result["actions"][0].detach().cpu().numpy()
            std_action = std_action[: min(2, len(std_action))]
            capture = copy_info(self.solver.capture_level1["info"])
            if float(capture["action_cost_weight"]) != 0.2:
                raise AssertionError(
                    "Original level-1 action-cost weight must remain 0.2"
                )
            entry = {
                "step": elapsed,
                "real_anchor_fingerprint": counterfactual_fingerprint(env),
                "real_anchor_rgb": np.asarray(env.render(), dtype=np.uint8).copy(),
                "actions_standard": std_action,
                "prefix_raw": np.asarray(recorded, np.float32).reshape(-1, 2).copy(),
                "rng_after_solve": self._solver_rng(),
                "level1_info": capture,
                "pixels_sha1": hashlib.sha1(
                    info["pixels"].detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
                "goal_sha1": hashlib.sha1(
                    info["goal"].detach().float().cpu().numpy().tobytes()
                ).hexdigest(),
            }
            path.append(entry)
            chunk, done, n = self._execute_tokens(
                std_action, limit=self.budget - elapsed
            )
            recorded.extend(chunk)
            elapsed += n
            success |= done

        return {
            "success": bool(success),
            "elapsed": elapsed,
            "path": path,
            "full_raw_actions": np.asarray(recorded, np.float32).reshape(-1, 2),
        }

    # -- continuation ------------------------------------------------------- #
    def continue_baseline(self, from_step, rng_after_anchor, goal_pixels):
        self._reset_solver_rng(rng_after_anchor)
        elapsed = from_step
        success = False
        continuation = []
        while elapsed < self.budget and not success:
            info = model_info(self.env, self.device, goal_pixels)
            self.solver.capture_level1.clear()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                result = self.solver.solve(
                    info, steps_taken=elapsed, eval_budget=self.budget
                )
            actions = result["actions"][0].detach().cpu().numpy()
            actions = actions[: min(2, len(actions))]
            chunk, done, n = self._execute_tokens(
                actions, limit=self.budget - elapsed
            )
            continuation.extend(chunk)
            elapsed += n
            success |= done
        return {
            "success": bool(success),
            "elapsed": elapsed,
            "end_state": np.asarray(self.env.unwrapped._get_obs(), np.float64).copy(),
            "continuation_raw": np.asarray(continuation, np.float32).reshape(-1, 2),
        }

    def branch_and_continue(
        self, initial_state, goal_state, entry, candidate_standard, goal_pixels
    ):
        raw_candidate = raw_actions_from_tokens(
            candidate_standard, self.mean, self.std
        )
        anchor = restore_eval_anchor(
            self.env,
            initial_state=initial_state,
            goal_state=goal_state,
            prefix_raw_actions=entry["prefix_raw"],
        )
        fingerprint, rgb0 = anchor[1], anchor[2]
        ever_success, executed = False, 0
        obs = None
        for raw_action in np.asarray(raw_candidate, np.float32).reshape(-1, 2):
            obs, _, success, truncated, _ = self.env.step(raw_action)
            executed += 1
            ever_success = ever_success or bool(success)
            if success or truncated:
                break
        elapsed = entry["step"] + executed
        if ever_success:
            outcome = {"success": True, "elapsed": elapsed,
                       "end_state": np.asarray(obs["state"], np.float64).copy()}
        elif elapsed < self.budget:
            outcome = self.continue_baseline(
                elapsed, entry["rng_after_solve"], goal_pixels
            )
        else:
            outcome = {"success": False, "elapsed": elapsed,
                       "end_state": np.asarray(obs["state"], np.float64).copy()}
        continuation = outcome.get("continuation_raw", np.empty((0, 2), np.float32))
        outcome["anchor_fingerprint"] = fingerprint
        outcome["anchor_rgb"] = rgb0
        outcome["raw_executed"] = np.concatenate(
            [raw_candidate[:executed], continuation], axis=0
        )
        outcome["score"] = score_continuation(
            outcome["end_state"], goal_state, outcome["success"]
        )
        return outcome

    # -- one anchor --------------------------------------------------------- #
    def collect_anchor(
        self,
        task: TaskSpec,
        entry: dict,
        trace: dict,
        initial_state,
        goal_state,
        goal_pixels,
        *,
        anchor_id: int,
        rng: np.random.Generator,
    ) -> dict:
        """Collect A* + antithetic candidates for ONE planning step.

        Never raises on a replay inconsistency: returns a record with
        `valid=False`, `invalid_reason` and `first_divergence_step`.
        """
        a0 = np.asarray(entry["actions_standard"])
        base = {
            "task_uid": task.task_uid,
            "source_episode_uid": task.source_episode_uid,
            "split": task.split,
            "anchor_id": int(anchor_id),
            "seed_idx": int(task.seed_idx),
            "planner_seed_base": int(self.seed),
            "steps_taken": int(entry["step"]),
            "planning_horizon": int(len(a0)),
            "baseline_success": bool(trace["success"]),
            "baseline_steps": int(trace["elapsed"]),
            "goal_offset_steps": int(task.goal_offset_steps),
            "epsilon": float(self.epsilon),
            "directions": int(self.directions),
            "pixels_sha1": entry["pixels_sha1"],
            "goal_sha1": entry["goal_sha1"],
            "ckpt": str(self.ckpt),
            "task_meta": dict(task.meta),
        }
        if a0.shape != (2, 10):
            base.update(valid=False,
                        invalid_reason=f"unsupported plan shape {a0.shape}")
            return base

        cands = antithetic_candidates(a0, self.epsilon, self.directions, rng)
        results = [
            self.branch_and_continue(
                initial_state, goal_state, entry, cand, goal_pixels
            )
            for cand in cands
        ]

        # --- candidate-to-candidate anchor equality (branch-order sanity) ---
        for r in results[1:]:
            if not np.allclose(
                results[0]["anchor_fingerprint"], r["anchor_fingerprint"],
                rtol=0, atol=1e-7,
            ) or not np.array_equal(results[0]["anchor_rgb"], r["anchor_rgb"]):
                base.update(valid=False,
                            invalid_reason="counterfactual anchors differ")
                return base

        # --- STRICT replay gate for the original-action (A*) branch ---------
        replay_fp_maxdiff = float(np.abs(
            np.asarray(results[0]["anchor_fingerprint"], np.float64)
            - np.asarray(entry["real_anchor_fingerprint"], np.float64)
        ).max())
        remaining = trace["full_raw_actions"][entry["step"]:]
        div = first_divergence(results[0]["raw_executed"], remaining, atol=1e-6)
        rgb_equal = np.array_equal(results[0]["anchor_rgb"], entry["real_anchor_rgb"])
        success_equal = bool(results[0]["success"]) == bool(trace["success"])

        base.update(
            replay_fp_maxdiff=replay_fp_maxdiff,
            replay_rgb_equal=bool(rgb_equal),
            first_divergence_step=int(div),
            replay_success_equal=bool(success_equal),
            replay_executed_steps=int(len(results[0]["raw_executed"])),
            replay_expected_steps=int(len(remaining)),
        )
        if div != -1 or not success_equal or not rgb_equal:
            base.update(
                valid=False,
                invalid_reason=(
                    "replay divergence: "
                    + (f"actions differ at step {div}; " if div != -1 else "")
                    + ("" if rgb_equal else "anchor RGB differs; ")
                    + ("" if success_equal else
                       f"A* success {results[0]['success']} != baseline {trace['success']}")
                ).strip(),
            )
            return base

        costs = np.asarray([r["score"] for r in results], np.float64)
        l1 = entry["level1_info"]
        z0 = l1["embed_0"][0, -1].detach().cpu().numpy().astype(np.float32)
        goal_emb = l1["goal_embed_0"][0, -1].detach().cpu().numpy().astype(np.float32)
        upper = bool(l1.get("goal_embed_0_in_upper_space", False))
        pairs = []
        for d in range(self.directions):
            i, j = 1 + 2 * d, 2 + 2 * d
            pairs.append({
                "plus": np.asarray(cands[i], np.float32).tolist(),
                "minus": np.asarray(cands[j], np.float32).tolist(),
                "plus_cost": float(costs[i]),
                "minus_cost": float(costs[j]),
            })
        base.update(
            valid=True,
            invalid_reason="",
            z0=z0.tolist(),
            goal_emb=goal_emb.tolist(),
            goal_is_upper=upper,
            pairs=pairs,
            astar_cost=float(costs[0]),
            candidate_costs=[float(c) for c in costs[1:]],
            candidate_successes=[bool(r["success"]) for r in results[1:]],
            astar_success=bool(results[0]["success"]),
            realized_cost_diff=[float(c - costs[0]) for c in costs[1:]],
            candidate_elapsed=[int(r["elapsed"]) for r in results[1:]],
            astar_elapsed=int(results[0]["elapsed"]),
            recoverable=(not trace["success"]) and any(
                bool(r["success"]) for r in results[1:]
            ),
            fragile=bool(trace["success"]) and not all(
                bool(r["success"]) for r in results[1:]
            ),
        )
        return base

    # -- one task ----------------------------------------------------------- #
    def collect_task(
        self,
        task: TaskSpec,
        *,
        max_decisions: int = 1,
        anchor_id_base: int = 0,
        write: bool = True,
    ) -> dict:
        ep = task.payload
        states = as_numpy_state(ep)
        initial_state = states[0].astype(np.float64)
        goal_state = states[-1].astype(np.float64)
        pixels = ep["pixels"]
        pixels = (
            pixels.detach().cpu().numpy()
            if torch.is_tensor(pixels)
            else np.asarray(pixels)
        )
        ep_pixels = np.transpose(pixels, (0, 2, 3, 1))  # CHW -> HWC
        goal_pixels = np.asarray(ep_pixels[-1], dtype=np.uint8)

        mode = task.meta.get("seed_mode", "chunk")
        if mode == "chunk":
            self._reseed(task.seed_idx)
        trace = self.trace_baseline(initial_state, goal_state, ep_pixels)
        if self.trace_hook is not None:
            self.trace_hook(task, trace)
        self._log(
            f"[{task.task_uid}] baseline success={trace['success']} "
            f"steps={trace['elapsed']}"
        )
        summary = {
            "kind": "task",
            "task_uid": task.task_uid,
            "source_episode_uid": task.source_episode_uid,
            "split": task.split,
            "seed_idx": int(task.seed_idx),
            "baseline_success": bool(trace["success"]),
            "baseline_steps": int(trace["elapsed"]),
            "n_decisions": int(min(len(trace["path"]), max_decisions)),
            "goal_offset_steps": int(task.goal_offset_steps),
        }
        if write:
            append_jsonl(self.jsonl_path, summary)

        rng = np.random.default_rng(self.seed + int(task.seed_idx))
        records = []
        for k, entry in enumerate(trace["path"][:max_decisions]):
            rec = self.collect_anchor(
                task, entry, trace, initial_state, goal_state, goal_pixels,
                anchor_id=anchor_id_base + k, rng=rng,
            )
            rec["kind"] = "anchor"
            records.append(rec)
            if write:
                append_jsonl(self.jsonl_path, rec)
            if rec.get("valid"):
                self._log(
                    f"  anchor {rec['anchor_id']} step {rec['steps_taken']}: "
                    f"cand_success={rec['candidate_successes']} "
                    f"dCost={[round(x, 4) for x in rec['realized_cost_diff']]}"
                )
            else:
                self._log(
                    f"  anchor {rec['anchor_id']} step {rec['steps_taken']}: "
                    f"INVALID — {rec['invalid_reason']}"
                )
        return {"summary": summary, "anchors": records}
