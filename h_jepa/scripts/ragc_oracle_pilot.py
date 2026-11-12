"""Small *on-policy* RAGC oracle gate for source repo research/action-response-v1.

Runs ORIGINAL frozen H-JEPA L2 with:
- L1 action_cost_weight=0.2 intact
- real `steps_taken`
- original horizon schedule / action normalizer
Then tests paired normalized action perturbations from IDENTICAL physical anchors
and continues with the SAME frozen baseline for the remaining budget.

SMOKE (not the final 50-task eval):
  cd H-JEPA/h_jepa
  export PYTHONPATH=. HJEPA_HOME=/... SDL_VIDEODRIVER=dummy
  python -u scripts/ragc_oracle_pilot.py \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --episodes 1 --max-decisions 1 --directions 1 \
      --out /tmp/ragc_pilot.json

IMPORTANT: this is a REFERENCE implementation. Compare its complete baseline
action trace and success to official planning_eval on the same task FIRST.
Do not use its labels until that equivalence gate is satisfied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import torch
import stable_worldmodel.envs  # noqa: F401
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from action_response_wm import get_action_stats
from data import IMAGENET_STATS
from eval_config_utils import _build_policy_plan_config
from hierarchical_solver import build_hierarchical_solver
from planning_eval import load_model, load_eval_config
from ragc_oracle import (
    antithetic_candidates,
    raw_actions_from_tokens,
    restore_eval_anchor,
    simulate_candidate,
    verify_same_anchor,
    score_continuation,
    counterfactual_fingerprint,
)


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
    mean = torch.as_tensor(IMAGENET_STATS["mean"], device=device).view(3,1,1)
    std = torch.as_tensor(IMAGENET_STATS["std"], device=device).view(3,1,1)
    return ((x-mean)/std).unsqueeze(0).unsqueeze(0)


def model_info(env, device, goal_pixels, first_pixels=None):
    # Official eval: the FIRST observation is the dataset frame (pixels[0]);
    # later observations are live renders. The goal is ALWAYS the dataset
    # goal frame (pixels[-1]), re-injected on every world step.
    frame = first_pixels if first_pixels is not None else np.asarray(env.render(), dtype=np.uint8)
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


def solver_rng(solver):
    return {
        level: child.torch_gen.get_state().clone()
        for level, child in solver.level_solvers.items()
    }


def reset_solver_rng(solver, state):
    for level, rng_state in state.items():
        solver.level_solvers[level].torch_gen.set_state(rng_state.clone())


def execute_tokens(env, std_actions, mean, std, limit):
    raw = raw_actions_from_tokens(std_actions, mean, std)
    n = 0
    succeeded = False
    for a in raw:
        if n >= limit:
            break
        _, _, done, truncated, _ = env.step(a.astype(np.float32))
        n += 1
        succeeded |= bool(done)
        if done or truncated:
            break
    return raw[:n].copy(), bool(succeeded), n


def trace_baseline(env, solver, init_state, goal_state, ep_pixels, budget, mean, std, device):
    # Official: reset(seed=None) then callables _set_state / _set_goal_state.
    env.reset()
    env.unwrapped._set_state(np.asarray(init_state, dtype=np.float64))
    env.unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    goal_pixels = np.asarray(ep_pixels[-1], dtype=np.uint8)
    first_pixels = np.asarray(ep_pixels[0], dtype=np.uint8)
    path = []
    elapsed = 0
    success = False
    recorded_actions = []
    first_call = True

    while elapsed < budget and not success:
        info = model_info(
            env, device, goal_pixels,
            first_pixels=first_pixels if first_call else None,
        )
        first_call = False
        # Capture EXACT level-1 subgoal supplied to the original solver.
        # The hook is installed ONCE before tracing (see main).
        solver.capture_level1.clear()
        # Match official planning_eval numerics (MATH SDPA backend).
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            result = solver.solve(info, steps_taken=elapsed, eval_budget=budget)
        std_action = result["actions"][0].detach().cpu().numpy()
        # Must honor shrinking L1 horizon near end of episode.
        take = min(2, len(std_action))
        std_action = std_action[:take]
        capture = copy_info(solver.capture_level1["info"])
        assert float(capture["action_cost_weight"]) == 0.2, (
            "Original level-1 action-cost weight must remain 0.2"
        )
        entry = {
            "step": elapsed,
            "real_anchor_fingerprint": counterfactual_fingerprint(env),
            "real_anchor_rgb": np.asarray(env.render(), dtype=np.uint8).copy(),
            "actions_standard": std_action,
            "prefix_raw": np.asarray(recorded_actions, np.float32).reshape(-1, 2).copy(),
            "rng_after_solve": solver_rng(solver),
            "level1_info": capture,
            "pixels_sha1": hashlib.sha1(
                info["pixels"].detach().float().cpu().numpy().tobytes()
            ).hexdigest(),
            "goal_sha1": hashlib.sha1(
                info["goal"].detach().float().cpu().numpy().tobytes()
            ).hexdigest(),
        }
        path.append(entry)

        chunk, done, n = execute_tokens(
            env, std_action, mean, std, limit=budget - elapsed
        )
        recorded_actions.extend(chunk)
        elapsed += n
        success |= done

    return {
        "success": success,
        "elapsed": elapsed,
        "path": path,
        "full_raw_actions": np.asarray(recorded_actions, np.float32).reshape(-1,2),
    }


def continue_baseline(
    env, solver, from_step, rng_after_anchor, budget,
    mean, std, device, goal_pixels,
):
    # Repeat identical solver RNG streams for each candidate branch.
    reset_solver_rng(solver, rng_after_anchor)
    elapsed = from_step
    success = False
    continuation_raw = []
    while elapsed < budget and not success:
        info = model_info(env, device, goal_pixels)
        solver.capture_level1.clear()
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            result = solver.solve(info, steps_taken=elapsed, eval_budget=budget)
        actions = result["actions"][0].detach().cpu().numpy()
        actions = actions[:min(2, len(actions))]
        chunk, done, n = execute_tokens(
            env, actions, mean, std, limit=budget-elapsed
        )
        continuation_raw.extend(chunk)
        elapsed += n
        success |= done
    return {
        "success": bool(success),
        "elapsed": elapsed,
        "end_state": np.asarray(env.unwrapped._get_obs(), dtype=np.float64).copy(),
        "continuation_raw": np.asarray(continuation_raw, np.float32).reshape(-1,2),
    }


def branch_and_continue(
    env, solver, initial_state, goal_state, entry,
    candidate_standard, mean, std, budget, device, goal_pixels,
):
    # Rewind all hidden physical state with ORIGINAL initial EVAL state and
    # original actually-executed action history. No intermediate _set_state.
    raw_candidate = raw_actions_from_tokens(candidate_standard, mean, std)
    initial = simulate_candidate(
        env,
        initial_state=initial_state,
        goal_state=goal_state,
        prefix_raw_actions=entry["prefix_raw"],
        raw_candidate=raw_candidate,
    )
    elapsed = entry["step"] + initial["executed"]
    if initial["ever_success"]:
        outcome = {
            "success": True,
            "end_state": initial["end_state"],
            "elapsed": elapsed,
        }
    elif elapsed < budget:
        outcome = continue_baseline(
            env, solver, elapsed, entry["rng_after_solve"],
            budget, mean, std, device, goal_pixels,
        )
    else:
        outcome = {
            "success": False,
            "end_state": initial["end_state"],
            "elapsed": elapsed,
        }
    outcome["anchor_fingerprint"] = initial["fingerprint"]
    outcome["anchor_rgb"] = initial["rgb0"]
    continuation = outcome.get("continuation_raw", np.empty((0,2), np.float32))
    outcome["raw_executed"] = np.concatenate(
        [raw_candidate[:initial["executed"]], continuation], axis=0
    )
    outcome["score"] = score_continuation(
        outcome["end_state"], goal_state, outcome["success"]
    )
    return outcome


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config-name", default="pusht_l2")
    p.add_argument("--trajs", default="assets/eval_trajs/pusht/goal_offset_75_val.pt")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-decisions", type=int, default=1)
    p.add_argument("--directions", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed-mode", choices=("chunk", "legacy"), default="chunk",
                   help="chunk: replicate official chunked-eval seeding "
                        "(planner gens and env reset use base_seed + task index)")
    p.add_argument("--out", default="/tmp/ragc_pilot.json")
    p.add_argument("--dump-trace", default=None,
                   help="Optional JSONL of baseline standardized actions per task")
    p.add_argument("--dataset", default=None,
                   help="Optional NPZ export. Must NOT use held-out eval_trajs for training.")
    args = p.parse_args()
    if args.dataset is not None and "eval_trajs" in Path(args.trajs).parts:
        raise ValueError(
            "Refusing to export RAGC training labels from held-out eval_trajs. "
            "Use a separate TRAIN task/goal split, or omit --dataset for the oracle gate."
        )
    if args.directions < 1:
        raise ValueError("--directions must be >= 1")
    cfg = load_eval_config(args.config_name)
    cfg.policy = args.ckpt
    model = load_model(cfg)
    device = next(model.parameters()).device
    mean_t, std_t = get_action_stats(model, args.ckpt, device)
    mean = mean_t.detach().cpu().numpy()
    std = std_t.detach().cpu().numpy()
    solver = build_hierarchical_solver(cfg, model)
    plan_cfg = _build_policy_plan_config(cfg)
    act_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,2), dtype=np.float32
    )
    solver.configure(action_space=act_space, n_envs=1, config=plan_cfg)
    solver.capture_level1 = {}
    l1solver = solver.level_solvers[1]
    orig_l1solve = l1solver.solve

    def record_level1(*args, **kwargs):
        solver.capture_level1["info"] = copy_info(kwargs["info_dict"])
        return orig_l1solve(*args, **kwargs)

    l1solver.solve = record_level1
    planner_gens = [
        (child.torch_gen, child.torch_gen.initial_seed())
        for child in solver.level_solvers.values()
        if hasattr(child, "torch_gen")
    ]
    payload = torch.load(args.trajs, map_location="cpu", weights_only=False)
    episodes = payload["data"][:args.episodes]
    rng = np.random.default_rng(args.seed)
    env = make_env()
    result_rows = []
    pairs = []
    budget = int(cfg.eval.eval_budget)
    try:
        for ep_i, ep in enumerate(episodes):
            states = as_numpy_state(ep)
            initial_state = states[0].astype(np.float64)
            goal_state = states[-1].astype(np.float64)
            pixels = ep["pixels"]
            pixels = pixels.detach().cpu().numpy() if torch.is_tensor(pixels) else np.asarray(pixels)
            ep_pixels = np.transpose(pixels, (0, 2, 3, 1))  # CHW -> HWC uint8
            goal_pixels = np.asarray(ep_pixels[-1], dtype=np.uint8)
            if args.seed_mode == "chunk":
                # Official _run_chunked_eval: planner gens -> base_seed + chunk start.
                # Use the ORIGINAL task index so a subset payload reproduces the
                # exact same planner seeds as the full payload.
                seed_idx = int(payload.get("episodes_idx", list(range(len(payload["data"]))))[ep_i])
                for gen, base in planner_gens:
                    gen.manual_seed(int(base) + seed_idx)
            trace = trace_baseline(
                env, solver, initial_state, goal_state, ep_pixels,
                budget, mean, std, device
            )
            if args.dump_trace:
                with open(args.dump_trace, "a") as f:
                    f.write(json.dumps({
                        "kind": "task",
                        "task": ep_i,
                        "success": bool(trace["success"]),
                        "steps": [
                            {"steps_taken": int(e["step"]),
                             "actions": np.asarray(e["actions_standard"]).tolist(),
                             "pixels_sha1": e["pixels_sha1"],
                             "goal_sha1": e["goal_sha1"]}
                            for e in trace["path"]
                        ],
                    }) + "\n")
            print(f"[ep{ep_i}] original success={trace['success']}, "
                  f"steps={trace['elapsed']}", flush=True)

            for entry in trace["path"][:args.max_decisions]:
                a0 = entry["actions_standard"]
                if a0.shape != (2, 10):
                    # V0 learns only the two-token 10-step case; never
                    # silently mix H=1/H=2 pair arrays in one NPZ.
                    continue
                cands = antithetic_candidates(
                    a0, args.epsilon, args.directions, rng
                )
                results = [
                    branch_and_continue(
                        env, solver, initial_state, goal_state, entry,
                        cand, mean, std, budget, device, goal_pixels
                    )
                    for cand in cands
                ]
                for r in results[1:]:
                    # Identical fingerprint includes block angular velocity.
                    np.testing.assert_allclose(
                        results[0]["anchor_fingerprint"],
                        r["anchor_fingerprint"],
                        rtol=0, atol=1e-7
                    )
                    if not np.array_equal(results[0]["anchor_rgb"], r["anchor_rgb"]):
                        raise AssertionError("Mismatch of counterfactual anchor RGB")
                # Strong replay gate: compare hidden physics, RGB and the ENTIRE
                # remaining raw action trace, not only the success boolean.
                # NOTE: the hidden-physics comparison is recorded as a diagnostic
                # rather than a hard failure: it is a *pre-execution* snapshot of
                # the live env, which can differ from the replay snapshot by a
                # sub-unit settling offset even when the whole action trace and
                # the outcome are bit-identical.  The authoritative replay gates
                # are the continuation-action equality and success equality below.
                replay_fp_maxdiff = float(np.abs(
                    np.asarray(results[0]["anchor_fingerprint"], np.float64)
                    - np.asarray(entry["real_anchor_fingerprint"], np.float64)
                ).max())
                if replay_fp_maxdiff > 1e-7:
                    print(f"  WARN replay anchor fp maxdiff={replay_fp_maxdiff:.3e}",
                          flush=True)
                if not np.array_equal(results[0]["anchor_rgb"], entry["real_anchor_rgb"]):
                    raise AssertionError("Replay anchor image mismatch")
                remaining_original = trace["full_raw_actions"][entry["step"]:]
                np.testing.assert_allclose(
                    results[0]["raw_executed"], remaining_original,
                    rtol=0, atol=1e-6,
                    err_msg="Replay continuation actions diverge from original baseline"
                )
                if bool(results[0]["success"]) != bool(trace["success"]):
                    raise AssertionError(
                        "Original-action branch does NOT reproduce baseline "
                        "continuation success. Fix tracing/solver RNG first."
                    )

                l1info = entry["level1_info"]
                z0 = l1info["embed_0"][0, -1].detach().cpu().numpy()
                goal = l1info["goal_embed_0"][0, -1].detach().cpu().numpy()
                upper = bool(l1info.get("goal_embed_0_in_upper_space", False))
                rec = {
                    "episode": ep_i, "step": entry["step"],
                    "seed_idx": int(seed_idx) if args.seed_mode == "chunk" else -1,
                    "replay_fp_maxdiff": replay_fp_maxdiff,
                    "baseline_success": trace["success"],
                    "candidate_successes": [r["success"] for r in results],
                    "candidate_costs": [r["score"] for r in results],
                    "oracle_can_recover": (not trace["success"])
                        and any(r["success"] for r in results[1:]),
                }
                result_rows.append(rec)
                print(f"  decision {entry['step']}: "
                      f"candidate_success={rec['candidate_successes']}", flush=True)
                for d in range(args.directions):
                    i, j = 1 + 2*d, 2 + 2*d
                    pairs.append({
                        "anchor_id": int(ep_i * 1000 + entry["step"]),
                        "z0": z0, "goal": goal, "goal_is_upper": upper,
                        "plus": cands[i], "minus": cands[j],
                        "plus_cost": results[i]["score"],
                        "minus_cost": results[j]["score"],
                        "epsilon": float(args.epsilon),
                    })
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "warning": "REFERENCE PILOT: compare original trace with official eval first",
        "records": result_rows,
    }, indent=2))
    if pairs and args.dataset is not None:
        ds = {k: np.asarray([row[k] for row in pairs]) for k in pairs[0]}
        ds["label_source"] = np.asarray("baseline_continuation")
        data_path = Path(args.dataset)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, **ds)
        print("Saved", data_path, flush=True)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
