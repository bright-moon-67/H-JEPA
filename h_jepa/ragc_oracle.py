"""Strict SAME-STATE counterfactual Push-T rollout helpers for RAGC data collection.

This module does NOT perform planner evaluation on its own. A planner-driven
collector must provide the actual baseline episode prefix + its goal.
Do not reset to an intermediate 7-D state; it excludes hidden block velocities.
"""
from __future__ import annotations
import numpy as np


def raw_actions_from_tokens(tokens, mean, std):
    """Standardized [H,10] level-1 tokens -> [5H,2] REAL env actions."""
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[-1] != 10:
        raise ValueError(f"Expected [H,10], got {tokens.shape}")
    mean = np.asarray(mean, np.float32).reshape(1, 2)
    std = np.asarray(std, np.float32).reshape(1, 2)
    if np.any(std <= 0):
        raise ValueError("Invalid action std")
    return tokens.reshape(-1, 2) * std + mean


def antithetic_candidates(a_star, eps, directions, rng):
    """Returns [A*, A*±eps*u_1, ...] in SOLVER standardized space.

    Candidates are NOT automatically clipped because the original WorldModelPolicy
    does not clip after inverse_transform. Report invalid raw actions explicitly.
    """
    base = np.asarray(a_star, np.float32)
    if base.ndim != 2 or base.shape[-1] != 10:
        raise ValueError(f"Expected [H,10], got {base.shape}")
    out = [base.copy()]
    for _ in range(int(directions)):
        u = rng.normal(size=base.shape).astype(np.float32)
        u /= max(float(np.linalg.norm(u)), 1e-12)
        out.extend([base + float(eps) * u, base - float(eps) * u])
    return out


def counterfactual_fingerprint(env):
    """Include hidden Pymunk block linear/angular velocity in audit."""
    p = env.unwrapped
    obs = np.asarray(p._get_obs(), dtype=np.float64).reshape(-1)
    vel = np.asarray(list(p.block.velocity), dtype=np.float64)
    w = float(p.block.angular_velocity)
    return np.r_[obs, vel, w]


def clear_residual_motion(env):
    """Zero velocities that `_set_state` does NOT restore.

    `PushtEnv._set_state` writes only agent/block position and block angle, then
    calls `space.step(dt)` once. Pymunk block linear/angular velocity therefore
    carries over from whatever ran BEFORE the restore, which makes a
    counterfactual anchor depend on branch order (measured drift ~0.4 units).
    Official eval never audits this because it only runs one branch per env.
    """
    p = env.unwrapped
    for obj in (getattr(p, "block", None), getattr(p, "agent", None)):
        if obj is None:
            continue
        obj.velocity = (0.0, 0.0)
        obj.angular_velocity = 0.0


def restore_eval_anchor(env, *, initial_state, goal_state, prefix_raw_actions):
    """Recreate the OFFICIAL eval setup, then replay its ACTUAL actions.

    Official `World.evaluate_from_dataset` resets with seed=None and empty
    options, then applies the eval callables `_set_state(state0)` and
    `_set_goal_state(goal_state)`. It does NOT pass options={'state': ...} to
    reset (that would add an extra hidden physics step from the sampled state).

    Pymunk keeps contact/impulse warm-start state inside `space`, and
    `_set_state` restores pos/angle but not block velocity, so reusing the env
    across counterfactual branches makes the anchor depend on branch ORDER
    (measured drift up to 1.4 units). A brand-new `space` per restore removes
    that history and makes every branch start from a bit-identical anchor.
    """
    env.reset()
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "_setup"):
        unwrapped._setup()
    clear_residual_motion(env)
    unwrapped._set_state(np.asarray(initial_state, dtype=np.float64))
    unwrapped._set_goal_state(np.asarray(goal_state, dtype=np.float64))
    for action in np.asarray(prefix_raw_actions, dtype=np.float32).reshape(-1, 2):
        env.step(action)
    return None, counterfactual_fingerprint(env), np.asarray(env.render()).copy()


def simulate_candidate(
    env, *, initial_state, goal_state, prefix_raw_actions, raw_candidate,
):
    obs, before, rgb_before = restore_eval_anchor(
        env,
        initial_state=initial_state,
        goal_state=goal_state,
        prefix_raw_actions=prefix_raw_actions,
    )
    ever_success = False
    executed = 0
    for raw_action in np.asarray(raw_candidate, dtype=np.float32).reshape(-1, 2):
        obs, _, success, truncated, _ = env.step(raw_action)
        executed += 1
        ever_success = ever_success or bool(success)
        if success or truncated:
            break
    return {
        "fingerprint": before,
        "rgb0": rgb_before,
        "end_state": np.asarray(obs["state"], dtype=np.float64).copy(),
        "ever_success": ever_success,
        "executed": executed,
    }


def verify_same_anchor(a, b, atol=1e-7):
    """Fail closed: same 7-D state is insufficient if hidden velocities differ."""
    np.testing.assert_allclose(a["fingerprint"], b["fingerprint"], rtol=0, atol=atol)
    if not np.array_equal(a["rgb0"], b["rgb0"]):
        raise AssertionError("Counterfactual anchor image mismatch")


def terminal_task_cost(end_state, goal_state, success):
    """Push-T training-label *fallback* only, NOT full-episode realized advantage.

    The actual eval success uses agent+block XY (<20) AND angle (<20 deg).
    For RAGC, prefer a CONTINUATION label obtained after applying a candidate
    then following the unchanged baseline policy until the episode budget ends.
    """
    s = np.asarray(end_state, np.float64).reshape(7)
    g = np.asarray(goal_state, np.float64).reshape(7)
    position_err = float(np.linalg.norm(s[:4] - g[:4]))
    angle_err = float(abs(np.arctan2(np.sin(s[4] - g[4]), np.cos(s[4] - g[4]))))
    # Unitless, bounded shaping; lower is better.
    return (
        (0.0 if bool(success) else 1.0)
        + 0.05 * min(position_err / 20.0, 10.0)
        + 0.05 * min(angle_err / (np.pi / 9.0), 10.0)
    )


def score_continuation(final_state, goal_state, ever_success):
    """Same metric as terminal_task_cost but after frozen-baseline continuation."""
    return terminal_task_cost(final_state, goal_state, ever_success)
