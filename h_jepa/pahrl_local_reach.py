"""Controlled local H-JEPA reachability trials for PAHRL Stage 1.

This module never runs inside the online planner path.  It builds the frozen
L3 policy exactly like ``planning_eval`` does, restores memorized TRAIN
physical states into the real ``MegaWrapper -> SyncWorld -> VariationWrapper``
chain, and drives the *original* receding-horizon planner towards a memorized
target observation.

Success is measured PHYSICALLY (the environment's own goal test, i.e. the ant
ending within ``goal_tol`` of the target xy).  Latent cosine distances are
recorded alongside, but they never decide success here; the arrival threshold
is calibrated from these trials separately.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from planning_eval import (
    _build_eval_process_from_policy_normalizer,
    _build_policy,
    img_transform,
    load_model,
)
from stable_worldmodel.world import _find_env_method

DEFAULT_CONFIG = Path("config/eval/ant_l3.yaml")


def build_eval_stack(checkpoint: Path, config: Path = DEFAULT_CONFIG):
    """Load the frozen checkpoint, normalizer processors and L3 policy."""
    cfg = OmegaConf.load(config)
    cfg.policy = str(Path(checkpoint).resolve())
    model = load_model(cfg)
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    policy = _build_policy(cfg, process, transform, model=model)
    if len(policy.solver.level_models) != 3:
        raise ValueError("Only H-JEPA L3 checkpoints are supported")
    return cfg, policy


def build_world(cfg, num_envs: int):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    cfg.world.num_envs = int(num_envs)
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    image_shape = (int(cfg.eval.img_size),) * 2
    return swm.World(**world_cfg, image_shape=image_shape)


def proprio_from_qpos_qvel(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(qpos)[2:], np.asarray(qvel)]).astype(
        np.float32
    )


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _env_data(env):
    owner = env
    while owner is not None and not hasattr(owner, "data"):
        nxt = getattr(owner, "env", None)
        if nxt is owner:
            break
        owner = nxt
    return getattr(owner, "data", None)


class LocalReachRunner:
    """Batched frozen-H-JEPA local trials with physical success labels."""

    def __init__(
        self,
        cfg,
        policy,
        *,
        max_steps: int = 60,
        reach_cosine_distance: float = 0.035,
        xy_tolerance: float = 1.0,
        base_seed: int = 90000,
    ):
        self.cfg = cfg
        self.policy = policy
        self.max_steps = int(max_steps)
        self.reach_cosine_distance = float(reach_cosine_distance)
        # Default equals the AntMaze task goal tolerance (``_goal_tol``), i.e.
        # the environment's own success test.  Not tuned on any eval outcome.
        self.xy_tolerance = float(xy_tolerance)
        self.base_seed = int(base_seed)
        self.world = None

    # ---------------------------------------------------------------- helpers
    def _enumerate_infos(self, pixels, proprio, goals, goal_proprios):
        return {
            "pixels": np.stack(pixels)[:, None],
            "proprio": np.stack(proprio)[:, None],
            "goal": np.stack(goals)[:, None],
            "goal_proprio": np.stack(goal_proprios)[:, None],
        }

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / max(float(np.linalg.norm(x)), 1e-8)

    def run(self, trials: list[dict]) -> list[dict]:
        """Run every trial (one env per trial, batched by ``num_envs``)."""
        results: list[dict] = []
        n = len(trials)
        if n == 0:
            return results
        self.world = build_world(self.cfg, n)
        try:
            self.policy.reset()
            self.world.set_policy(self.policy)
            seeds = [self.base_seed + i for i in range(n)]
            for i, env in enumerate(self.world.envs.unwrapped.envs):
                env.action_space.seed(int(seeds[i]))
            self.world.reset(seed=seeds, options=[{} for _ in range(n)])
            # evaluate_from_dataset clears this every step; a terminated task
            # must not trigger gymnasium's autoreset accounting.
            self.world.envs.unwrapped._autoreset_envs = np.zeros(n)

            pixels, proprios, goals, goal_proprios = [], [], [], []
            replay_ok = []
            for i, env in enumerate(self.world.envs.unwrapped.envs):
                trial = trials[i]
                set_state = _find_env_method(env, "set_state")
                if set_state is None:
                    raise RuntimeError("Environment lacks set_state")
                set_state(
                    qpos=np.asarray(trial["source_qpos"], dtype=np.float64),
                    qvel=np.asarray(trial["source_qvel"], dtype=np.float64),
                )
                data = _env_data(env)
                got_qpos = np.asarray(data.qpos).copy()
                got_qvel = np.asarray(data.qvel).copy()
                replay = bool(
                    np.array_equal(
                        got_qpos, np.asarray(trial["source_qpos"], dtype=np.float64)
                    )
                    and np.array_equal(
                        got_qvel, np.asarray(trial["source_qvel"], dtype=np.float64)
                    )
                )
                replay_ok.append(replay)

                set_nav = _find_env_method(env, "set_navigation_goal")
                if set_nav is None:
                    raise RuntimeError("Environment lacks set_navigation_goal")
                set_nav(goal_xy=np.asarray(trial["target_xy"], dtype=np.float64))

                get_ob = _find_env_method(env, "get_ob")
                if get_ob is None:
                    raise RuntimeError("Environment lacks get_ob")
                pixels.append(np.asarray(get_ob()))
                proprios.append(
                    proprio_from_qpos_qvel(got_qpos, got_qvel)
                )
                goals.append(np.asarray(trial["target_pixels"]))
                goal_proprios.append(
                    proprio_from_qpos_qvel(
                        trial["target_qpos"], trial["target_qvel"]
                    )
                )

            infos = self._enumerate_infos(pixels, proprios, goals, goal_proprios)
            target_z = [self._unit(t["target_embedding"]) for t in trials]
            done = np.zeros(n, dtype=bool)
            arrival_step = np.full(n, -1, dtype=np.int64)
            min_cos = np.full(n, np.inf, dtype=np.float64)
            min_xy = np.full(n, np.inf, dtype=np.float64)
            terminated_any = np.zeros(n, dtype=bool)
            step = 0
            start = time.time()

            while step < self.max_steps:
                # The original planner replans only when its action buffer is
                # empty (receding_horizon * action_block env steps).
                plan_now = len(self.policy._action_buffer) == 0
                if plan_now:
                    prepared = self.policy._prepare_info(dict(infos))
                    with torch.inference_mode():
                        cur = self.policy.solver._encode_all_levels(
                            prepared, key="pixels"
                        )["embed_3"][:, 0].float().cpu().numpy()
                    if step > 0:
                        for i in range(n):
                            if done[i]:
                                continue
                            dist = float(
                                max(
                                    0.0,
                                    1.0
                                    - float(target_z[i] @ self._unit(cur[i])),
                                )
                            )
                            min_cos[i] = min(min_cos[i], dist)
                            get_xy = _find_env_method(
                                self.world.envs.unwrapped.envs[i], "get_xy"
                            )
                            xy = np.asarray(get_xy())
                            d_xy = float(
                                np.linalg.norm(
                                    xy
                                    - np.asarray(trials[i]["target_xy"], float)
                                )
                            )
                            min_xy[i] = min(min_xy[i], d_xy)
                            if terminated_any[i] and arrival_step[i] < 0:
                                arrival_step[i] = step
                            if (
                                d_xy <= self.xy_tolerance
                                and arrival_step[i] < 0
                            ):
                                arrival_step[i] = step
                            if arrival_step[i] >= 0:
                                done[i] = True
                    if done.all():
                        break

                actions = self.policy.get_action(infos)
                _, _, term, trunc, env_infos = self.world.envs.step(actions)
                self.world.envs.unwrapped._autoreset_envs = np.zeros(n)
                terminated_any |= np.asarray(term).reshape(-1).astype(bool)
                pixels = env_infos["pixels"]
                proprios = env_infos["proprio"]
                # Goal stream stays fixed for the whole trial.
                infos = self._enumerate_infos(
                    [pixels[i, -1] for i in range(n)],
                    [proprios[i, -1] for i in range(n)],
                    goals,
                    goal_proprios,
                )
                step += 1

            # Final physical state check after the last executed step.
            for i, env in enumerate(self.world.envs.unwrapped.envs):
                get_xy = _find_env_method(env, "get_xy")
                xy = np.asarray(get_xy())
                d_xy = float(
                    np.linalg.norm(xy - np.asarray(trials[i]["target_xy"], float))
                )
                min_xy[i] = min(min_xy[i], d_xy)
                if arrival_step[i] < 0 and d_xy <= self.xy_tolerance:
                    arrival_step[i] = step
                data = _env_data(env)
                prepared = self.policy._prepare_info(
                    self._enumerate_infos(
                        [np.asarray(_find_env_method(env, "get_ob")())],
                        [proprio_from_qpos_qvel(data.qpos, data.qvel)],
                        [goals[i]],
                        [goal_proprios[i]],
                    )
                )
                with torch.inference_mode():
                    cur = self.policy.solver._encode_all_levels(
                        prepared, key="pixels"
                    )["embed_3"][:, 0].float().cpu().numpy()[0]
                dist = float(max(0.0, 1.0 - float(target_z[i] @ self._unit(cur))))
                min_cos[i] = min(min_cos[i], dist)
                results.append(
                    {
                        **{
                            k: v
                            for k, v in trials[i].items()
                            if k
                            not in {
                                "source_qpos",
                                "source_qvel",
                                "target_qpos",
                                "target_qvel",
                                "target_pixels",
                                "target_embedding",
                            }
                        },
                        "seed": int(seeds[i]),
                        "state_replay_ok": bool(replay_ok[i]),
                        "terminated_at_goal": bool(terminated_any[i]),
                        "arrival_step": int(arrival_step[i]),
                        "min_cosine_distance": float(min_cos[i]),
                        "min_xy_distance": float(min_xy[i]),
                        "local_success": bool(
                            terminated_any[i]
                            or (
                                arrival_step[i] >= 0
                                and min_xy[i] <= self.xy_tolerance
                            )
                        ),
                        "xy_tolerance": float(self.xy_tolerance),
                        "arrival_cosine_threshold": float(
                            self.reach_cosine_distance
                        ),
                        "max_steps": int(self.max_steps),
                        "wall_clock_s": round(time.time() - start, 3),
                        "source_qpos_sha256": sha256_array(
                            trials[i]["source_qpos"]
                        ),
                        "source_qvel_sha256": sha256_array(
                            trials[i]["source_qvel"]
                        ),
                        "target_xy": [float(x) for x in trials[i]["target_xy"]],
                    }
                )
        finally:
            self.world.close()
            self.world = None
        return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
