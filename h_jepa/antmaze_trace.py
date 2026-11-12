"""Opt-in AntMaze planning trace logger (strict no-op when disabled).

Enabled from the evaluation entry point with ``+eval.trace_dir=<path>``
(relative paths resolve under the evaluation output directory, whether the
directory is run relative to the checkpoint or anywhere else).

The tracer only *observes* data the evaluation already produces:

* ``World.step`` is called once per environment step -> per-step records.
* ``WorldModelPolicy.get_action`` returns the action applied to the envs.
* each level solver's ``solve`` and its cost model's ``get_cost`` produce
  the planning records (horizon, iterations, wall clock, chosen actions,
  selected subgoal latents and GD cost curve).

It never draws random numbers, never changes the action stream, never runs
an extra model forward pass and never mutates planner state.  When enabled,
it only wraps callables on the live objects; when disabled nothing is
imported into the evaluation path beyond this module and no attribute is
touched.

Outputs (under ``<results_dir>/<trace_dir>/``)::

    meta.json           run identity, task mapping, maze layout
    steps.jsonl         one JSON line per env step
    planning.jsonl      one JSON line per (planning call, level)
    tasks/task_XXX.json per-task copies (atomic, written on finish)
    trace_done.json     completion marker + record counts

Latent vectors are summarized (bucket-mean over the embedding dimension);
they are *not* decoded to XY.  Only the chosen plan is stored, never the
``num_samples`` candidates or per-iteration computation graphs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_BUCKETS = 64


def _jsonable(value: Any) -> Any:
    """Convert numpy / torch scalars and arrays into JSON-friendly objects."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # torch tensors are handled by the callers; leave unknown objects as str
    return str(value)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:  # pragma: no cover - torch is always available in eval
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _flatten_to_1d(value: Any) -> np.ndarray | None:
    arr = _as_numpy(value)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _env_item(infos: dict, key: str, i: int):
    value = infos.get(key)
    if value is None:
        return None
    try:
        return value[i]
    except Exception:
        return value


def _bucket_vector(tensor) -> list[float] | None:
    """Mean-pool to one vector over all but the last dim, then bucket the last dim."""
    import torch

    if tensor is None or not torch.is_tensor(tensor):
        return None
    with torch.no_grad():
        x = tensor.detach().float()
        if x.numel() == 0:
            return None
        if x.ndim == 1:
            v = x
        else:
            v = x.reshape(-1, x.shape[-1]).mean(dim=0)
        v = v.reshape(-1)
        d = int(v.numel())
        if d > _BUCKETS:
            pad = (-d) % _BUCKETS
            if pad:
                v = torch.cat([v, v.new_zeros(pad)])
            v = v.reshape(_BUCKETS, -1).mean(dim=1)
        return [float(z) for z in v.cpu().tolist()]


def _cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= 0 or ny <= 0:
        return None
    return float(np.dot(x, y) / (nx * ny))


def _latent_env_summary(tensor, n_envs: int) -> dict[str, list]:
    """Per-env compact summary of a ``(B, ...)`` latent tensor."""
    import torch

    out: dict[str, list] = {
        "norm_last": [],
        "vec64": [],
        "shape": list(tensor.shape) if torch.is_tensor(tensor) else None,
    }
    if tensor is None or not torch.is_tensor(tensor):
        return out
    # One host transfer for the whole batch (avoids per-env GPU syncs).
    with torch.no_grad():
        cpu = tensor.detach().float().cpu()
    for i in range(n_envs):
        if i >= cpu.shape[0]:
            out["norm_last"].append(None)
            out["vec64"].append(None)
            continue
        x = cpu[i]
        with torch.no_grad():
            last = x[-1] if x.ndim >= 2 else x
            out["norm_last"].append(float(last.reshape(-1).norm()))
            out["vec64"].append(_bucket_vector(x))
    return out


class AntMazeTrace:
    """Observe an in-flight planning evaluation and write JSONL traces."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        worker: str | None = None,
        seed: int | None = None,
        task_ids: list[int] | None = None,
        replanning_interval: int = 10,
        meta: dict | None = None,
        action_scaler=None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "tasks").mkdir(parents=True, exist_ok=True)

        self.worker = worker
        self.seed = seed
        self.task_ids = list(task_ids) if task_ids is not None else None
        self.replanning_interval = max(1, int(replanning_interval))
        self.meta = dict(meta or {})
        self._action_scaler = action_scaler

        self._n_envs: int | None = None
        self._step = 0
        self._pending_actions: np.ndarray | None = None
        self._prev_subgoal: dict[int, list] = {}
        self._costs: list = []
        self._active_level: int | None = None
        self._step_fh = None
        self._plan_fh = None
        self._start_time = time.time()
        self._finished = False

        self._task_steps: dict[int, list] = defaultdict(list)
        self._task_planning: dict[int, list] = defaultdict(list)
        self._n_step_records = 0
        self._n_planning_records = 0
        self._n_solves = 0

    # ------------------------------------------------------------------ setup
    def _task_id(self, env_idx: int) -> int:
        if self.task_ids is not None and env_idx < len(self.task_ids):
            return int(self.task_ids[env_idx])
        return int(env_idx)

    def begin(self) -> None:
        meta = {
            "format": "antmaze_trace_v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "worker": self.worker,
            "seed": self.seed,
            "task_ids": self.task_ids,
            "replanning_interval": self.replanning_interval,
            **self.meta,
        }
        tmp = self.out_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(_jsonable(meta), indent=1))
        os.replace(tmp, self.out_dir / "meta.json")
        self._step_fh = open(self.out_dir / "steps.jsonl", "w", buffering=1)
        self._plan_fh = open(self.out_dir / "planning.jsonl", "w", buffering=1)

    def attach(self, world, policy) -> None:
        """Install observation wrappers on live eval objects."""
        solver = getattr(policy, "solver", None)
        level_solvers = getattr(solver, "level_solvers", None)
        if level_solvers:
            for level in sorted(level_solvers):
                self._wrap_solver(level_solvers[level], int(level))
        elif solver is not None:
            self._wrap_solver(solver, int(getattr(solver, "level", 1)))

        self._wrap_policy(policy)
        self._wrap_world_step(world)

        # n_envs is known once the vector env exists.
        self._n_envs = int(getattr(world, "num_envs", 0) or 0)

    def _wrap_policy(self, policy) -> None:
        if not hasattr(policy, "get_action"):
            return
        original = policy.get_action
        trace = self

        def get_action(*args, **kwargs):
            action = original(*args, **kwargs)
            arr = _as_numpy(action)
            trace._pending_actions = None if arr is None else np.array(arr, copy=True)
            return action

        policy.get_action = get_action

    def _wrap_world_step(self, world) -> None:
        if not hasattr(world, "step"):
            return
        original = world.step
        trace = self

        def step():
            if trace._step == 0:
                trace._record_infos(
                    world.infos,
                    terminated=False,
                    truncated=False,
                    action=None,
                )
            out = original()
            action = trace._pending_actions
            trace._pending_actions = None
            trace._step += 1
            # ``world.infos`` now holds the post-step observations.
            n = trace._n_envs or int(getattr(world, "num_envs", 0) or 0)
            for i in range(n):
                terminated = bool(np.asarray(world.terminateds).reshape(-1)[i])
                truncated = bool(np.asarray(world.truncateds).reshape(-1)[i])
                trace._record_infos(
                    world.infos,
                    terminated=terminated,
                    truncated=truncated,
                    action=None if action is None else action[i],
                    env_idx=i,
                )
            return out

        world.step = step

    def _wrap_solver(self, solver, level: int) -> None:
        if not hasattr(solver, "solve"):
            return
        original = solver.solve
        trace = self

        def solve(*args, **kwargs):
            before = len(getattr(solver, "solve_records", []) or [])
            trace._active_level = level
            trace._costs = []
            out = None
            try:
                out = original(*args, **kwargs)
            finally:
                trace._active_level = None
            if out is not None:
                trace._record_planning(solver, level, args, kwargs, out, before)
            return out

        solver.solve = solve

        model = getattr(solver, "model", None)
        if model is not None and hasattr(model, "get_cost"):
            original_cost = model.get_cost

            def get_cost(info_dict, action_candidates, *a, **kw):
                cost = original_cost(info_dict, action_candidates, *a, **kw)
                if trace._active_level == level and hasattr(cost, "detach"):
                    # Keep GPU tensors only; a single .cpu() happens at solve
                    # end (avoids a sync on every GD iteration).
                    trace._costs.append(cost.detach().mean(dim=1).float())
                return cost

            model.get_cost = get_cost

    # --------------------------------------------------------------- records
    def _record_infos(
        self,
        infos: dict,
        *,
        terminated: bool,
        truncated: bool,
        action: Any,
        env_idx: int | None = None,
    ) -> None:
        n = self._n_envs or (len(next(iter(infos.values()))) if infos else 0)
        self._n_envs = n
        indices = range(n) if env_idx is None else [env_idx]

        for i in indices:
            record = {
                "t": int(self._step),
                "env": int(i),
                "task": self._task_id(i),
                "xy": _flatten_to_1d(infos.get("xy")[i]) if "xy" in infos else None,
                "goal_xy": (
                    _flatten_to_1d(infos.get("goal_xy")[i])
                    if "goal_xy" in infos
                    else None
                ),
                "qpos": _flatten_to_1d(infos.get("qpos")[i]) if "qpos" in infos else None,
                "qvel": _flatten_to_1d(infos.get("qvel")[i]) if "qvel" in infos else None,
                "action": _flatten_to_1d(action),
                "action_norm": self._normalized_action(action),
                "success": bool(
                    np.asarray(infos["success"]).reshape(-1)[i]
                    if "success" in infos
                    and np.asarray(infos["success"]).size > i
                    else False
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            if record["t"] == 0:
                # One-off fingerprints of the state the planner actually sees.
                record["hashes"] = {
                    "qpos": _sha256_value(_env_item(infos, "qpos", i)),
                    "qvel": _sha256_value(_env_item(infos, "qvel", i)),
                    "xy": _sha256_value(_env_item(infos, "xy", i)),
                    "goal_xy": _sha256_value(_env_item(infos, "goal_xy", i)),
                    "pixels": _sha256_value(_env_item(infos, "pixels", i)),
                    "goal_pixels": _sha256_value(
                        _env_item(infos, "goal", i)
                    ),
                }
            self._write(self._step_fh, record)
            self._task_steps[record["task"]].append(record)
            self._n_step_records += 1

    def _normalized_action(self, action) -> list[float] | None:
        """Reconstruct the model-space action from the applied one.

        ``WorldModelPolicy.get_action`` applies ``inverse_transform`` before
        returning; applying the forward transform again recovers the planner's
        normalized action up to float round-off.
        """
        if self._action_scaler is None or action is None:
            return None
        arr = _as_numpy(action)
        if arr is None:
            return None
        try:
            return self._action_scaler.transform(arr.reshape(1, -1))[0].tolist()
        except Exception:
            return None

    def _record_planning(self, solver, level: int, args, kwargs, out, before) -> None:
        info_dict = kwargs.get("info_dict")
        if info_dict is None and args:
            info_dict = args[0]
        info_dict = info_dict or {}

        steps_taken = kwargs.get("steps_taken")
        if steps_taken is None and len(args) > 2:
            steps_taken = args[2]
        steps_taken = int(steps_taken or 0)

        horizon = kwargs.get("planning_horizon")
        if horizon is None:
            horizon = getattr(solver, "horizon", None)
        horizon = None if horizon is None else int(horizon)

        n = self._n_envs or int(getattr(solver, "n_envs", 0) or 0)

        records = list(getattr(solver, "solve_records", []) or [])[before:]
        solve_record = records[0] if records else {}
        n_iters = solve_record.get("n_iters")

        predictions = (out or {}).get("predictions", {}) or {}
        pred = predictions.get("predicted_embed_0")
        pred_summary = _latent_env_summary(pred, n)
        cos_prev = []
        for i, vec in enumerate(pred_summary["vec64"]):
            cos_prev.append(_cosine(vec, self._prev_subgoal.get((level, i))))
        self._prev_subgoal = dict(self._prev_subgoal)
        for i, vec in enumerate(pred_summary["vec64"]):
            self._prev_subgoal[(level, i)] = vec

        goal = info_dict.get("goal_embed_0")
        goal_summary = _latent_env_summary(goal, n)

        actions = (out or {}).get("actions")
        action_list = _as_numpy(actions)
        if action_list is not None:
            action_list = np.asarray(action_list, dtype=np.float64).tolist()

        costs = self._flatten_cost_batches(solver, n_iters, n)
        self._n_solves += 1

        record = {
            "kind": "planning",
            "level": int(level),
            "call_idx": int(steps_taken // self.replanning_interval),
            "steps_taken": steps_taken,
            "horizon": horizon,
            "num_samples": int(getattr(solver, "num_samples", 0) or 0),
            "n_steps": int(getattr(solver, "n_steps", 0) or 0),
            "n_iters": _jsonable(n_iters),
            "wall_clock": float(solve_record.get("wall_clock", 0.0) or 0.0),
            "goal_level": _jsonable(info_dict.get("goal_embed_0_level")),
            "goal_in_upper_space": bool(
                info_dict.get("goal_embed_0_in_upper_space", False)
            ),
            "goal_norm_last": goal_summary["norm_last"],
            "goal_vec64": goal_summary["vec64"],
            "pred_shape": pred_summary["shape"],
            "pred_norm_last": pred_summary["norm_last"],
            "pred_vec64": pred_summary["vec64"],
            "pred_cos_prev": cos_prev,
            "actions": action_list,
            "cost_batches": costs,
            "tasks": [self._task_id(i) for i in range(n)],
        }
        self._write(self._plan_fh, record)
        for i in range(n):
            per_env = {
                "level": record["level"],
                "call_idx": record["call_idx"],
                "steps_taken": record["steps_taken"],
                "horizon": record["horizon"],
                "n_iters": (
                    int(n_iters[i]) if isinstance(n_iters, list) and i < len(n_iters) else None
                ),
                "pred_norm_last": (
                    pred_summary["norm_last"][i] if i < len(pred_summary["norm_last"]) else None
                ),
                "pred_cos_prev": cos_prev[i] if i < len(cos_prev) else None,
                "actions": None if action_list is None else action_list[i],
            }
            self._task_planning[record["tasks"][i]].append(per_env)
        self._n_planning_records += 1

    def _flatten_cost_batches(self, solver, n_iters, n_envs: int) -> list[dict] | None:
        """Regroup the interleaved per-batch cost calls of one solver call.

        GradientSolver processes envs in deterministic slices of
        ``batch_size`` and runs ``n_iters`` GD steps + one final cost call per
        slice, so the flat call sequence can be split exactly.
        """
        import torch

        if not self._costs or not isinstance(n_iters, list) or not n_iters:
            self._costs = []
            return None
        try:
            flat = torch.cat([c.reshape(-1) for c in self._costs]).cpu()
        except Exception:
            return None
        finally:
            self._costs = []

        batch_size = int(getattr(solver, "batch_size", 0) or 0) or n_envs
        sizes: list[int] = []
        remaining = n_envs
        while remaining > 0:
            size = min(batch_size, remaining)
            sizes.append(size)
            remaining -= size

        batches = []
        cursor = 0
        env_start = 0
        for k, size in enumerate(sizes):
            start_env = sum(sizes[:k])
            if start_env >= len(n_iters):
                break
            iters = int(n_iters[start_env])
            count = (iters + 1) * size
            chunk = flat[cursor : cursor + count]
            if chunk.numel() != count:
                break
            cursor += count
            mat = chunk.reshape(iters + 1, size)
            batches.append(
                {
                    "batch": k,
                    "env_start": int(env_start),
                    "env_end": int(env_start + size),
                    "n_envs": int(size),
                    "n_iters": iters,
                    "cost_curve": [float(x) for x in mat.mean(dim=1).tolist()],
                    "cost_final_per_env": [float(x) for x in mat[-1].tolist()],
                }
            )
            env_start += size
        return batches or None

    @staticmethod
    def _write(fh, record: dict) -> None:
        if fh is None:
            return
        fh.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")

    # ---------------------------------------------------------------- finish
    def finish(self, error: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        for fh in (self._step_fh, self._plan_fh):
            try:
                if fh is not None:
                    fh.flush()
                    os.fsync(fh.fileno())
                    fh.close()
            except Exception:
                pass
        self._step_fh = None
        self._plan_fh = None

        n = self._n_envs or 0
        for task, steps in self._task_steps.items():
            path = self.out_dir / "tasks" / f"task_{int(task):03d}.json"
            payload = {
                "task": int(task),
                "n_envs": n,
                "steps": steps,
                "planning": self._task_planning.get(task, []),
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(_jsonable(payload)))
            os.replace(tmp, path)

        done = {
            "format": "antmaze_trace_done_v1",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "wall_clock": time.time() - self._start_time,
            "n_step_records": self._n_step_records,
            "n_planning_records": self._n_planning_records,
            "n_solves": self._n_solves,
            "tasks": sorted(int(t) for t in self._task_steps),
            "error": error,
        }
        tmp = self.out_dir / "trace_done.json.tmp"
        tmp.write_text(json.dumps(done, indent=1))
        os.replace(tmp, self.out_dir / "trace_done.json")


def resolve_trace_dir(cfg, results_dir: str | Path) -> Path | None:
    """Return the trace directory requested by ``eval.trace_dir`` (or None)."""
    value = None
    try:
        value = cfg.eval.get("trace_dir", None)
    except Exception:
        value = None
    if value in (None, False, "", "null", "none"):
        return None
    if value is True:
        value = "trace"
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(results_dir) / path
    return path


def _sha256_value(value) -> str | None:
    arr = _as_numpy(value)
    if arr is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_file(path: str | Path, chunk: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _cfg_get(obj, name: str, default=None):
    try:
        if hasattr(obj, "get"):
            return obj.get(name, default)
        return getattr(obj, name, default)
    except Exception:
        return default


def _task_ids_from_payload(load_eval_trajs_path: str | Path | None) -> list[int] | None:
    if not load_eval_trajs_path:
        return None
    import torch

    payload = torch.load(load_eval_trajs_path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes_idx")
    if episodes is None or len(episodes) == 0:
        episodes = list(range(len(payload.get("data", []))))
    return [int(x) for x in episodes]


def _maze_meta(world) -> dict:
    try:
        base = world.envs.envs[0].unwrapped
        og_env = getattr(base, "env", base)
        meta = {
            "maze_map": np.asarray(og_env.maze_map).tolist(),
            "maze_unit": float(getattr(og_env, "_maze_unit", np.nan)),
            "env_name": str(getattr(og_env, "env_name", "")),
        }
        return meta
    except Exception:
        return {}


def install_trace(
    *,
    cfg,
    world,
    policy,
    results_dir: str | Path,
    load_eval_trajs_path: str | Path | None = None,
    extra_meta: dict | None = None,
    process: dict | None = None,
) -> AntMazeTrace | None:
    """Install the tracer when ``eval.trace_dir`` is set. Returns None otherwise."""
    out_dir = resolve_trace_dir(cfg, results_dir)
    if out_dir is None:
        return None

    task_ids = _task_ids_from_payload(load_eval_trajs_path)
    worker = None
    output_cfg = _cfg_get(cfg, "output", {}) or {}
    output_dir = _cfg_get(output_cfg, "dir", None)
    if output_dir:
        name = Path(str(output_dir)).name
        # ``eval_ant_l3_par/w1`` -> "w1"; otherwise keep the directory name.
        worker = name if name.startswith("w") and name[1:].isdigit() else name

    replanning_interval = 1
    try:
        plan_cfg = cfg.hierarchical_plan_config
        replanning_interval = int(plan_cfg.receding_horizon) * int(
            plan_cfg.level1.action_block
        )
    except Exception:
        try:
            replanning_interval = int(cfg.plan_config.receding_horizon) * int(
                cfg.plan_config.action_block
            )
        except Exception:
            replanning_interval = 1

    eval_cfg = _cfg_get(cfg, "eval", {}) or {}
    solver = getattr(policy, "solver", None)
    level_solvers = getattr(solver, "level_solvers", None) or (
        {str(getattr(solver, "level", 1)): solver} if solver is not None else {}
    )
    planner_seeds = {}
    batch_sizes = {}
    for level, level_solver in level_solvers.items():
        gen = getattr(level_solver, "torch_gen", None)
        if gen is not None:
            planner_seeds[str(level)] = int(gen.initial_seed())
        if getattr(level_solver, "batch_size", None) is not None:
            batch_sizes[str(level)] = int(level_solver.batch_size)

    env_seed = _cfg_get(eval_cfg, "env_seed", None)
    env_seed_by_task = None
    if env_seed is not None and task_ids is not None:
        env_seed_by_task = {str(t): int(env_seed) + int(t) for t in task_ids}

    n_envs = int(getattr(world, "num_envs", 0) or 0)
    env_to_batch = None
    if n_envs and batch_sizes:
        batch_size = int(next(iter(batch_sizes.values())))
        env_to_batch = [i // max(1, batch_size) for i in range(n_envs)]

    policy_path = str(_cfg_get(cfg, "policy", ""))
    meta = {
        "policy": policy_path,
        "checkpoint_sha256": _sha256_file(policy_path) if policy_path else None,
        "eval_budget": _cfg_get(eval_cfg, "eval_budget", None),
        "num_eval": _cfg_get(eval_cfg, "num_eval", None),
        "env_seed": env_seed,
        "env_seed_by_task": env_seed_by_task,
        "planner_seeds": planner_seeds,
        "batch_size_by_level": batch_sizes,
        "env_to_batch": env_to_batch,
        **_maze_meta(world),
        **(extra_meta or {}),
    }

    tracer = AntMazeTrace(
        out_dir,
        worker=worker,
        seed=int(_cfg_get(cfg, "seed", 0) or 0),
        task_ids=task_ids,
        replanning_interval=replanning_interval,
        meta=meta,
        action_scaler=(process or {}).get("action"),
    )
    tracer.begin()
    tracer.attach(world, policy)
    print(f"[antmaze_trace] recording to {out_dir}")
    return tracer
