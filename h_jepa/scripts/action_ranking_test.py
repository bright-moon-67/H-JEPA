"""Counterfactual Action Ranking Test (plan2/4.md).

Question: are the planner's predicted action costs consistent with what the
REAL simulator produces, when scored with the SAME latent goal cost?

For a real Push-T state s_t and a fixed goal:
  1. run the original hierarchical solver once -> A* (normalized action tokens)
  2. sample candidates A*, A_1..A_K around A*
  3. J_pred(A_i): the original LevelCostModel.get_cost (same subgoal, same
     projection, same criterion, same action cost) for the candidate
  4. J_real(A_i): execute A_i in the real env from s_t, encode the real frames
     with the frozen encoder, substitute them into the SAME get_cost path
     (via a temporary rollout stub), so projection/criterion/action cost are
     identical
  5. compare rankings (Spearman/Kendall), A* regret, and physical block errors

Run from h_jepa/:
    PYTHONPATH=. python -u scripts/action_ranking_test.py \
        --config-name pusht_l2 --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
        --episodes 3 --decision-steps 0 40 --candidates 6 --out /tmp/ranking.json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401
import torch
from scipy.stats import kendalltau, spearmanr

from action_response_wm import get_action_stats
from data import IMAGENET_STATS
from eval_config_utils import _build_policy_plan_config
from hierarchical_solver import build_hierarchical_solver
from planning_eval import load_model

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def make_env():
    return gym.make(
        "swm/PushT-v1", render_mode="rgb_array", resolution=224,
        with_target=False, relative=True, max_episode_steps=200,
    )


def prep_image(frame, device):
    x = torch.from_numpy(np.ascontiguousarray(frame)).to(device)
    x = x.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(IMAGENET_STATS["mean"], device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STATS["std"], device=device).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]


def render_state(env, state):
    env.reset(options={"state": np.asarray(state, dtype=np.float64)})
    return np.asarray(env.render(), dtype=np.uint8).copy()


def wrapped_angle_diff(a, b):
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(abs(np.degrees(d)))


def execute_candidate(env, state, tokens, mean, std, model, device, h_tokens):
    """Execute h_tokens normalized action tokens (each = 5 env steps)."""
    env.reset(options={"state": np.asarray(state, dtype=np.float64)})
    zs = [prep_image(np.asarray(env.render(), dtype=np.uint8), device)]
    for t in range(h_tokens):
        raw = tokens[t].detach().cpu().numpy().reshape(5, 2) * std + mean
        raw = np.clip(raw, -1.0, 1.0)
        for a in raw:
            env.step(a.astype(np.float32))
        zs.append(prep_image(np.asarray(env.render(), dtype=np.uint8), device))

    level1 = model.get_level(1)
    with torch.no_grad():
        embeds = []
        for x in zs:
            out = level1.encode({"pixels": x}, key="pixels")["embed_0"]
            embeds.append(out[:, 0])
    z_seq = torch.stack(embeds, dim=1).unsqueeze(1)  # [1,1,h+1,D]
    end_state = np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    return z_seq, end_state


class RolloutStub:
    """Temporarily replace LevelCostModel.rollout with real encoded frames."""

    def __init__(self, cost_model, z_seq):
        self.cost_model = cost_model
        self.z_seq = z_seq
        self.orig = cost_model.rollout

    def __enter__(self):
        def fake(info_dict, candidates):
            out = dict(info_dict)
            out["predicted_embed_0"] = self.z_seq.expand(
                candidates.shape[0], -1, -1, -1
            ).clone()
            return out
        self.cost_model.rollout = fake
        return self

    def __exit__(self, *a):
        self.cost_model.rollout = self.orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pusht_l2")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--trajs", default="assets/eval_trajs/pusht/goal_offset_75_val.pt")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--decision-steps", type=int, nargs="+", default=[0, 40])
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="/tmp/ranking.json")
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parent.parent / "config" / "eval"),
    ):
        cfg = compose(config_name=args.config_name)
    cfg.policy = args.ckpt

    model = load_model(cfg, model=None)
    mean_t, std_t = get_action_stats(model, args.ckpt, device)
    mean = mean_t.cpu().numpy()
    std = std_t.cpu().numpy()

    hier = build_hierarchical_solver(cfg, model)
    env = make_env()
    plan_cfg = _build_policy_plan_config(cfg)
    # The action-cost term is model-internal (compares predicted macro-actions
    # with the upper plan's actions) and has no observable counterpart in the
    # real simulator, so the ranking test uses the goal-cost term only.
    import dataclasses

    levels = tuple(
        dataclasses.replace(lv, action_cost_weight=0.0)
        if hasattr(lv, "action_cost_weight") else lv
        for lv in plan_cfg.levels
    )
    plan_cfg = dataclasses.replace(plan_cfg, levels=levels)
    # The solver expects a BATCHED action space: (n_envs, action_dim).
    batched_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32
    )
    hier.configure(action_space=batched_action_space, n_envs=1, config=plan_cfg)
    level1_cost_model = hier.level_solvers[1].model
    if os.environ.get("RANK_DEBUG"):
        _orig_rollout = level1_cost_model.rollout

        def _dbg_rollout(info_dict, candidates):
            px = info_dict.get("pixels")
            print(f"  [rollout] pixels={tuple(px.shape) if torch.is_tensor(px) else type(px)} "
                  f"cands={tuple(candidates.shape)}", flush=True)
            return _orig_rollout(info_dict, candidates)

        level1_cost_model.rollout = _dbg_rollout

    # capture the level-1 info (embed_0 / subgoal) during the real solve
    captured = {}
    orig_l1_solve = hier.level_solvers[1].solve

    def wrapped_l1_solve(*a, **kw):
        captured["info"] = {k: (v.clone() if torch.is_tensor(v) else v)
                            for k, v in kw["info_dict"].items()}
        if os.environ.get("RANK_DEBUG"):
            for k, v in captured["info"].items():
                if torch.is_tensor(v):
                    print(f"  [l1 info] {k}: {tuple(v.shape)}", flush=True)
        return orig_l1_solve(*a, **kw)

    hier.level_solvers[1].solve = wrapped_l1_solve

    payload = torch.load(args.trajs, map_location="cpu", weights_only=False)
    episodes = payload["data"][: args.episodes]

    records = []
    for ep_i, ep in enumerate(episodes):
        states = ep["state"].numpy() if torch.is_tensor(ep["state"]) else np.asarray(ep["state"])
        goal_state = states[-1]
        for t in args.decision_steps:
            if t >= len(states) - 1:
                continue
            s_t = states[t]
            captured.clear()

            frame_cur = render_state(env, s_t)
            frame_goal = render_state(env, goal_state)
            info = {"pixels": prep_image(frame_cur, device),
                    "goal": prep_image(frame_goal, device)}

            res = hier.solve(info, steps_taken=0, eval_budget=int(cfg.eval.eval_budget))
            a_star = res["actions"][0].detach().cpu()  # [H_tokens,10]
            level_info = captured["info"]
            level_info["cost_last_n"] = int(getattr(plan_cfg.level1, "cost_last_n", 1))
            level_info["intermediate_cost_weight"] = float(
                getattr(plan_cfg.level1, "intermediate_cost_weight", 1.0)
            )

            k = args.candidates
            cands = [a_star]
            for _ in range(k - 1):
                cands.append(a_star + torch.from_numpy(
                    rng.normal(0, args.sigma, size=a_star.shape).astype(np.float32)))
            cands = torch.stack(cands)  # [K,H,10]

            rec = {"episode": ep_i, "step": int(t), "a_star_pred_cost": None,
                   "horizons": {}, "goal_state": goal_state.tolist(),
                   "start_state": s_t.tolist()}

            for h in (1, 2):
                if h > a_star.shape[0]:
                    continue
                info_h = dict(level_info)
                # match subgoal horizon: level-1 tokens are 5 env steps each
                goal_emb = info_h.get("goal_embed_0")
                if torch.is_tensor(goal_emb) and goal_emb.ndim >= 3 and goal_emb.shape[-2] > h:
                    info_h["goal_embed_0"] = goal_emb[:, :h]
                goal_act = info_h.get("goal_action")
                if torch.is_tensor(goal_act) and goal_act.ndim >= 3 and goal_act.shape[-2] > h:
                    info_h["goal_action"] = goal_act[:, :h]
                cand_h = cands[:, :h]
                info_pred = hier.level_solvers[1]._expand_info_dict_for_samples(
                    info_h, start_idx=0, end_idx=1, num_samples=k
                )
                j_pred = level1_cost_model.get_cost(
                    info_pred, cand_h.unsqueeze(0).to(device)
                )[0].detach().cpu().numpy()

                j_real = np.zeros(k)
                phys = []
                for c in range(k):
                    z_seq, end_state = execute_candidate(
                        env, s_t, cand_h[c], mean, std, model, device, h
                    )
                    info_real = hier.level_solvers[1]._expand_info_dict_for_samples(
                        info_h, start_idx=0, end_idx=1, num_samples=1
                    )
                    with RolloutStub(level1_cost_model, z_seq):
                        j_real[c] = float(level1_cost_model.get_cost(
                            info_real, cand_h[c:c + 1].unsqueeze(0).to(device)
                        )[0, 0].cpu())
                    block_err = float(np.linalg.norm(end_state[2:4] - goal_state[2:4]))
                    ang_err = wrapped_angle_diff(end_state[4], goal_state[4])
                    phys.append({"block_err_px": block_err, "angle_err_deg": ang_err})

                rho = float(spearmanr(j_pred, j_real).statistic) if k > 2 else float("nan")
                tau = float(kendalltau(j_pred, j_real).statistic) if k > 2 else float("nan")
                star_real = float(j_real[0])
                best_real = float(j_real.min())
                best_pred_idx = int(np.argmin(j_pred))

                rec["horizons"][str(h)] = {
                    "j_pred": j_pred.tolist(),
                    "j_real": j_real.tolist(),
                    "spearman": rho,
                    "kendall": tau,
                    "a_star_real_regret": star_real - best_real,
                    "a_star_rank_real": int(np.argsort(j_real).tolist().index(0)),
                    "best_pred_idx": best_pred_idx,
                    "best_pred_is_real_best": bool(best_pred_idx == int(np.argmin(j_real))),
                    "physical": phys,
                    "a_star_physical": phys[0],
                    "real_best_physical": phys[int(np.argmin(j_real))],
                }
            records.append(rec)
            print(f"[ep{ep_i} t={t}] "
                  + " ".join(f"h{h}: rho={rec['horizons'][str(h)]['spearman']:+.2f} "
                             f"regret={rec['horizons'][str(h)]['a_star_real_regret']:.4f}"
                             for h in (1, 2) if str(h) in rec["horizons"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config_name, "records": records}, indent=2))

    print("\n=== aggregate ===")
    for h in ("1", "2"):
        rows = [r["horizons"][h] for r in records if h in r["horizons"]]
        if not rows:
            continue
        rhos = np.array([r["spearman"] for r in rows])
        regrets = np.array([r["a_star_real_regret"] for r in rows])
        print(f"h={h} tokens ({5 * int(h)} env steps): n={len(rows)} "
              f"mean_spearman={rhos.mean():+.3f} "
              f"frac_rho>0={(rhos > 0).mean():.1%} "
              f"mean_A*_regret={regrets.mean():.4f} "
              f"best_pred_hits_real_best={np.mean([r['best_pred_is_real_best'] for r in rows]):.1%}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
