# RAGC integration patch (SOURCE-CHECKED at research/action-response-v1)

Keep a separate branch, record `git rev-parse HEAD`. Only after gate/labels pass.

## A. Hook in `h_jepa/hierarchical_solver.py`

Add in `HierarchicalSolver.__init__`:

    self.ragc_refiner = None

Add method in `HierarchicalSolver`:

    def set_ragc_refiner(self, refiner):
        self.ragc_refiner = refiner

In `HierarchicalSolver.solve`, near the END (currently last line is
`return {'actions': result['actions']}`), replace only that return with:

    final_actions = result['actions']
    if self.ragc_refiner is not None:
        final_actions = self.ragc_refiner.refine(
            solver=self.level_solvers[1],
            level_info=level_info,  # from THE ACTUAL Level-1 solve, not top-level
            original_actions=final_actions,
        )
    return {'actions': final_actions}

Never set action_cost_weight=0.0; leave original 0.2.
Never use `steps_taken=0` for later decision points.
Do not change or rerun the original GradientSolver's 90-iteration loop.

## B. Config-based loading in `h_jepa/planning_eval.py`

Within `_build_policy`, after `solver = build_solver(cfg, model)` and
before returning `WorldModelPolicy`, add:

    ragc_cfg = cfg.get("ragc", None)
    if _is_hierarchical_solver(cfg) and ragc_cfg is not None \
            and bool(ragc_cfg.get("enabled", False)):
        from dataclasses import replace
        from ragc import RAGCConfig, RealizedAdvantageHead, RAGCRefiner
        from action_response_wm import get_action_stats
        from pathlib import Path

        ckpt = Path(str(ragc_cfg["head_checkpoint"]))
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        if payload.get("supervision") != "baseline_continuation":
            raise RuntimeError("RAGC head was not trained on REAL task costs")
        if int(payload.get("train_steps", 0)) < 1:
            raise RuntimeError("Untrained RAGC head checkpoint")
        base = RAGCConfig(**payload["ragc_config"])
        options = dict(ragc_cfg.get("refine", {}))
        config = replace(base, **options)
        head = RealizedAdvantageHead(config)
        head.load_state_dict(payload["head_state_dict"])
        model_device = next(model.parameters()).device
        head = head.to(model_device).eval()
        mean, std = get_action_stats(model, cfg.policy, model_device)
        solver.set_ragc_refiner(RAGCRefiner(head, config, mean, std))

No changes to `WorldModelPolicy`, `_run_chunked_eval`, or policy execution length.

## C. New file `h_jepa/config/eval/pusht_l2_ragc.yaml`

    defaults:
      - pusht_l2
      - _self_
    ragc:
      enabled: false
      head_checkpoint: ${oc.env:HJEPA_HOME}/ragc/pusht/realized_advantage_head.pt
      refine:
        refinement_steps: 3
        step_size: 0.10
        advantage_weight: 0.30
        trust_radius: 0.40
        trust_weight: 0.10
        max_original_cost_regression: 0.10
        min_advantage_improvement: 0.005
        apply_when_horizon_one: false

IMPORTANT: Hydra defaults resolution in this repo must be inspected.
If `defaults` produces wrong contents, copy `pusht_l2.yaml` to a new config
and add the `ragc:` section instead. Do NOT edit the original file.

## D. Quick validation gate

1. Set `ragc.enabled=false` => new policy must reproduce baseline actions
   and outcomes exactly under identical task IDs/seeds.
2. Set `refine.advantage_weight=0` => same exact actions.
3. Train one tiny synthetic task-cost dataset to check forward/backprop.
4. Check `refiner.last_stats`: accepted count, bounded actions, old cost change.
5. For actual simulator integration, do 1 task, then 4 tasks.
6. Only AFTER real continuation labels + comparisons: run held-out 20-task
   pilot and full 50-task paired evaluation.
7. Keep `output.dir` different per setting or previous chunk JSONs get reused.
8. Record extra refiner compute; baseline planner solve_records do not account
   for those extra passes.

## E. Required warnings

- `goal_embed_0` can be HIGHER-level latent even when same numeric dim 192.
  Never subtract it from z0 as if they lived in one space.
- The top-down goal can change over planning calls; training samples must use
  the ACTUAL `level_info` captured during each original solve.
- `GradientSolver.solve` returns CPU actions and its own predictions are stale
  if the post-solve action is corrected. In current HierarchicalSolver only
  actions are returned; DO NOT return stale predictions from here.
- L1 Horizon may decay to one token; head must handle H=1 and H=2.
- Existing `action_ranking_test.py` sets level-1 action_cost_weight=0 and
  steps_taken=0 for late decision steps. DO NOT copy those modifications.
- Use `WorldModelPolicy`'s action scaler from the training checkpoint.
- `PushT._set_state` misses block velocity / angular velocity. For actual
  candidate causal outcome labels use the SAME initial EVAL state AND identical
  completed action prefix; assert block hidden physics matches at anchor.
- Always preserve goal image and render `with_target=True` for true eval.
- Do NOT train from the final 50 test tasks; it leaks answers.
- A short-horizon position/rotation potential is not final success, and
  should not be described as "realized task advantage" without qualification.
- No new module is proven to improve success until closed-loop held-out eval.


## F. Runnable oracle gate CLI (source-aligned reference)

    cd H-JEPA/h_jepa
    export PYTHONPATH=. SDL_VIDEODRIVER=dummy HJEPA_HOME=/...
    python -u scripts/ragc_oracle_pilot.py \
      --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
      --episodes 1 --max-decisions 1 --directions 1 \
      --out /tmp/ragc_oracle_smoke.json

This script does not patch GradientSolver, and uses the original hierarchical
Action Cost and real steps_taken. It performs full frozen-policy continuation.
The very FIRST gate is official-eval equivalence: task seed, exact action
trace, horizon sequence, render, termination, and episode success must match.
Only then trust it to make causal labels. On default heldout eval_trajs,
the script refuses --dataset training export to prevent evaluation leakage.
