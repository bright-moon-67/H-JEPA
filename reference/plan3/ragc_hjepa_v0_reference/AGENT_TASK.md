# AI Agent coding task — RAGC-HJEPA (success rate first)

Reference branch: `research/action-response-v1` at
https://github.com/bright-moon-67/H-JEPA
Read the actual files and record `git rev-parse HEAD` before changing code.

## Goal
Improve complete Push-T success rate, not latent R² or event AUPRC. Retain
frozen H-JEPA L1/L2 checkpoint, original 2-level top-down solver, original
action_cost_weight=0.2, 10-env-step execution, and original horizon schedule.

## Stage 0: causal opportunity gate (DO THIS FIRST)
Take REAL baseline policy trajectories on NON-TEST evaluation-like tasks.
Capture per planning call:
- original environment state, goal_state, env randomization seed,
  all completed RAW 2-D actions in the episode (not just 7-D state at t);
- actual current `level_info` and output standardized `[H,10]` actions;
- `steps_taken` & eval_budget; first compare unmodified baseline replay.
Generate +/- antithetic candidates around the actual selected action,
run from identical full hidden state via reset-from-initial + prefix replay,
and use identical baseline continuation for each candidate. Record final
success + remaining budget + terminal pose error.
This requires wiring into the existing eval harness. Never label
continuation success from just a 10-step endpoint.

If almost no candidate recovers failed tasks in this small gate, STOP RAGC;
a local gradient adjustment has no demonstrated opportunity to help.

## Stage 1: collect paired REAL success-aware outcomes
Train/val/test split by episode+goal+anchor, NOT by perturbed candidate.
Use the actual frozen base policy action neighborhoods, not old random
`free/approach/push` collector actions. Save NPZ train/val with:
  anchor_id, z0, goal, goal_is_upper, plus, minus,
  plus_cost, minus_cost
where costs are from simulator continuation and the same frozen policy.
Do not train on the held-out final 50 tasks.

## Stage 2: head training
`PYTHONPATH=. python -u scripts/train_ragc.py --train ... --val ...
  --out /tmp/ragc_head.pt --steps 100`
The first 100 iterations are ONLY a smoke; compare out-of-sample candidate
pair preference accuracy and pair MSE against chance/zero. More iterations
require a separate experimental decision.

## Stage 3: guarded post-solve action correction
Copy `PATCH_GUIDE.md` precisely, adapting ONLY verified source paths.
Run `python -m pytest -q tests/test_ragc.py`.
Check exact NO-OP with disabled config or weight=0, and original
`LevelCostModel.get_cost` includes the original 0.2 action matching.
Report max |corrected-original| in standardized token space and in raw
environment action space, original vs new cost and added wall time.

## Stage 4: paired closed-loop evaluation
Use the same task IDs, seeds, 90-step budget, and model checkpoint.
Do not reuse an output directory containing existing chunks.
Compare ORIGINAL vs RAGC using successes gained/lost, total success,
runtime/FLOPs and additional GD calls. Fixed 5-step policy is a useful
separate performance reference, but NOT part of RAGC itself.

## Stop conditions
- No real counterfactual continuation improves original failures => stop.
- Validation preference <= random (or sign test non-improving) => stop.
- Corrected actions often exceed trust or original-cost guard => inspect.
- Held-out closed-loop wins fail to exceed losses => stop/rework.

## Source bugs to audit before relying on old data
- Original `scripts/action_ranking_test.py` zeroed `action_cost_weight=0.2`.
  Do NOT use this altered solver as data teacher.
- That script passed `steps_taken=0` at later decision points.
- In some revisions `action_response_data.DATASET_KEYS` omits
  `prefix_actions` even while collector outputs it; verify persisted H5.
- `_set_state(state7)` does NOT restore full Pymunk physical state.
- Goal overlay changes encoder output nontrivially; match real eval images.
