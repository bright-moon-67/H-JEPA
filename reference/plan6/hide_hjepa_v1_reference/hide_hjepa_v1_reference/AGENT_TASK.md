# Execute HiDe-HJEPA V1: Agent implementation and GO/NO-GO contract

**GOAL:** increase *real closed-loop AntMaze L3 success*, not a proxy
metric, on fixed 12 tasks x 2 deterministic settle seeds.

**Source:** `research/ant-trace-audit-v1` (48ef1c2), CLEAN branch
`research/ant-hide-decouple-v1`. Never develop on PAHRL branch; never mix
with CLFB/RAGC/Feedback-N.

## Mandatory order (STOP immediately on failed gate)

### I. Repository contract, source patch, and no-op

1. Verify exact source commit, clean `git status`, original 17/24 baseline
   manifest, checkpoint and normalizer hashes.
2. Install reference modules, config and strict patch via `apply_hide.py`.
   If text anchors differ, inspect the actual source and adapt only
   semantically equivalent statements; never force fuzzy replacement.
3. Check that `HierarchicalSolver.solve()` tags navigation ONLY when:
     - level3 is active (planning_horizon > 1), or
     - level2 follows a REAL predicted L3 goal.
   L1 MUST remain untouched, L3 skip means L2 does not get nav cost.
4. Verify `LevelCostModel.get_cost()` uses ORIGINAL
   `_project_to_goal_space()`, original criterion, original action penalty,
   THEN navigation correction. Prediction/goal time dims must be aligned.
5. Run 19+ CPU tests and full repo tests.
6. Run 2-task deterministic 22-step baseline vs default-off and
   explicit `hide_nav.enabled=false`. Compare ALL actions, cost records,
   subgoals, planning and environment step hashes. Timings may differ.
   Original planner seed, Batch size and settle seeds MUST match.
   Fail -> fix and repeat, no algorithm run.

### II. Live-domain TRAIN state extraction and XY head

7. Run `export_hide_live_train.py`, restoring TRAIN HDF5 qpos/qvel into
   the genuine `MegaWrapper->SyncWorld->VariationWrapper` environment
   and re-rendering with `get_ob`. DON'T encode HDF5 pixels directly.
8. For 32 training states confirm exact reset->set_state->qpos/qvel
   replay; confirm state encoder and GOAL encoder give identical L3
   outputs for identical live pixels+proprio. If fail -> NO-GO.
9. Train one 384→128→128→2 head using 30 epochs and episode-disjoint
   train/val/test. Training labels may be TRUE XY from TRAIN only;
   online planner cannot read true XY.
10. If heldout real-state test median error >=goal radius or P90 >=2
    radii -> NO-GO. Do not tune on development success labels.

### III. Predicted-L3 source and mandatory audit (most important)

11. Extract >=32 *genuine* frozen L3 `rollout` prediction pairs, from
    the `test_episodes` stored in head file, with the correct temporal
    stride and **true recorded L3 macro-actions**. Use the existing
    H-JEPA encoder/action_encoder machinery, not randomly generated
    latent samples. Include >=12 physically nontrivial movements.
12. Save the `start_z, pred_z, true_end_z, true_end_xy, episode` NPZ
    contract in README; report action-window indices, time alignment,
    pre/post-predictor shapes and checkpoint hashes.
13. Run `audit_hide_predictions.py`. If predicted latent XY error
    violates the frozen radius thresholds or motion direction cosine
    <0.5, NO-GO: report predictor-space off-manifold failure.
    Never bypass this audit or pretend real-state accuracy suffices.

### IV. Frozen TRAIN cost scales and GPU smoke

14. Run `collect_hide_cost_samples.py` and `calibrate_hide_scales.py`.
    If API mismatch, validate against actual GD action priors,
    `_project_to_goal_space`, `criterion` contracts, not dummy values.
15. Ensure the calibrated scales and α2=α3=0.25 are fixed BEFORE
    seeing the 24 closed-loop outcomes. No grid-search.
16. GPU smoke with 2 tasks and a few planning calls. Confirm:
    - same checkpoint & normalizer hashes, `head.eval()` and
      all `head.parameters.requires_grad == False`
    - L3 loss `[B,400]`, L2 loss `[B,100]` in ORIGINAL sample configs
    - projected L2 predictions/targets both L3 `[*,384]`
    - `cost_last_n=1` on L2/L3; original criterion shape/time intact
    - no NaN/Inf and nonzero action gradients through frozen head
    - no new graph/goal-oracle/true XY or H-JEPA model weight updates
    - goal state not overwritten; L3→L2 still FULL latent
    - measured navigation-vs-base cost and chosen candidate XY changes
    - compare original actual wall clock; overhead gate<=1.5x.

### V. One fixed paired development pilot ONLY after every gate above

17. Run exactly existing 12 dev tasks × 2 initializations (1000, 2000);
    original planner seed, worker/chunk/batch mappings, 350-step budget
    and 10-step action execution. Never vary alpha based on outcomes.
18. Use `compare_hide_dev.py`, report 17/24 baseline vs HiDe successes,
    per-task rescue/destroy/net and actual extra planning wallclock.
19. Engineering GO = net>=+2, destroyed<=1, overhead<=1.5x.
    Otherwise NO-GO. No hyperparameter sweep, no official 50.
20. If GO, run once on NEW independent tasks, then decide if to evaluate
    official 50. This is a separate validation, not tuning.

## Required artifacts

- code diff and one clean Git commit
- source model/normalizer/data/head/audit/scales SHA hashes
- live render replay & state/goal parity proof
- episode-split train/val/test XY regression errors
- true recorded L3 macroaction prediction pairs metadata
- prediction-audit true-vs-predicted XY errors, direction cosine, time indices
- base/nav training cost scale medians and output file
- default-off/full no-op trace hashes and L2/L3 GD gradient test
- paired 24-episode success table, overhead, GO/NO-GO

## Explicitly prohibited

- No true online XY/BFS/maze map as control inputs.
- No treating a random predicted latent as a true future prediction.
- No reusing PAHRL's HDF5-encoded (mismatched) `states_train.npz`.
- No changing JEPA dynamics, normalization, gradient solver n_steps,
  num_samples, original Horizon Schedule, L1 or execution cadence.
- No success-rate proxy substituted for actual environment success.
- No indefinite tuning when the first fixed V1 fails.
