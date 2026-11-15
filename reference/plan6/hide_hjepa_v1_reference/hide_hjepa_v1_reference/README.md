# HiDe-HJEPA V1 — Functional Navigation / Control Decoupling

**Reference code, NOT a GPU-validated or proven success-rate improvement.**

Repo: https://github.com/bright-moon-67/H-JEPA  
**Base:** `research/ant-trace-audit-v1` (commit `48ef1c2`, validate with `git rev-parse HEAD`).  
**New branch:** `research/ant-hide-decouple-v1`.

This is **HiDe-inspired** functional decomposition, NOT a reproduction
of HiDe's original high-level MVProp planner / actor-critic policy.
H-JEPA remains a *frozen differentiable world model + gradient planner*.

**Research objective:** improve the real 12-task x 2-settle-seed
deterministic AntMaze L3 closed-loop baseline **17/24**. No trick success
metrics. No official 50 tasks while developing.

## Core method

- L3: `original_cost + 0.25 * train_calibrated_scale_l3 * nav_xy_error`.
- L2: same only when it follows an ACTUAL L3 subgoal, after its original
  `_project_to_goal_space` projection, comparing matching L3 latents.
- L1: original planner untouched.
- Encoder/predictor/action models: all original parameters frozen.
- L3 -> L2: **full latent** subgoals unchanged; no privileged true XY.
- Navigation XY head: trained on **TRAIN only, LIVE-rendered** L3 observations,
  with training-only XY supervision; inference gets ONLY latent z.
- Original L3/L2 criteria and action costs remain, reducing but not
  eliminating the risk of false straight-through-wall XY preferences.
- Cost scales computed from original frozen TRAIN candidate rollouts, NOT
  success labels from the development set.
- No extra replanning frequency, optimizer step changes, CLFB, PAHRL, RAGC.

## Files

```
hide_hjepa_v1_reference/
  README.md
  AGENT_TASK.md
  apply_hide.py
  patch_spec.json
  h_jepa/
    hide_nav.py                     # head + differentiable L3/L2 navigation cost
    hide_integration.py             # model/data SHA, train-only audit, installation
    config/eval/ant_l3_hide.yaml    # Hydra inherits original ant_l3
    scripts/
      export_hide_live_train.py     # reproduces LIVE World render domain
      train_hide_nav.py             # 30-epoch fixed readout, episode-disjoint splits
      audit_hide_predictions.py     # mandatory held-out predictor test
      collect_hide_cost_samples.py  # frozen TRAIN paired base/nav costs
      calibrate_hide_scales.py      # median normalization, no dev tuning
      compare_hide_dev.py          # strict seed-paired 12x2 results
    tests/
      test_hide_nav.py
      test_hide_install.py
      test_hide_compare.py
```

## A. Integrate in a CLEAN branch (no PAHRL/CLFB changes)

```bash
cd /root/rivermind-data/H-JEPA
git status --short                       # must be clean
git switch research/ant-trace-audit-v1
git switch -c research/ant-hide-decouple-v1

unzip /path/to/hide_hjepa_v1_reference.zip -d /tmp
SRC=/tmp/hide_hjepa_v1_reference
cp "$SRC/h_jepa/"hide_*.py h_jepa/
cp "$SRC/h_jepa/scripts/"*.py h_jepa/scripts/
cp "$SRC/h_jepa/tests/"*.py h_jepa/tests/
cp "$SRC/h_jepa/config/eval/ant_l3_hide.yaml" h_jepa/config/eval/
python "$SRC/apply_hide.py" "$PWD"
cd h_jepa
PYTHONPATH=. python -m pytest tests/test_hide_nav.py \
  tests/test_hide_install.py tests/test_hide_compare.py -q
PYTHONPATH=. python -m pytest tests -q
```

`apply_hide.py` is deliberately strict and not idempotent: if any anchor
doesn't match exactly, it refuses to write ANY file. Inspect actual source
and adapt the smallest anchors, do not force broad replacements. The only
modified existing files are `h_jepa/hierarchical_solver.py` and
`h_jepa/planning_eval.py`, both default OFF.

## B. Frozen model / training-domain provenance

```bash
export HJEPA_HOME=/root/rivermind-data/hjepa_home
export PYTHONPATH=.
export MUJOCO_GL=egl
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TRAIN="$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5"
```

**Do not reuse PAHRL V1's raw-HDF5 `states_train.npz`:** a severe offline/live
L3 latent mismatch was observed (median cosine distance ~0.441).

### B1. Export LIVE-domain states from TRAIN physical states

```bash
python scripts/export_hide_live_train.py \
  --checkpoint "$CKPT" --dataset "$TRAIN" \
  --output artifacts/hide/live_train_states.npz \
  --max-episodes 150 --every 10 --batch-size 16
```

For each TRAIN qpos/qvel state, the script restores state using the genuine
World wrapper, renders `get_ob()` locally, then uses
`WorldModelPolicy._prepare_info -> HierarchicalSolver._encode_all_levels`.
It stores `[N,384]` L3, `[N,D2]` L2, TRAIN-only `xy`, episode IDs and hashes.
It checks **state/goal encoding parity on IDENTICAL pixels+proprio**.
It does NOT run in the online eval planner path.

This is a reference implementation; verify `set_state` replay, env `get_ob`,
image shape, `goal_proprio` processing, and L3 parity on YOUR GPU first.
The script hashes the entire 22GB training dataset; budget for this I/O.

### B2. Train ONE projector, frozen JEPA

```bash
python scripts/train_hide_nav.py \
  --input artifacts/hide/live_train_states.npz \
  --output artifacts/hide/nav_projector.pt \
  --epochs 30 --batch-size 256 --seed 42 --goal-radius 1.0
```

- 384 → 128 → 128 → 2 MLP, ReLU, fixed AdamW.
- Train/validation/test split by TRAIN episode, not by nearby frames.
- Hold-out real-state gate: median XY error <1.0; P90 error <2.0.
- Do not adjust architecture/epochs on the AntMaze dev tasks.
- Output: `nav_projector.pt` + `nav_projector.train_metrics.json`.
- If gate fails, NO-GO. Don't proceed to GD integration.

### B3. MANDATORY predictor-latent audit

A readout of encoded real frames is NOT necessarily valid on `L3.rollout`'s
predictions. A GradientSolver can exploit a projector off the training
manifold, creating fake navigation progress.

Create `artifacts/hide/prediction_pairs_train_heldout.npz` from the
**test_episodes listed inside nav_projector.pt metadata** with:

```
start_z[N,384]
pred_z[N,384]
true_end_z[N,384]
true_end_xy[N,2]
episode[N]
metadata = json.dumps({
  "source_split":"train_heldout",
  "predictor_source":"frozen_l3_true_macro_action_rollout",
  "checkpoint_sha256":"<same SHA as projector>",
  "time_scale_verified":True
})
```

These MUST come from original frozen **L3 rollout of true recorded
L3 macro-actions**, with correct action encoder + temporal stride and
matching real TRAIN future state. **Do not use random latent vectors,
linear interpolation, or random GD candidates as ground truth.**
Implementation of this task-specific trajectory/action alignment is
intentionally left for your repo agent, which can inspect the checkpoint
and dataset loaders directly. Do not claim it is already generated here.

```bash
python scripts/audit_hide_predictions.py \
  --pairs artifacts/hide/prediction_pairs_train_heldout.npz \
  --head artifacts/hide/nav_projector.pt \
  --output artifacts/hide/prediction_audit.json
```

Gate: >=32 actual heldout pairs, >=12 nontrivial movements,
median predicted XY error <1.0, P90 <2.0, mean motion-direction
cosine >=0.5, head SHA and original checkpoint SHA match.
These are PRESET screening criteria, not proven sufficient for
closed-loop safety.

If fail, NO-GO. Do not optimize candidate actions using this readout.

### B4. Cost magnitude calibration from TRAIN rollouts

```bash
python scripts/collect_hide_cost_samples.py \
  --states artifacts/hide/live_train_states.npz \
  --checkpoint "$CKPT" \
  --head artifacts/hide/nav_projector.pt \
  --output artifacts/hide/cost_samples_train.npz \
  --num-pairs 64 --candidates 4

python scripts/calibrate_hide_scales.py \
  --samples artifacts/hide/cost_samples_train.npz \
  --projector artifacts/hide/nav_projector.pt \
  --output artifacts/hide/cost_scales.json
```

Calibration uses SAME candidate latent actions to compute original
LevelCostModel cost and new XY navigation loss (at the SAME L3 goal
space/time index). Scale is fixed `median(base) / median(nav)` on TRAIN
rollouts for L3 and L2 independently. No use of dev success results.

Source-model smoke may require adjustment if your installed model's
latent action queue or action encoder semantics differ. If it's invalid,
fix the API contract — don't fabricate scales.

## C. Enabling HiDe: only after ALL offline gates pass

Hydra profile `ant_l3_hide` inherits the original `ant_l3` solver config.
Use no optimization changes except the explicitly added cost:

```bash
TASKS=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt

python eval.py --config-name ant_l3_hide \
  policy="$CKPT" seed=42 \
  load_eval_trajs_path="$TASKS" \
  eval.num_eval=12 eval.eval_budget=350 \
  +eval.chunk_size=null +eval.env_seed=1000 \
  solver.solvers.level1.batch_size=2 \
  solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 \
  hide_nav.enabled=true \
  hide_nav.projector_path="$PWD/artifacts/hide/nav_projector.pt" \
  hide_nav.prediction_audit_path="$PWD/artifacts/hide/prediction_audit.json" \
  hide_nav.scale_stats_path="$PWD/artifacts/hide/cost_scales.json" \
  output.dir=outputs/hide_dev1000
```

Repeat with `env_seed=2000`, separate output directory. **FIRST** perform
2-task 22-step disabled/explicit-disabled/no-op trace smoke and compare all
actual actions and costs. Use the exact original baseline worker/BATCH
mapping, planner seed, Settle seed, SDPA backend, 10-step execution,
350-step budget. The CPU tests do not prove this GPU no-op gate.

A bad / stale / missing projector, scale or audit file raises instead
of quietly falling back to arbitrary values.

Do not optimize any of `alpha_l3=0.25` or `alpha_l2=0.25` on dev tasks.
Do not simultaneously enable PAHRL, CLFB, RAGC or Feedback-N.

## D. One bounded paired 12x2 development pilot

The known deterministic baseline is:
- `env_seed=1000`: **8/12**
- `env_seed=2000`: **9/12**
- Total **17/24** (12 tasks with two variants of physical initialization).

Compare if available:
```bash
python scripts/compare_hide_dev.py \
  --baseline-1000 /path/to/baseline_env1000 \
  --hide-1000 /path/to/hide_dev1000 \
  --baseline-2000 /path/to/baseline_env2000 \
  --hide-2000 /path/to/hide_dev2000
```

GO **only if** net gain >=2, destroyed <=1, and measured extra planning
runtime <=1.5x. This is an engineering filter, NOT statistical proof.
If GO, next evaluate on NEW independent tasks before official 50.
If NO-GO, do NOT sweep alphas or train many heads on the same dev tasks.

Required report: per-task successes, rescue/destroy, relative wallclock,
actual XY progress and collisions (offline diagnostics only), gradient
finite/scale checks, actual goal/subgoal shape/time alignments, hashes,
changeset commit and explicit GO/NO-GO.

## Distinct method hazards

1. **Does not know maze topology.** XY loss alone can bias the solver to
   walk through walls. The unchanged JEPA cost is retained, but might not
   suffice. A NO-GO would be a useful result.
2. **Real vs predicted latent domain.** Offline real-state accuracy alone
   is insufficient. Predictive audit is mandatory.
3. **No XY inference cheat.** XY labels may train the head on TRAIN episodes,
   but NO true env XY is read at evaluation time for planning.
4. **Not a HiDe reproduction.** No MVProp, learned attention, actor-critic
   high/low policies. Cite "HiDe-inspired functional separation".
5. **The complete source patch has not been tested on your GPU.**
   The authoring environment ran CPU tests only; the repo agent must test
   original Checkpoint, normalizer, source anchors and full trajectories.

## Scientific citation

HiDe: *Learning Functionally Decomposed Hierarchies for Continuous Control
Tasks With Path Planning*. Adaptation is *not* a reproduction of its full
implementation; only its functional state/goal separation is used.
