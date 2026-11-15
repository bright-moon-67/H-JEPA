# CLFB-HJEPA V1 — Reference implementation

Target: `bright-moon-67/H-JEPA`, branch `research/ant-trace-audit-v1`,
base commit `48ef1c2`. **Reference code, NOT yet validated with an AntMaze GPU checkpoint.**

## Contents

- `h_jepa/cross_level_beam.py`: genuine L3 top-M latent rollout diversity,
  4-beam, isolated 16×30 L2 feasibility optimizer, rank selection, decision logs.
- `patch_spec.json`: nine exact-match source edits for the audited H-JEPA version.
- `apply_clfb.py`: preflight patch installer; aborts without writing on version mismatch.
- `h_jepa/tests/test_cross_level_beam.py`: CPU unit tests.

**Design constraints**:
1. Default off / beam_width=1: original solver code path and output unchanged.
2. Top-M are the original *optimized* L3 actions, not newly sampled actions.
3. Save L3 action and corresponding predicted latent **together**.
4. Use L2 `LevelCostModel.get_cost`, including real projection/criterion,
   `cost_last_n`, intermediate weight, original optimizer class and LR.
5. The probe never invokes `live_l2.solve()`, alters `live_l2.init`,
   `live_l2.torch_gen`, or runs a new world-model training step.
6. No true simulator reward, maze BFS, XY oracle, or RAGC Head.
7. No extra L2 probe when L3 or L2 is skipped by horizon schedule.

## Install (AI Agent)

Create a fresh branch from the **exact** base commit.

```bash
cd /root/rivermind-data/H-JEPA
git switch research/ant-trace-audit-v1
git switch -c research/ant-clfb-v1

# Extract the ZIP to a TEMP location, e.g. /tmp/clfb_v1_reference
cp /tmp/clfb_v1_reference/h_jepa/cross_level_beam.py h_jepa/
cp /tmp/clfb_v1_reference/h_jepa/tests/test_cross_level_beam.py h_jepa/tests/

python /tmp/clfb_v1_reference/apply_clfb.py "$PWD"
cd h_jepa
PYTHONPATH=. python -m pytest tests/test_cross_level_beam.py -q
PYTHONPATH=. python -m pytest tests -q
```

The patcher intentionally *refuses* to patch if the source does not match.
It is not idempotent: run only once on a fresh branch.

## Config

Do **not** modify the original `ant_l3.yaml` in-place.
`clfb` is optional and defaults to absent/off.

Activation using Hydra CLI (add keys, because baseline config has no `clfb`):

```
+clfb.enabled=true
+clfb.top_m=32
+clfb.beam_width=4
+clfb.probe_samples=16
+clfb.probe_steps=30
+clfb.seed=420042
```

The original 400 candidate L3 GD iterations remain unchanged.
The original 10-step execution cadence, 350-step budget, L1/L2 solver settings,
and env seeds remain unchanged.

## Before paired pilot: mandatory no-op / micro-gates

1. Compare *base repository* vs patched `clfb` ABSENT; action arrays, cost
   trajectories, per-episode successes, planner RNG states must match.
2. Compare `+clfb.enabled=true +clfb.beam_width=1` vs baseline;
   solver outputs must match exactly.
3. In CLFB-enabled runs confirm every L3 `candidate_pool.actions[:,0]` equals
   original `result['actions']`.
4. Verify that after CLFB switches L3 candidate, both L3 action and L3
   `predicted_embed_0` correspond to its rollout; inspect L2 target latent/action
   source and time index.
5. Verify the live L2 torch generator and cached `.init` were not touched by
   probes; the L2 cost model's own optimizer GD sees no extra randomness.
6. Validate real model cost shapes and finite outputs for 1–2 real tasks.
7. If model `rollout` has stochastic state mutation, evaluate its effects before
   using this code. Model must be in `eval()` mode.
8. Ensure source commit and env settle-state mappings agree with your existing
   17/24 deterministic dev baseline.

If a gate fails: **stop**. Do not proceed to 24-episode comparison.

## Paired AntMaze development run (example)

From `H-JEPA/h_jepa` (replace `<ANT_L3_CKPT>` with the actual checkpoint path):

```bash
export HJEPA_HOME=/root/rivermind-data/hjepa_home
export PYTHONPATH=.
export ANT_CKPT='<ANT_L3_CKPT>'
DEV='audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt'

python eval.py --config-name ant_l3 \
  policy="$ANT_CKPT" \
  load_eval_trajs_path="$DEV" \
  eval.num_eval=12 \
  +eval.env_seed=1000 \
  +eval.chunk_size=null \
  solver.solvers.level1.batch_size=2 \
  solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 \
  +clfb.enabled=true \
  +clfb.top_m=32 \
  +clfb.beam_width=4 \
  +clfb.probe_samples=16 \
  +clfb.probe_steps=30 \
  +clfb.seed=420042 \
  output.dir=eval_clfb_dev12_env1000
```

Repeat using `+eval.env_seed=2000` in a separate output directory. Match the
**actual** previous baseline's planner seed, batch grouping, SDPA backend and
task order; do not compare against a differently grouped run.

Expected files: `metrics.yaml`, `planning_compute.json`, `clfb_records.json`.
In chunked mode, CLFB records are stored in each `chunks/tasks_*.json`.

Do not compare old 38/50 (unseeded) with the new deterministic protocol.

## Gate

Known baseline: 17/24 on the 12 task × 2 settle-seed development set.
GO only if:
- net closed-loop gain >= 2;
- original successes destroyed <= 1;
- measured planner overhead <= 1.5x.

Otherwise NO-GO. No parameter sweep and no automatic official 50-task eval.

## Implementation caveats

- This implementation probes L2 feasibility using a separate action tensor,
  AdamW (or whatever original L2 optimizer class was selected), independent RNG,
  and `torch.autograd.grad` to avoid accumulating World-Model parameter grads.
  It still uses the **same frozen L2/upper models**.
- For speed and exact pairwise comparability, all four probes use common random
  numbers within each batch; the separate probe seed is independent of the
  live solver's RNG.
- Greedy farthest-first uses cosine distance over full predicted latent
  trajectories (context frame excluded), not privileged environment XY.
- The L3 ranking uses **rank within Top-M**, the L2 ranking uses rank within
  the K-beam; the two normalized ranks each carry weight 0.5.
- The feasibility estimate is a *model proxy*, not realized success.
  This is precisely the hypothesis the closed-loop pilot will test.
- `planning_compute.json` counts the original `GradientSolver.solve` calls.
  It does **not** count internal CLFB probe optimizer iterations; use
  `clfb_records.json`'s `probe_and_select_wall_s` and total evaluation time,
  plus a profiler if you require exact FLOPs.
- The check `pool[:,0] == original` is exact by design; if it fails due to
  unexpected source changes, investigate, do not weaken it.
