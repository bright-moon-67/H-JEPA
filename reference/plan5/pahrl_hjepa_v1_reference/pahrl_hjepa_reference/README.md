# PAHRL-inspired Reachability Graph Guidance for Frozen H-JEPA L3

**Reference implementation / engineering starter — NOT a validated GPU integration.**

Target: public `bright-moon-67/H-JEPA`, branch `research/ant-trace-audit-v1`,
base commit `48ef1c2`. Create a separate `research/ant-pahrl-graph-v1`.
Do not mix with `main`, CLFB, RAGC, ARN, adaptive replanning or Feedback-5.

## What is — and is not — reproduced from PAHRL?

The uploaded paper *Planning-Augmented Hierarchical Reinforcement Learning*
(Gieselmann & Pokorny, RA-L 2021) uses **distributional hierarchical Q-values
as temporal-distance estimates**, builds a state-memory graph, searches by
Dijkstra, and feeds graph waypoints to learned local goal-conditioned policies.

This package is **PAHRL-inspired**, not a faithful PAHRL reproduction:
- real TRAIN observations -> frozen H-JEPA **level-3 embeddings**;
- observed TRAIN trajectory transitions -> **directed** state-memory graph;
- separately validated H-JEPA local-controller transitions -> optional extra
  directed edges (no fabricated reverse/near-latent edges);
- Dijkstra over **observed elapsed steps** (not a learned Q distance);
- online nearest-latent start/goal attachments ONLY after separate local
  controller validation;
- graph supplies **temporary L3 goal latent** at the ORIGINAL planning calls;
  original L3 GD -> L2 GD -> L1 GD remains unchanged.
- no actor-critic training, no online XY/BFS/map oracle, no reward-label peeking.

If graph connectivity or local-controller calibration fails, STOP. Do not
disable the gates or pretend a small latent distance establishes reachability.

## Folder layout

```
pahrl_hjepa_reference/
  README.md
  AGENT_TASK.md
  apply_pahrl.py
  patch_spec.json
  h_jepa/
    pahrl_memory.py
    pahrl_waypoints.py
    scripts/
      export_pahrl_ant_states.py
      build_pahrl_graph.py
      propose_pahrl_links.py
      calibrate_pahrl_attachments.py
      audit_pahrl_graph.py
    tests/
      test_pahrl_memory.py
```

## Install in a CLEAN experimental branch

```bash
cd /root/rivermind-data/H-JEPA
git fetch origin
git switch research/ant-trace-audit-v1
git switch -c research/ant-pahrl-graph-v1
# Replace this path with where the ZIP was extracted:
SRC=/tmp/pahrl_hjepa_reference
cp "$SRC/h_jepa/pahrl_memory.py" h_jepa/
cp "$SRC/h_jepa/pahrl_waypoints.py" h_jepa/
cp "$SRC/h_jepa/scripts/"*.py h_jepa/scripts/
cp "$SRC/h_jepa/tests/test_pahrl_memory.py" h_jepa/tests/
cp "$SRC/h_jepa/tests/test_pahrl_calibration.py" h_jepa/tests/
python "$SRC/apply_pahrl.py" "$PWD" # strict source anchors, preflight
cd h_jepa
PYTHONPATH=. python -m pytest tests/test_pahrl_memory.py tests/test_pahrl_calibration.py -q
PYTHONPATH=. python -m pytest tests -q
```

`apply_pahrl.py` edits ONLY `h_jepa/hierarchical_solver.py` and
`h_jepa/planning_eval.py`. It refuses to patch source that differs from the
expected version. It is not idempotent; apply once. All new features default
OFF. Before experimenting, capture exact Git commit/checkpoint hash.

## STAGE 0: TRAIN-only memory export (GPU encoder, no environment rollouts)

```bash
cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME=/root/rivermind-data/hjepa_home
export PYTHONPATH=.
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"

python scripts/export_pahrl_ant_states.py \
  --checkpoint "$CKPT" \
  --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
  --output artifacts/pahrl/states_train.npz \
  --every 10 --max-episodes 300 --batch-size 32

python scripts/build_pahrl_graph.py \
  --states artifacts/pahrl/states_train.npz \
  --output artifacts/pahrl/graph_observed_only.npz \
  --max-observed-gap 10
```

Exporter calls EXACT eval preprocessing (`_prepare_info`) then
`_encode_all_levels` and records L3 `[N,384]`, source training episode, time,
SHA256 of checkpoint AND `normalizer.pt`. `xy_diagnostic_only` is kept in the
**state export**, NOT the deployable graph.

**Do not** export from `visual_antmaze_medium_stitch_val.h5`, the official
50-task pack, or the 12-task development results.

Check `graph_observed_only.summary.json` for connected components. It is
entirely possible that observed-only trajectories have very poor graph
connectivity; in that case, obtain additional *verified* cross-episode
transitions or STOP. Blind kNN connectivity is prohibited.

### Optional offline coverage audit

Agent must independently create `queries_dev.npz` containing
`start_z[N,D]` and `goal_z[N,D]`, from the 12 development task
start/goal observations using the exact same live encoding pipeline.
The dev task `.pt` contains **placeholder zero pixels**, not actual rendered
images! Use the deterministic `xy_neutral` World/settle/reset path, not the
zeros in `.pt`. Do not use XY as planner input.

```bash
python scripts/audit_pahrl_graph.py \
  --graph artifacts/pahrl/graph_observed_only.npz \
  --queries artifacts/pahrl/queries_dev.npz \
  --diagnostic-cosine-cutoff 0.05
```

This prints a **possible** route coverage count; it does NOT validate
attachment distance or certify route viability. It uses a declared diagnostic
cutoff only. Do not tune on the dev success labels.

STAGE-0 NO-GO if no meaningful directed routes or graph coverage is too poor.
Graph connectivity, query alignment and cross-wall false-match auditing must
be checked before modifying online policy behavior.

## STAGE 1: real local reachability and attachment calibration

This is the mandatory intervention test; requires YOUR original AntMaze
simulator/checkpoint. A general-purpose GPU rollout validator is deliberately
**not fabricated** in this package because the real project depends on the
`MegaWrapper -> SyncWorld -> VariationWrapper` seed/settle-state protocol.

1. Freeze L1/L2/L3, planners, normalization and `env_seed`.
2. Use TRAIN-only source rows from `states_train.npz` (source_row) to
   reconstruct full `qpos`, `qvel`, target observation, and identical
   physics state via the exact wrapper path (not bare `gym.make`).
3. Generate cross-episode proposals (NOT edges):
   ```
   python scripts/propose_pahrl_links.py \
     --states artifacts/pahrl/states_train.npz \
     --output artifacts/pahrl/unverified_links.jsonl
   ```
   The proposal script uses training XY **only to prioritize experiments**
   and never exposes it to the online selector.
4. For a proposed directed edge i->j, run >=3 controlled local H-JEPA
   reach trials from memorized source state to memorized target. Verify
   physical state replay, consistent goals, actual reaching, and
   source-episode separation. If 3/3 succeeded, write:
   ```
   {"from_node":12,"to_node":345,"successes":3,"trials":3,
    "mean_steps":18.7,"source_split":"train","evidence_sha256":"<rollout-log-sha256>"}
   ```
   to `validated_links.jsonl`. NEVER infer reverse reachability.
5. For ONLINE query attachments, perform >=8 held-out TRAIN-only local
   H-JEPA controller trials with ≥3 positives and ≥3 negatives, including
   difficult pairs across walls. Example JSONL trial:
   ```
   {"cosine_distance":0.024,"local_success":true,
    "source_split":"train","source_episode_uid":"train_17"}
   ```
   and a corresponding failure:
   ```
   {"cosine_distance":0.12,"local_success":false,
    "source_split":"train","source_episode_uid":"train_19"}
   ```
   Run one fixed conservative calibration, not a parameter grid:
   ```
   python scripts/calibrate_pahrl_attachments.py \
      --trials-jsonl artifacts/pahrl/attachment_trials.jsonl \
      --output artifacts/pahrl/attachment_certificate.json
   ```
   If successes/failures overlap in latent distance, this **refuses to
   certify online attachments** -> STOP. Even when it passes, the test is
   an engineering gate, not a statistical generalization guarantee.
6. Build final certified graph:
   ```
   python scripts/build_pahrl_graph.py \
     --states artifacts/pahrl/states_train.npz \
     --output artifacts/pahrl/graph_certified.npz \
     --validated-links-jsonl artifacts/pahrl/validated_links.jsonl \
     --attachment-certificate artifacts/pahrl/attachment_certificate.json
   ```

**WARNING:** A graph node can be a recorded expert/exploration state, yet the
frozen H-JEPA low-level controller may not be able to reach it from the query.
Local-controller evidence is fundamental, not an optional add-on.

## STAGE 2: tiny integration and strict no-op gate

Hydra CLI additional keys (when config has no `pahrl_graph` section):

```
+pahrl_graph.enabled=true
+pahrl_graph.graph_path=artifacts/pahrl/graph_certified.npz
+pahrl_graph.reach_cosine_distance=0.035
+pahrl_graph.k_attach=3
+pahrl_graph.max_waypoint_calls=6
```

`reach_cosine_distance` also needs a local-controller arrival calibration or
an independently specified engineering threshold; do not tune it on final
success metrics. All above parameters MUST be frozen before Stage 3.

At active L3 calls: the controller may replace `goal_embeddings['embed_3']`
with the graph waypoint embedding. This also updates the original L2
terminal anchoring through the existing goal_embeddings dictionary.
At L3 horizon=1, no graph goal is inserted; lower levels retain REAL goal.

Mandatory gates:
- `pahrl_graph` absent: every action/cost/seed same as original.
- `enabled=false`: same as original.
- uncertified or disconnected graph: fail closed / same original action.
- exact live online embedding == graph exported embedding for a matched
  stored training state, to numerical tolerance.
- same checkpoint & normalizer hashes.
- no graph module reads online `xy` or BFS.
- goal embedding and propagated last-anchored subgoal refer to SAME active
  target; no mismatch when L3 horizon=1.
- reset controller between tasks/chunks, no inter-environment leakage.
- use the established `env_seed = base + global_task_id` from 48ef1c2.
- match original planner seed, batch grouping, SDPA backend, 350 step budget.
- no extra replanning calls, and original L3/L2/L1 costs unchanged when off.

If any gate fails, STOP. Do not run the 24-episode pilot.

## STAGE 3: paired **closed-loop** development comparison

Known deterministic AntMaze L3 dev baseline: `8/12` for `env_seed=1000`
and `9/12` for `env_seed=2000`, total `17/24`.
These are 12 distinct tasks under two settle states, NOT 24 independent tasks.

Example (run from h_jepa; use actual known checkpoint):
```
python eval.py --config-name ant_l3 \
  policy="$CKPT" seed=42 \
  load_eval_trajs_path=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
  eval.num_eval=12 +eval.chunk_size=null +eval.env_seed=1000 \
  solver.solvers.level1.batch_size=2 \
  solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 \
  +pahrl_graph.enabled=true \
  +pahrl_graph.graph_path=artifacts/pahrl/graph_certified.npz \
  +pahrl_graph.reach_cosine_distance=0.035 \
  +pahrl_graph.k_attach=3 \
  +pahrl_graph.max_waypoint_calls=6 \
  output.dir=eval_dev12_env1000_pahrl
```
Repeat with env_seed=2000 in a **fresh** output directory. Match ORIGINAL
`batch_size=2` and per-worker/grouping setup from the baseline's manifest.
Do not accidentally use `chunk_size=1` because that changes RNG protocol.

Report:
- per-task baseline vs graph-guided success, rescue/destroy, net;
- number of actual goal overrides; graph route coverage and fallback reasons;
- waypoint reached vs timed out; false attachment/cross-wall errors;
- actual runtime/extra compute (graph search + original GD);
- full source checkpoint/graph/normalizer hashes.

**Engineering GO:** net >= +2 of 24, original successes destroyed <=1,
extra compute <=1.5x, correct deterministic pairing and task/seed isolation.
Otherwise NO-GO. No threshold or node-density search on the same dev set.
If GO, evaluate once on NEW independent tasks. Official 50 remain untouched
until a final confirmed, frozen algorithm version.

## Important limits (do not overclaim)

- We have not run this module on your GPU or official AntMaze checkpoint.
- CPU tests cover graph correctness, provenance safety, and controller logic,
  NOT H-JEPA's numerical outputs or real reachability.
- `GraphMemory` only encodes observed transitions + verified extra edges.
  It may be disconnected. That is a genuine method risk, not a coding bug.
- The controller uses *cosine* latent distance for query attachment and
  arrival detection ONLY after external calibration. This is still a proxy.
- The 0.035 arrival threshold is an engineering default, NOT learned from
  the PAHRL paper; validate separately and freeze before the dev pilot.
- Graph guidance changes TOP-LEVEL goal selection, not the original H-JEPA
  planner parameters. No Q-value temporal-distance network is trained.
  PAHRL's full method is different. Only claim "PAHRL-inspired".
- The paper also describes top-level implicit GOAL SETS. This package keeps
  the original goal image. Goal-set recognition should be a later, separate
  experiment only after graph routing improves real success.

Paper: Gieselmann & Pokorny, *Planning-Augmented Hierarchical Reinforcement
Learning*, IEEE Robotics and Automation Letters 6(3), 2021.
DOI: 10.1109/LRA.2021.3071062.
