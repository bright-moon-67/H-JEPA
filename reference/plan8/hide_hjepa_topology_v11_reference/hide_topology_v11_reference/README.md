# HiDe-H-JEPA V1.1 — *real-observation* task-space topology

Target: AntMaze L3, pretrained and **frozen** world model, paired success rate.
Goal: rescue long-route errors without optimizing the unfaithful L3
100-frame predicted XY. This is a deliberately separate V1.1 candidate, NOT
a verified improvement or a faithful HiDe/PAHRL reproduction.

**Code status**: 100% standalone reference core, CPU-tested. Actual branch
`research/ant-hide-decouple-v1` commit `4872a40` was NOT publicly retrievable
at creation time; therefore no guessed exact-source automatic patch is provided.
Agent must adapt goal adapter to its real `hierarchical_solver.py` and run
GPU/checkpoint integration tests before any pilot.

## Why this instead of changing cost_last_n?

`ant_l3.yaml`: level3 horizon=5, cost_last_n=1, samples=400, iterations=90.
The 3-level model is trained with per-level one-step rollout_n=1; L3 stride
is 2x2 over level1 5-frame tokens. Existing HiDe B3 found at H=5:
decoded-XY error 1.909 vs no-motion 1.882 (not a useful navigation reward).
The positive B2 finding is **readout from REAL encoded observations**:
test median 0.270, p90 0.662.

This prototype reads XY only from `readout(E3(REAL current image))` and from
`readout(E3(REAL goal image))`. It never calls the XY head on a predicted
L3 rollout. An offline train-only traversal graph chooses a near-term route
waypoint; its *LIVE-domain* train embedding (all levels) becomes a temporary
goal for the original untouched H-JEPA L3/L2/L1 planners.

**Caveat on fairness**: this reference graph builder consumes training-set
physical XY as OFFLINE supervision to construct a traversal topology. This
is additional task-space/map supervision and MUST be disclosed when reporting
results. It is NOT purely self-supervised JEPA navigation. Online evaluation
MUST NOT use simulator XY, BFS, occupancy-grid oracle, or episode labels.
If your scientific protocol forbids even offline XY supervision, this specific
graph builder is out of scope; use decoder-produced train XY and perform an
extra explicit false-edge audit instead.

## Contents

```
topo_graph.py          : observed directed transitions -> grid-cell graph
goal_adapter.py        : default-off local waypoint -> all 3 H-JEPA goal embeds
build_topology.py      : strict train-only two-export build CLI
tests/test_topo_graph.py
README.md
AGENT_TASK.md
```

### Data contract

1. `dense_train_xy.npz` — ALL TRAIN data, frames sorted by episode then time:
   `xy[N,2]`, `episode[N]`, `step[N]`, `metadata` JSON serialized scalar;
   `metadata={"source_split":"train","dataset_name":"..."}`
   Use dense **real consecutive** frame transitions, not every-10-frame skips.
   NO dev/official test data is allowed.
2. `live_train_states.npz` — TRAIN live-rendered states, from the B1 export:
   `xy[M,2]`, `embed_1`, `embed_2`, `embed_3`, `metadata`.
   `metadata` must also contain `source_split`, `dataset_name`, and
   `checkpoint_sha256` / `normalizer_sha256`.
   Embeddings MUST be from exact eval preprocessing + exact goal/state
   normalization, not offline HDF5 image domain; B1 parity must be rechecked.
   Each map-cell representative includes real *all-level* goal embeddings.
3. The local controller map-certification record must be grounded in
   train-only held-out real AntMaze waypoints; set
   `AdapterConfig.certification_evidence_id` ONLY after that audit.
   The reference library checks that the field is present, not its truth;
   the integrating agent must verify a real evidence file and hash.

### Build train topology

```
cd h_jepa
PYTHONPATH=/path/to/hide_topology_v11_reference python \
    /path/to/hide_topology_v11_reference/build_topology.py \
    --dense-train-xy artifacts/hide/dense_train_xy.npz \
    --live-train-states artifacts/hide/live_train_states.npz \
    --output artifacts/hide/topology_train_live.npz \
    --grid-width 1.0 --min-edge-support 1
```

`1.0` is an **illustrative default**, NOT a validated cell width.
Choose ONE width based on TRAIN wall clearance / XY decoder p90 localization
error and freeze it BEFORE any dev task success analysis. Do NOT grid-search.

Every directed edge is a directly observed consecutive TRAIN XY frame transition
between two 4-connected cells, rejecting diagonals, jumps and episode boundary
links. Different episodes are pooled only by belonging to the same grid cell;
**this pooling can still join opposite sides of a thin wall**. Use the full
observed position trace and wall-check diagnostics to catch false shared-cell
connectivity. If ambiguous, keep cells split by traversal component and require
independent local-controller proof before merging. No kNN latent edges.

### Integrate to original solver

The public main code computes:
```
obs_embeddings = self._encode_all_levels(info_dict=info_dict,key='pixels')
goal_embeddings = self._encode_all_levels(info_dict=info_dict,key='goal')
for level in range(len(self.level_models),0,-1):
    ...
```

In the experimental branch, after the TWO original encodings and before the
`for level` loop:

```
if (
    self.hide_topology_adapter is not None
    and self.hide_topology_adapter.config.enabled
    and len(self.level_models) == 3
    and horizon_schedules[3][call_idx] > 1
):
    with torch.no_grad():
        live_xy = self.hide_readout(
            obs_embeddings["embed_3"].squeeze(1)
        ).detach().cpu().numpy()
        goal_xy = self.hide_readout(
            goal_embeddings["embed_3"].squeeze(1)
        ).detach().cpu().numpy()

    replacement, override_mask = self.hide_topology_adapter.choose(
        live_xy, goal_xy, goal_embeddings, planning_call=call_idx
    )
    goal_embeddings = {
        k: torch.where(
            override_mask[:,None,None], replacement[k], goal_embeddings[k]
        ) for k in ("embed_1","embed_2","embed_3")
    }
```

This is **integration pseudocode**: check the actual readout method, tensor
shape, normalization and goal state encoder parity in your `4872a40` branch.
Do not guess symbols. `goal_embeddings` MUST replace ALL levels so original
L3->L2->L1 anchoring always refers to the same active waypoint, not one L3
temporary target mixed with the real lower-level goal. Keep the original
`info_dict["goal"]` and environment success goal unchanged.

At L3 horizon=1, preserve original goal, skip adapter. Do not lower execution
interval (original 10 steps), add optimizer candidates, modify 400x90 GD or
change the checkpoint. Original checkpoint `normalizer.pt` hashes must match.
Reset the controller per environment/task and chunk (including early success).
Inspect `hierarchical_solver.py` multi-env state isolation.

The adapter intentionally persists a waypoint until the REAL decoded current
state reaches it, or a fixed call timeout triggers a permanent task fallback.
No single-frame reward or true XY can influence selection online.

## Gates BEFORE 24-episode paired dev pilot

**Gate 0 — independent function audit**
- No-op disabled/absent bit-for-bit same raw steps and planner records.
- All 3 goal levels replaced consistently, no L3 horizon-one leakage.
- Same checkpoint, normalizer, original 10-step execution and RNG.
- CPU tests: `PYTHONPATH=. pytest -q tests/test_topo_graph.py`.
- Report training-only graph node/edge evidence and discarded jumps.

**Gate 1 — graph geometry / localization**
- On TRAIN-only held-out states, prove decoded XY localization sufficient near
  narrow corridors and walls, not just median error.
- Inspect false cell-merges and any path across a wall; any false shortcut
  is NO-GO pending a single legitimate fix in train map construction.
- Graph routes for development start/goal images must be generated from
  the online **decoded** XY; use this for *coverage only*.
- Prespecify a minimum 12/24 valid routes with plausible intermediate LIVE
  representatives. If lower, NO-GO. No threshold sweep on dev outcomes.

**Gate 2 — real waypoint executability**
- On 12 disjoint TRAIN-only starts, attempt one frozen-HJEPA waypoint each
  under exact `MegaWrapper -> SyncWorld -> VariationWrapper` deterministic
  reset/settle protocol. 20-40 environment frame local waypoint targets,
  original action and replanning schedule; use real XY only for scoring.
- Engineering threshold: at least 9/12 reached, 0 false-arrival decisions
  (arrived according to decoded XY but not physical success), and no
  deterministic run-to-run drift. If FAIL, NO-GO.
- Certify `attach_radius`, `attach_margin`, `arrival_radius`, lookahead and
  a bounded timeout from TRAIN-only calibration (one frozen choice).
  1.0 / 0.8 in `AdapterConfig` are illustrative and unvalidated.

**Gate 3 — closed-loop pilot**
- 12 original dev tasks × settle seed 1000,2000; n=24 episode trials from
  12 navigation tasks, paired against 17/24 deterministic baseline.
- Keep exactly original checkpoint, planner seed, batch groupings, 350 steps,
  L3 400 samples × 90 iters, L2/L1 configs, action horizon, output collection.
- GO only if NET>=+2, destroyed original successes<=1,
  real total planning wall-clock overhead<=1.5x, repeatable paired seeds.
  Report goal overrides, waypoint reached/timeouts, route coverage, success
  per unique task and both settle seeds. No sweep on same 12 dev tasks.
- If GO, run NEW held-out tasks, not official original 50, before final eval.

**Interpretation**: if topology helps, compare against original H-JEPA and
against a planner with the same offline supervised XY data (e.g. a simple
coordinate-map waypoint controller) before claiming a new H-JEPA method.
Never claim this is *full HiDe* or *PAHRL* reproduction, or that a CPU test
proves any success-rate gain.

## Non-goals
- No gradient loss on `P(z_hat_{t+5})`.
- No automatic shortening of L3 horizon 5 (a different algorithm).
- No change `cost_last_n=1` to 2 claiming this scores near-term states;
  it actually scores the *last two*.
- No CLFB, RAGC, Feedback-5 or PAHRL V1 latent-kNN graph in this branch.
- No tuning against isolated dev task 04 or 09.
