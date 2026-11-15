# PAHRL-inspired graph guidance — Stage 0/1 audit (V1)

Branch: `research/ant-pahrl-graph-v1` (base `48ef1c2`)
Reference: `reference/plan5/pahrl_hjepa_v1_reference/pahrl_hjepa_reference`
Date: 2026-11-15

This document records the Stage-0 (TRAIN-only memory export + graph) and
Stage-1 (local reachability / attachment calibration) evidence produced by the
reference package, and the gate verdict. Nothing here changes the frozen
checkpoint, the 10-step execution cadence, or the original L3→L2→L1 planner.

## 0. Artifact provenance

| artifact | SHA256 |
|---|---|
| checkpoint `ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt` | `c55d9835d6c774da4363c45f8c390c8abed5a496282dd5b9815615e417f69f99` |
| `normalizer.pt` | `1e86ce0a72836c69455a73f614765b7c186bebeca0904be60a9db12ed889fb44` |
| `visual_antmaze_medium_explore_stitch_train.h5` (21.98 GB) | `75f583348e76546136e6c9d5caf7319df5dc55fdd4aee07c25a5b525e3357419` (official `SHA256SUMS`) |
| `artifacts/pahrl/states_train.npz` | `da92111e21f4eb79935a89edc0e96be002cbcc89df05dc37df44f280e30c0ad4` |
| `artifacts/pahrl/graph_observed_only.npz` | `bd7a163b729640869cd6ca968036dd67cd7491463683c7da566c05f01d340711` |
| `artifacts/pahrl/queries_dev.npz` | `a82b99bcac5e000d6485cb63c2a44e4443417ce51659e7717b4e401137764ad3` |

All graph inputs come from the TRAIN split (`--max-episodes 300` of 25,000
train episodes). No validation episodes and none of the official 50 tasks were
read.

## 1. Stage 0 — TRAIN-only memory export

```
python scripts/export_pahrl_ant_states.py \
  --checkpoint $HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
  --dataset $HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5 \
  --output artifacts/pahrl/states_train.npz \
  --every 10 --max-episodes 300 --batch-size 32
```

* 12,300 states (300 episodes × 41 samples), 384-d L3 embeddings.
* Encoder contract exactly `WorldModelPolicy._prepare_info ->
  HierarchicalSolver._encode_all_levels` (the eval path).

Graph build (`--max-observed-gap 10`) produced:

```json
{"nodes": 12300, "directed_edges": 12000, "episodes": 300,
 "weak_components": 300, "largest_weak_component": 41,
 "certified_online": false, "attachments_enabled": false}
```

The observed-only graph is 300 disconnected chains (one per episode) — by
construction, no cross-trajectory edges and no reverse edges.

## 2. Stage 0 — dev query export and route audit

`scripts/export_pahrl_dev_queries.py` rebuilds the live `xy_neutral` t=0 state
(`MegaWrapper → SyncWorld → VariationWrapper`, per-task `action_space` seeding,
`settled_info_from_xy`, `set_state`, `set_navigation_goal`) and encodes it.

Verified bit-exact against a real `eval.py` trace for tasks 0 and 1: qpos,
qvel, xy, pixels, goal_pixels all match.

Route audit at the declared diagnostic cutoff 0.05:

```
{"queries": 24, "possible_route_count": 0, "possible_route_rate": 0.0}
```

Nearest-node latent distances for the 24 dev queries (start / goal) have
median ≈ 0.43 / 0.49 and max ≈ 0.89 / 0.91.

**Stage-0 gate: NO-GO on observed-only connectivity.** No dev task yields a
directed start→goal route inside a single episode chain.

## 3. Root-cause diagnosis — live/graph embedding parity

The reference's Stage-2 gate requires *exact live online embedding == graph
exported embedding for a matched stored training state*. Measured on 48 stored
states restored into the live wrapper chain:

| metric | value |
|---|---|
| pixel-exact re-renders | **0 / 48** |
| pixel mean abs diff | median 0.24 / 255 (visually identical) |
| L3 cosine distance live vs graph node | min 0.0001, p25 0.146, **median 0.441**, max 0.931 |
| L3 cosine distance between two TRAIN frames at the same xy | median **0.0097** (p25 0.0046, p75 0.0198) |

`set_state` itself is faithful: rendering the same state twice (including after
random steps) is bit-identical. The residual difference is a small systematic
re-render offset of the released TRAIN frames versus the local renderer, which
the L3 encoder amplifies to ≈0.44 for a large fraction of states.

Consequence: the latent distance for a *correct* match is ≈0.44, while the
intrinsic same-position noise is ≈0.01. Any attachment cutoff must exceed the
correct-match distance, which destroys separability against wrong nodes.

## 4. Stage 1 — local reachability trials

`scripts/collect_pahrl_attachment_trials.py` samples 8 held-out TRAIN episodes
(300–307, step 300, never in the graph), encodes the query from the **live**
render, attaches to (a) the physically nearest same-region node and (b) the
most tempting latent neighbour separated by walls, then runs the frozen L3
policy for 60 env steps. Success is physical (the environment's own goal
tolerance = 1.0); the latent threshold never decides success.

| query | kind | attach cosine | xy dist | BFS | physical success | min cosine reached |
|---|---|---|---|---|---|---|
| train_300 | same_region | 0.662 | 0.20 | 0 | yes | 0.663 |
| train_301 | same_region | 0.458 | 0.09 | 0 | yes | 0.400 |
| train_302 | same_region | 0.301 | 0.09 | 0 | yes | 0.324 |
| train_303 | same_region | 0.694 | 0.38 | 0 | yes | 0.532 |
| train_304 | same_region | 0.866 | 0.16 | 0 | yes | 0.811 |
| train_305 | same_region | 0.135 | 0.02 | 0 | yes | 0.113 |
| train_306 | same_region | 0.565 | 1.24 | 1 | yes | 0.669 |
| train_307 | same_region | 0.490 | 0.06 | 0 | yes | 0.659 |
| train_300 | cross_wall | 0.971 | 7.50 | 3 | no | 0.963 |
| train_301 | cross_wall | 0.968 | 7.83 | 3 | no | 0.952 |
| train_302 | cross_wall | 0.951 | 7.94 | 3 | no | 0.948 |
| train_303 | cross_wall | 0.905 | 7.63 | 3 | no | 0.905 |
| train_304 | cross_wall | 0.946 | 7.90 | 3 | no | 0.951 |
| train_305 | cross_wall | 0.867 | 7.21 | 5 | no | 0.863 |
| train_306 | cross_wall | 0.961 | 7.83 | 5 | no | 0.962 |
| train_307 | cross_wall | 0.956 | 7.68 | 3 | no | 0.954 |

All 16 trials replayed the source physics state bit-exactly
(`state_replay_ok=true`). Every same-region success reached the node within the
first 10 environment steps (i.e. at the first replanning boundary); every
cross-wall trial still had a minimum xy error of 5.7–8.2 units after 60 steps.
The frozen L3 controller walks to physically near nodes and does not cross
walls — the correct behaviour, but only at distances where the latent signal
is already dominated by the live/stored render offset.

## 5. Attachment calibration result

```
{"calibration_passed": true,
 "max_cosine_distance": 0.8666393458843231,
 "num_trials": 16, "positive_trials": 8, "negative_trials": 8,
 "positive_max": 0.8663239479064941,
 "negative_min": 0.8669547438621521,
 "rule": "midpoint_between_max_positive_and_min_negative",
 "evidence_sha256": "a23b180b036f2549add469a4b5533880c19c69949a8511a6a465747e0a8a9eb1"}
```

The mechanical rule passes with a margin of **0.00063** on a 0–1 scale, i.e. a
cutoff of 0.867 — effectively "almost orthogonal". The implied arrival
threshold from the same rollouts is 0.807 (success min-cosine max 0.752 vs
failure min-cosine min 0.863), and every successful trial had already
"arrived" at the first planning boundary (step 10). Both numbers are far
outside the reference's intended 0.035 band.

## 6. Certified graph coverage

Building `graph_certified.npz` with the certificate above and re-running the
dev-query audit at the certified cutoff 0.8666 still gives:

```
{"queries": 24, "possible_route_count": 0, "possible_route_rate": 0.0}
```

Even with a degenerate attachment threshold, no dev task has a directed
start→goal route: the graph is 300 disjoint episode chains and both
attachments never land on the same chain in the right order.

## 7. Stage 2 partial gates

| gate | result |
|---|---|
| `pahrl_graph` absent vs `enabled=false` | **PASS**: `steps.jsonl` byte-identical (SHA256 `b987f505…`), all `planning.jsonl` records identical except `wall_clock` |
| uncertified graph + `enabled=true` | **PASS (fail closed)**: `ValueError: PAHRL online mode is blocked…` |
| live embedding == graph embedding for a stored state | **FAIL** (0/48 exact; median cosine 0.441) |

## 8. Verdict

**NO-GO. The 24-episode closed-loop pilot is not run.**

1. **Stage 0 connectivity fails.** Observed-only TRAIN graph = 300 disjoint
   chains; 0/24 dev tasks have a directed route, both at the declared 0.05
   diagnostic cutoff and at the certified 0.867 cutoff.
2. **The only available cross-episode evidence would be expensive and is not
   the binding constraint.** Even the permissive certified attachment cannot
   find a same-chain start/goal pair.
3. **The latent attachment mechanism is degenerate on this checkpoint/data
   pair.** The live re-render of a stored training state differs from the
   stored frame by a small visual offset that the L3 encoder amplifies to
   ≈0.44 (intrinsic same-position noise ≈0.0097). The certified cutoff (0.867)
   and arrival threshold (0.807) therefore cannot distinguish "at the node"
   from "elsewhere".
4. Default-off integration is correct and strictly no-op, so nothing here
   contaminates the frozen baseline.

## 9. What a V2 would need (not implemented here)

* Rebuild graph embeddings from **live re-renders of recorded TRAIN states**
  (same renderer as online queries), so that the live-parity gate can be met,
  or match the collection renderer exactly; then re-run Stage 0/1.
* Or train the PAHRL-style temporal-distance/attachment model on live-domain
  observations rather than raw HDF5 frames.
* Cross-episode links must still be validated with real frozen-H-JEPA local
  rollouts (3/3 per direction) before they can be added.

