# Implementation assignment: HiDe-H-JEPA V1.1 task-space route guidance

**Basis:** research/ant-hide-decouple-v1, commit 4872a40; 48ef1c2
determinism protocol; baseline 17/24 dev trials (12 tasks, settle seeds 1000
and 2000). Do NOT merge old PAHRL/CLFB/RAGC or change baseline checkpoint.

Goal: real AntMaze L3 closed-loop success. The original B3 long-horizon
`P(predicted_L3_z_{t+5}) -> XY` is NO-GO. DO NOT tune alpha to make it work.
Use the B2 verified real-state XY decoder ONLY for online localization;
real-episode training trajectory evidence gives global task-space route;
frozen original 3-level H-JEPA pursues a LIVE-rendered intermediate goal.

0. Verify checkout `git rev-parse HEAD` shows 4872a40 and clean status. Create
   `research/ant-hide-topology-v11` (do not overwrite existing commits).
   Confirm all Hide V1 audits and the original B2 projector are frozen.
1. Inspect real `h_jepa/hierarchical_solver.py`, `h_jepa/planning_eval.py`,
   B1 live exporter, state-vs-goal encoding parity, checkpoint and normalizer.
   The attached reference is portable, not a source-version patch. No guessed
   auto-replacements. Keep hide V1 optional-off.
2. Build TRAIN-only dense XY transition export from the same 150 episodes as
   B1. Raw successive environment frame `episode`, `step`, `xy`; no dev
   task, val or official 50 in graph. Use the `build_topology.py` contract.
   Use only a single prespecified grid width based on TRAIN evidence.
3. B1 already has 6150 LIVE-rendered states: prepare the all-level
   `embed_1`/`embed_2`/`embed_3` array and source XY per state, hashed to
   frozen checkpoint and normalizer. Check `state` vs `goal` encoder parity
   at each level, not just level 3.
4. Build directed graph. Each edge must correspond to true sequential TRAIN
   movement. Audit possible wall bridges caused by merging spatial bins from
   different episodes; split ambiguous bins or require independent local
   evidence. No reversed/cross-wall/kNN edges invented.
5. `AdapterConfig(enabled=False)` bitwise exact no-op against original.
   On true enable, use the trained B2 projector on **current REAL state and
   goal observation only**. NEVER project `predicted_embed_0[..., -1]`.
6. Insert TopologyGoalAdapter immediately after original all-level obs/goal
   encodings, before high-to-low solve. Override all level goal embeddings
   for the same train LIVE representative when L3 horizon>1. Never modify
   environment task goal, goal-success condition, L3 candidate count,
   90-iteration GD, execution period or planner RNG. At L3 horizon==1
   no active graph waypoint, real task goal. Freeze a waypoint between
   replans; advance only upon observed (decoded) arrival and calibrated
   threshold; timeout -> permanent task-level baseline fallback.
7. Pre-commit no-op smoke on same 2 tasks x 22 steps: compare actions,
   cost traces, RNG and success state; ignore wall-clock in trace equality.
   Validate graph hash match and multi-env reset on all worker groupings.
8. Coverage gate on original dev t=0 current/goal LIVE encoded observations,
   using decoder only (no true XY). Require >=12/24 valid nontrivial
   routes; audit all routed bridges. Otherwise NO-GO.
9. Train-only local controller gate: 12 actual 20-40-frame waypoint rollouts
   with exact seeded world+wrapper/settle chain, H-JEPA frozen. >=9/12
   successes, zero false-arrival from decoder. Record paths and hash.
   Calibrate attach-radius/margin/arrival-radius/timeout/one lookahead on
   training only, freeze once. Otherwise NO-GO; do not run pilot.
10. Paired 24 episode dev only after gates pass; baseline 8/12+9/12.
    Report rescue, destroy, net, wall-clock overhead, waypoint utilization,
    source provenance and exact task/seed pairing. GO if >=+2 net,
    <=1 destroyed, <=1.5x compute. Else NO-GO and end branch. No sweep.
11. Only with GO: new independent held-out task set, then compare ablations:
    original H-JEPA vs coordinate-only local waypoint planner with same
    offline map supervision vs H-JEPA + topology. Do not claim HiDe novelty
    without this fair comparison.

**Early technical caution:** The original model was trained with L3
rollout_n=1, but B3 decoded the H=5 endpoint. Fixing the XY readout
with a new linear head is NOT justified. The original L3 400x90 full
horizon should remain intact in this candidate, since its latent
objective is the original source of 76% official-baseline success.
This candidate never optimizes the decoder's predicted XY, so
B3 failure is isolated instead of hidden.
