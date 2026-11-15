# Agent execution order — PAHRL-inspired graph guidance, frozen H-JEPA

Repo: https://github.com/bright-moon-67/H-JEPA
Base branch: research/ant-trace-audit-v1
Base commit: 48ef1c2 (must verify exact branch/source layout)
New branch: research/ant-pahrl-graph-v1

Single goal: improve real AntMaze L3 closed-loop success over the deterministic
17/24 development baseline without any test set leakage.

## First actions
1. Inspect repo status and exact base commit; create a new clean branch.
2. Integrate files + strict `apply_pahrl.py`. If patch anchor mismatches, manually
   inspect new source before adapting; do not blindly replace.
3. Run CPU tests, all repo tests, and a frozen-checkpoint smoke.
4. Check that default-off and explicitly disabled modes exactly reproduce
   the original full trajectory and cost curve. This gate is nonnegotiable.

## Next: TRAIN-ONLY graph
5. Export actual train HDF5 observation embeddings from the **eval encoder**
   using the H-JEPA ckpt and its normalizer. Never use val or 50 official tasks.
6. Build observed-only directed graph. Inspect components, coverage, edge
   provenance. Do not add kNN shortcuts and do not turn on graph mode yet.
7. Export t=0 dev query embeddings using exact replayed xy_neutral world state.
   (The dev task .pt has placeholder ZERO pixels. Do not encode them.)
8. Measure graph route availability and detached subgraphs. If poor, report
   NO-GO or conduct ONE prespecified batch of controller-verified link trials
   from TRAIN trajectories. No endless threshold tuning.

## Local reachability gate
9. For cross-trajectory edges, restore physical state with the project's
   exact wrapper chain and seeded action_space; validate 3/3 directional
   H-JEPA local rollouts. Record source/target and success.
10. Validate current/goal query-to-node attachments with independent,
    train-only positives and cross-wall hard negatives. Use one calibration
    without searching thresholds. If distances overlap, NO-GO.
11. Confirm goal-neighborhood/arrival threshold separately, with recorded
    local successes; don't use official task outcomes to tune it.
12. Build certified graph and report exact SHA/checkpoint/normalizer/training
    provenance.

## Integration gate
13. Check L3 goal override propagation into L2 terminal subgoal anchoring.
14. Check that at L3 horizon==1 original task goal is used, not graph route.
15. Check controller reset and independent env/batch indexing; goal change
    and waypoint timeout fallbacks.
16. Check on-policy qpos/qvel, pixels and goal hashes, planner RNG unaffected.
17. For 2-3 dev episodes compare graph disabled / no-route fallback against
    original actions step-by-step. Only then enable actual graph routes.

## Fixed small closed-loop pilot
18. Use exactly the existing 12 dev tasks x settle seed 1000/2000,
    original planner seed, 350 steps and batch grouping.
19. Log rescue, destroy, net, actual goal overrides, route coverage,
    waypoint arrival/timeout, wall time and planner cost.
20. GO if net>=+2, destroyed<=1, overhead<=1.5x. Otherwise NO-GO;
    do not sweep graph densities, thresholds, nearest-K or model weights.
21. If GO, validate on independent NEW tasks before any official 50 eval.

## Prohibited
- No full PAHRL DHAC/TDHAC training in V1.
- No online xy, BFS, collision map or true simulator reward for choices.
- No training/world-model checkpoint change.
- No RAGC, CLFB, ARN, Feedback-N or adaptive replanning simultaneously.
- No unofficial manual changes to successful/failed tasks.
- No unvalidated cross-wall / reverse graph edges.
- No official 50-task success test while developing.
- No "leave process running" async promises; complete one bounded stage and
  submit the results and actual Git commit each time.

Deliver: source changes, tests, metadata hashes, stage0 graph metrics,
stage1 certification proof, stage2 no-op evidence and (if permitted) stage3
paired success table. Stop on a failed gate.
