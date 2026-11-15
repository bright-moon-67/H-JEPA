#!/usr/bin/env bash
# Held-out validation for HiDe-HJEPA V1.1 (reference step 11, part 1).
#
# 12 independent AntMaze tasks that were NEVER used for any threshold,
# calibration or tuning decision: enumerated distance-3 cell pairs minus the
# development pack minus the official 50-task pack (0 overlap), sampled with a
# fixed seed. Same checkpoint, seed, batch groups, 350-step budget and 10-frame
# replanning as the development pilot; only the task pack and the settle seed
# (3000, unused so far) are new.
#
#   tmux new-session -d -s hjepa_heldout "bash scripts/topology_v11_heldout.sh"
#   tail -f /tmp/topo_heldout.log
set -euo pipefail

cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME="${HJEPA_HOME:-/root/rivermind-data/hjepa_home}"
export PYTHONPATH=.
export MUJOCO_GL=egl
PY=/opt/conda/envs/hjepa/bin/python
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TASKS=artifacts/hide/topology_v11_heldout_n12_d3.pt
ART=artifacts/hide
SEED="${1:-3000}"

echo "### held-out baseline env_seed=${SEED} $(date -Is)"
"$PY" eval.py --config-name ant_l3 \
  policy="$CKPT" seed=42 \
  load_eval_trajs_path="$TASKS" \
  eval.num_eval=12 eval.eval_budget=350 \
  +eval.chunk_size=null +eval.env_seed="$SEED" \
  solver.solvers.level1.batch_size=2 \
  solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 \
  +eval.trace_dir=trace \
  output.dir="heldout_b0_env${SEED}"

echo "### held-out topology env_seed=${SEED} $(date -Is)"
"$PY" eval.py --config-name ant_l3_topo \
  policy="$CKPT" seed=42 \
  load_eval_trajs_path="$TASKS" \
  eval.num_eval=12 eval.eval_budget=350 \
  +eval.chunk_size=null +eval.env_seed="$SEED" \
  solver.solvers.level1.batch_size=2 \
  solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 \
  hide_topology.enabled=true \
  hide_topology.graph_path="$PWD/$ART/topology_train_live.npz" \
  hide_topology.readout_path="$PWD/$ART/nav_projector.pt" \
  hide_topology.geometry_audit_path="$PWD/$ART/topology_graph_audit.json" \
  hide_topology.certification_path="$PWD/$ART/topology_gate2_certification.json" \
  hide_topology.record_path="$PWD/$ART/topology_decisions_heldout_env${SEED}.json" \
  +eval.trace_dir=trace \
  output.dir="heldout_topo_env${SEED}"

echo "### HELDOUT COMPLETE env_seed=${SEED} $(date -Is)"
