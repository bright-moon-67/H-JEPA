#!/usr/bin/env bash
# Reference step 11, part 2: fairness ablations on the SAME held-out pack.
#
# A1  oracle_localization  UPPER BOUND, deliberately privileged: replaces the
#     learned 2-D readout with the simulator's true xy (denormalized). It can
#     only ever be reported as a bound, never as the method.
# A2  anti_route           INFORMATION-MATCHED CONTROL: same graph, same
#     readout, same checkpoint, but the waypoint is the neighbour FARTHEST
#     from the goal. If the gain were just "any goal perturbation", this would
#     gain too.
#
# Waits for the held-out baseline/topology session to finish first so the two
# do not contend for the GPU.
#
#   tmux new-session -d -s hjepa_ablation "bash scripts/topology_v11_ablation.sh"
#   tail -f /tmp/topo_ablation.log
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

echo "### waiting for held-out session $(date -Is)"
for _ in $(seq 1 480); do
  if grep -q "HELDOUT COMPLETE" /tmp/topo_heldout.log 2>/dev/null; then
    echo "    held-out finished"
    break
  fi
  sleep 15
done

run_arm () {
  local ABL="$1"
  echo "### ablation ${ABL} env_seed=${SEED} $(date -Is)"
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
    hide_topology.ablation="$ABL" \
    hide_topology.record_path="$PWD/$ART/topology_decisions_${ABL}_env${SEED}.json" \
    +eval.trace_dir=trace \
    output.dir="heldout_${ABL}_env${SEED}"
}

run_arm oracle_localization
run_arm anti_route

echo "### ABLATION COMPLETE env_seed=${SEED} $(date -Is)"
