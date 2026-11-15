#!/usr/bin/env bash
# Gate C: the ONE fixed paired 12x2 development pilot for HiDe-HJEPA V1.1.
#
# Runs the two deterministic settle seeds sequentially against the audited
# baseline directories (eval_dev12_env1000 / eval_dev12_env2000, 8/12 + 9/12).
# Nothing here tunes parameters: alpha/radius/lookahead are frozen in
# config/eval/ant_l3_topo.yaml and certified by the Gate-1/Gate-2 artifacts.
#
# Run protected (survives the chat terminal):
#   tmux new-session -d -s hjepa_pilot "bash scripts/topology_v11_pilot.sh"
#   tmux attach -t hjepa_pilot        # optional
# Status:
#   tail -f /tmp/topo_pilot.log
set -euo pipefail

cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME="${HJEPA_HOME:-/root/rivermind-data/hjepa_home}"
export PYTHONPATH=.
export MUJOCO_GL=egl

CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TASKS=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt
ART=artifacts/hide

for SEED in 1000 2000; do
  echo "=== HiDe-topology pilot env_seed=${SEED} ($(date -Is)) ==="
  /opt/conda/envs/hjepa/bin/python eval.py \
    --config-name ant_l3_topo \
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
    hide_topology.record_path="$PWD/$ART/topology_decisions_env${SEED}.json" \
    +eval.trace_dir=trace \
    output.dir="hide_topo_dev${SEED}"
  echo "=== done env_seed=${SEED} ($(date -Is)) ==="
done
echo "PILOT COMPLETE"
