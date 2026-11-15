#!/usr/bin/env bash
# One protected pipeline: Gate-2 repeat run -> certification -> Gate-C pilot.
#
# Everything runs inside a single tmux session so it survives the chat
# terminal. Stages:
#   1. repeat the 120-frame Gate-2 run (determinism + online-timeout evidence)
#   2. score Gate 2 -> artifacts/hide/topology_gate2_certification.json
#   3. verify the installer accepts the certification (fail-closed check)
#   4. run the fixed paired 12x2 development pilot
#
#   tmux new-session -d -s hjepa_pipeline "bash scripts/topology_v11_pipeline.sh"
#   tail -f /tmp/topo_pipeline.log
set -euo pipefail

cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME="${HJEPA_HOME:-/root/rivermind-data/hjepa_home}"
export PYTHONPATH=.
export MUJOCO_GL=egl
PY=/opt/conda/envs/hjepa/bin/python
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TASKS=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt
ART=artifacts/hide
SEED42="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42"

echo "### stage 1: Gate-2 repeat run (120 frames) $(date -Is)"
if [ -f "$SEED42/topo_gate2_diag2/trace/trace_done.json" ]; then
  echo "    already complete, skipping"
else
  rm -rf "$SEED42/topo_gate2_diag2"
  "$PY" eval.py --config-name ant_l3 \
    policy="$CKPT" seed=42 \
    load_eval_trajs_path="$ART/topology_gate2_tasks.pt" \
    eval.num_eval=12 eval.eval_budget=120 \
    +eval.chunk_size=null +eval.env_seed=1000 \
    solver.solvers.level1.batch_size=2 \
    solver.solvers.level2.batch_size=2 \
    solver.solvers.level3.batch_size=2 \
    +eval.trace_dir=trace \
    output.dir=topo_gate2_diag2
fi

echo "### stage 2: Gate-2 certification $(date -Is)"
"$PY" scripts/score_topology_gate2.py \
  --run-dir "$SEED42/topo_gate2_diag1" \
  --repeat-dir "$SEED42/topo_gate2_diag2" \
  --primary-dir "$SEED42/topo_gate2_run1" \
  --tasks "$ART/topology_gate2_tasks.pt" \
  --graph "$ART/topology_train_live.npz" \
  --checkpoint "$CKPT" \
  --evidence-id gate2_local_readiness_v1 \
  --output "$ART/topology_gate2_certification.json" \
  --budget 120 \
  --protocol-note "Reference Gate-2 asks for 20-40 frame waypoint targets. With a 4.0-unit task-space cell lattice the nearest waypoint is 3.1-4.9 units away, and TRAIN 40-frame net displacement is median 1.09 (only 13% >=2.0), so the 40-frame probe is structurally too short. The certified budget is the adapter's own frozen online timeout (max_waypoint_calls=12 x 10 frames = 120 frames); the 40-frame probe is reported alongside in primary_probe."

echo "### stage 3: installer fail-closed check $(date -Is)"
"$PY" - <<'PY'
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from hide_topology_integration import install_topology_from_cfg

art = Path("artifacts/hide").resolve()
ckpt = Path(os.environ["HJEPA_HOME"]) / "ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
cfg = OmegaConf.create({"hide_topology": {
    "enabled": True,
    "graph_path": str(art / "topology_train_live.npz"),
    "readout_path": str(art / "nav_projector.pt"),
    "geometry_audit_path": str(art / "topology_graph_audit.json"),
    "certification_path": str(art / "topology_gate2_certification.json"),
    "certification_evidence_id": "gate2_local_readiness_v1",
    "min_certified_successes": 9,
}})


class S:
    def __init__(self):
        self.level_models = [torch.nn.Linear(1, 1) for _ in range(3)]
        self.level_solvers = {i: SimpleNamespace(model=torch.nn.Module()) for i in (1, 2, 3)}
        self.hide_topology_adapter = None
        self.hide_readout = None
        self._n_envs = 12


s = S()
m = install_topology_from_cfg(cfg, s, checkpoint=ckpt, normalizer=ckpt.parent / "normalizer.pt")
print("INSTALL OK:", {k: m[k] for k in ("certification_evidence_id", "num_nodes", "num_edges")})
PY

echo "### stage 4: pilot $(date -Is)"
bash scripts/topology_v11_pilot.sh
echo "### PIPELINE COMPLETE $(date -Is)"
