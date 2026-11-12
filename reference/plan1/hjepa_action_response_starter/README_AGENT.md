# H-JEPA Action-Response Network — Agent implementation handoff

Scope: §2.1 and §2.2 only. Do NOT implement Event Boundary, event JEPA,
planning-cost changes, Adaptive MPC, or modifications to the normal evaluation
pipeline in this phase.

## Source-aligned interfaces
- `h_jepa/config/train/base/pusht.yaml`: `frameskip: 5`,
  `level1.wm.embed_dim: 192`, action raw dim 2, 224x224 RGB,
  training keys include `pixels/action/state`.
- `h_jepa/models/hjepa.py`: `get_level(1)`.
- `h_jepa/models/jepa.py`: `get_level(1).encode(
  {"pixels": images[:, None]}, key="pixels")["embed_0"][:,0]`.
- `h_jepa/data.py`: `IMAGENET_STATS`.
- `stable_worldmodel/envs/pusht/env.py`: `gym.make("swm/PushT-v1")`,
  `reset(seed=...)`, `step(action)`, `render()`, `info["n_contacts"]`.
- `stable_worldmodel/envs/pusht/env.py::_set_state` restores 7-element
  observation only, and advances Pymunk one sub-step: not a complete snapshot.

## Prepare
Run from the project `h_jepa/` directory, with Python environment already
installed and a TRUSTED pretrained checkpoint:
```
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
PYTHONPATH=. python -m pytest -q tests/test_action_response.py
python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_train.npz --pairs 64 --seed 42
python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_val.npz --pairs 16 --seed 1042
python -u scripts/analyze_action_response_pusht.py \
  --data /tmp/pusht_cf_val.npz
python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_fixed.npz --pairs 24 --seed 2042 --fixed-probes
python -u scripts/analyze_action_response_pusht.py \
  --data /tmp/pusht_cf_fixed.npz
python -u scripts/train_action_response.py \
  --train /tmp/pusht_cf_train.npz \
  --val /tmp/pusht_cf_val.npz \
  --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
  --out /tmp/action_response_head.pt --steps 20 --batch-size 8
```

## Required fast checks
1. Same `seed + prefix actions` must give EXACT same anchor RGB for A and B,
   and approximately equal 7-D anchor observation; collector asserts this.
2. A/B endpoint images should generally differ. If they do not, check actions,
   rendering, goal overlays and physics.
3. `R(z,A,A)==0`, `R(z,A,B)==-R(z,B,A)`, gradients finite: pytest covers.
4. Encoder returns `[B,192]`, not patch tokens / `[B,1,192]`; script checks.
5. Run model `--steps 1` to verify real checkpoint/encoder wiring with no
   long training. `--steps 20` is only a smoke run, not a scientific result.
6. Check `val_mse` vs `zero_effect_mse`: a low MSE on a mostly zero-effect
   dataset is not evidence of learning controllable effects.
7. Physical analysis uses `state` and `n_contacts` only for post-hoc audit,
   never as model input or training target.

## Caveats and common AI-agent mistakes
- Never relabel an expert observation with a different invented action.
- Do NOT call `_set_state(state7)` twice and claim the two worlds have identical
  hidden physics; reset with the same seed and replay identical roll-in actions.
- Do NOT confuse `action_block=5` with action dimension: use `[B,5,2]`
  raw environment actions. This head does not consume the normalized
  action embedding of the pretrained JEPA. Do not silently apply its scaler.
- Input H-JEPA image normalization is ImageNet, RGB channels first,
  with tensor shape `[B,1,3,224,224]` for `JEPA.encode`.
- Freeze *all* original world-model parameters and keep `.eval()` during
  encoding. `torch.no_grad()` on endpoint targets is mandatory.
- Do not change `main_hjepa.py`, `hjepa_forward`, `hierarchical_solver.py`,
  `adaptive_policy.py`, or evaluation configs in this stage.
- Push-T `state[5:7]` is agent velocity, NOT block velocity; `_set_state`
  does not restore block linear/angular velocities.
- Contact coverage from random collection can be low. Report it; do not
  imply model learned pushing without action branches that move the object.
- `with_target=False` prevents the rendered goal overlay from confounding
  counterfactual outcome differences.
- Split by anchor/seed across train and val. Distinct action pairs from an
  identical anchor must NOT be split across train and val.
- No guarantee of reduced Push-T failure or event-discovery quality from
  merely passing these engineering tests.
- `torch.load(weights_only=False)` only on your own trusted checkpoints.

## Integration and omissions
This starter intentionally leaves `main_hjepa.py` and the planner unchanged.
For a production-scale dataset, replace NPZ by an append-only HDF5 builder
and add balanced contact-rich anchor sampling, then compare against a frozen
original World Model counterfactual difference baseline.
