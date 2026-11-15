# HiDe-HJEPA V1 审计报告（Gate I–III）

**分支**：`research/ant-hide-decouple-v1`（基线 `research/ant-trace-audit-v1` / `48ef1c2`）
**参考包**：`reference/plan6/hide_hjepa_v1_reference/hide_hjepa_v1_reference/`
**日期**：2026-11-15
**最终裁决**：**NO-GO（B3 预测潜空间审计未过）——未运行闭环 Pilot**

---

## 0. 结论摘要

| 门禁 | 结果 | 关键数字 |
|---|---|---|
| I 源码安装 + CPU 测试 | **PASS** | HiDe 21/21，全仓库 64/64 |
| I 默认关闭 no-op | **PASS** | 2 任务 × 22 步，`steps.jsonl` 逐字节一致 |
| II LIVE 域真实状态读出 | **PASS** | test 中位 0.270 / P90 0.662（阈值 1.0 / 2.0） |
| III 冻结 L3 预测审计 | **NO-GO** | 中位 1.909 / P90 3.389 / 方向余弦 0.262 |
| IV 成本尺度标定 | **未执行**（按契约在 B3 失败后停止） | — |
| V 24 Episode 闭环对照 | **未执行** | — |

**核心证据（不是调参问题，而是方法前提不成立）**：在 L3 规划真实使用的 5 步宏观视界（= 100 个环境帧）上，
冻结 L3 预测的导航位置**不优于「假设静止不动」**：

| 指标（5 步 / 100 帧，70 对） | 冻结 L3 预测 | 零位移基线 |
|---|---|---|
| latent 余弦 vs 真实末状态 | **0.528** | 0.544 |
| 解码 XY 误差中位数 | **1.909** | 1.882 |

即 `P_φ(ẑ_{t+H})` 在该视界上不携带真实未来位置信息。若强行把它加入 L3/L2 代价，
GradientSolver 可以直接利用这一「虚假导航进展」，正是 plan6/plan61 第 8 节所预防的失效模式。
因此按 `AGENT_TASK.md` III.13 与 V 的先后约束，停止后续阶段，且**未**在开发集上做任何权重/记录头搜索。

---

## 1. 仓库契约与源码补丁（Gate I.1–I.5）

- 起点：`git switch research/ant-trace-audit-v1`（`48ef1c2`）→ `git switch -c research/ant-hide-decouple-v1`。
- PAHRL 遗留的 `h_jepa/artifacts/pahrl/*.npz`（含被明令禁止复用的失配离线缓存 `states_train.npz`）
  已移出工作树至 `/root/rivermind-data/hjepa_home/tmp/pahrl_artifacts_deprecated/`，未参与任何本实验步骤。
- 参考 `apply_hide.py` 在真实源码上**拒绝写入**（3 处锚点存在空行漂移，与 PAHRL 阶段记录一致）：
  - `hierarchical_solver.py`：`self.ragc_selector ...` 与 `def set_ragc_selector` 之间多一空行；
  - `hierarchical_solver.py`：`level_info['action_cost_weight'] ...` 与 `if subgoal_latents is None:` 之间多一空行；
  - `planning_eval.py`：`solver = build_solver(cfg, model)` 与 `# Optional Feedback-N control:` 之间多一空行。
- 按 README「只适配最小语义等价锚点」的要求，新增 `reference/plan6/adaptation/patch_spec_adapted.json`
  （与参考 spec 仅在这 3 处空行不同，插入代码逐字一致）与 10 行等价 applier
  `apply_hide_adapted.py`。参考包原件未改动。
- 补丁只改 `h_jepa/hierarchical_solver.py`（+16 行）与 `h_jepa/planning_eval.py`（+12 行），**默认关闭**。

接入点语义核验：

| 位置 | 行为 |
|---|---|
| `HierarchicalSolver.solve()` | 仅 `hide_nav_enabled` 时写入 `_hide_nav_active`；**L3 恒可用，L2 仅在跟随真实 L3 子目标时可用** |
| L3 被 Horizon Schedule 跳过 | `continue` 发生在 `level_info` 构造之前 → 不会产生 L2 导航项 |
| `LevelCostModel.get_cost()` | 先原始 `_project_to_goal_space()` + 原始 `criterion()` + 原始 action cost，**最后**叠加导航项；`pred_for_cost` 保留 autograd 图 |
| L1 | 不注册 adapter，完全不变 |
| L3→L2 目标 | 仍是**完整 384 维 latent**，不传 XY |

## 2. no-op 门禁（Gate I.6，真实 GPU）

命令（两次仅差配置/开关）：

```bash
PYTHONPATH=. python eval.py --config-name ant_l3        seed=42 output.dir=hide_noop_b0 ...
PYTHONPATH=. python eval.py --config-name ant_l3_hide   seed=42 output.dir=hide_noop_off \
  hide_nav.enabled=false ...
# 公共参数：eval.num_eval=2 +eval.chunk_size=null +eval.env_seed=1000 eval.eval_budget=22
#          solver.solvers.level{1,2,3}.batch_size=2 +eval.trace_dir=trace
#          任务 audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt
```

| 文件 | 结果 |
|---|---|
| `trace/steps.jsonl`（动作、归一化动作、子目标、环境步哈希） | **IDENTICAL（逐字节）** |
| `trace/tasks/task_00{0,1}.json` | **IDENTICAL** |
| `trace/planning.jsonl` | 除 `wall_clock` 外**全部字段一致** |
| `planning_compute.json` | 除 `wall_clock` / `wall_clock_per_env` 外一致；`n_iters` 全等 |
| `metrics.yaml` | `episode_successes` / `seeds` / `steps_to_success` 一致 |

产物：`$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/hide_noop_{b0,off}/`。

## 3. LIVE 域训练状态导出与读出器（Gate II.7–II.10）

**训练数据来源**：`$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5`
（`sha256 75f583348e76546136e6c9d5caf7319df5dc55fdd4aee07c25a5b525e3357419`，22GB）
仅取前 150 集、`every=10`。

**渲染域**：`MegaWrapper → SyncWorld → VariationWrapper`，`set_state(qpos,qvel)` 后 `get_ob()` 实时渲染，
经 `WorldModelPolicy._prepare_info → HierarchicalSolver._encode_all_levels` 编码（**不使用 HDF5 中的像素**）。
eval 路径不带 variation 选项，故两端同为默认外观域（与历史记录的「live 白底 / 离线灰底」一致）。

| 项 | 值 |
|---|---|
| 状态数 | 6,150（150 集 × 41 帧） |
| L3 / L2 维度 | 384 / 384 |
| **state vs goal 编码器 parity** | `max|ΔL3| = 0.0`（同像素同 propio 完全一致） |
| checkpoint sha256 | `c55d9835…f69f99` |
| normalizer sha256 | `1e86ce0a…9fb44` |
| 产物 | `artifacts/hide/live_train_states.npz`（sha256 `55789361…e24f4`，本地未入库） |

**读出器**：384→128→128→2，ReLU，30 epoch，AdamW(lr 1e-3, wd 1e-4)，seed 42，按**集**切分
（105 / 22 / 23 集），XY 仅在训练集内归一化。学习率/架构未做任何搜索。

| 集 | 中位 XY 误差 | P90 |
|---|---|---|
| val (902) | 0.309 | 0.694 |
| **test (943)** | **0.270** | **0.662** |

门禁 `median < 1.0 且 P90 < 2.0` → **PASS**。产物 `artifacts/hide/nav_projector.pt`
（sha256 `69e3c034…bfad0`）+ `nav_projector.train_metrics.json`。

## 4. 冻结 L3 预测审计（Gate III.11–III.13）—— **NO-GO**

### 4.1 时间尺度（由 checkpoint 推导，非假设）

| 层 | 依据 | 结论 |
|---|---|---|
| L1 | `action_embed.patch_embed` 的 `in_channels=40` ÷ `action_dim=8` | 5 环境帧 / token |
| L2 | `temporal_stride=2` | 2 个 L1 token |
| L3 | `temporal_stride=2` | 2 个 L2 token |

⇒ **1 个 L3 宏观步 = 20 环境帧**；`level3.horizon=5`、`cost_last_n=1`
⇒ L3 代价实际比较的是 **+100 环境帧** 的预测 latent。
`JEPA.rollout` 在 `H=1` 上下文下产生 6 个 latent（索引 0 为起点，1..5 为 5 步预测），
`criterion` 取 `[1:][-1]` 即第 5 步，与本文审计使用的索引一致。

### 4.2 预测对构造（真实记录动作，非合成）

`scripts/make_hide_prediction_pairs.py`：

- 取 head 文件 `metadata.test_episodes`（23 个**训练集内留出** episode）；
- 选**互不重叠**的 L3 窗口（每集 4 个：k=0/5/10/15），要求真实位移 ≥ 1.0；
- 起止帧真实状态经 LIVE 渲染重新编码 → `start_z` / `true_end_z` / `true_end_xy`；
- 宏观动作由 checkpoint 自身的 level-1 `action_embed`、level-2/3 `SequenceEncoder` +
  `_action_windows` 从**记录的真实动作**编码（完全复刻 `HJEPA.encode_hierarchical`），
  幅度与 checkpoint latent action queue 一致（queue std 均值 0.616 vs 宏观动作 |a| 均值 0.903）；
- 冻结 L3 `rollout` 得到 `pred_z`。

产物 `artifacts/hide/prediction_pairs_train_heldout.npz`（70 对，sha256 `b117aa5c…0328`），
metadata 含 `source_split=train_heldout`、`predictor_source=frozen_l3_true_macro_action_rollout`、
`time_scale_verified=true`、`audited_env_frame_horizon=100`。

### 4.3 审计结果（`scripts/audit_hide_predictions.py`，阈值预先固定）

| 指标 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| 对数 / 非平凡移动对 | 70 / 68 | ≥32 / ≥12 | PASS |
| 中位预测 XY 误差 | **1.909** | < 1.0 | **FAIL** |
| P90 预测 XY 误差 | **3.389** | < 2.0 | **FAIL** |
| 平均运动方向余弦 | **0.262** | ≥ 0.5 | **FAIL** |
| 预测/真实误差比 | 7.06 | — | — |
| 真实状态读出（同批） | 0.270 | — | 正常 |

`status: NO-GO` → `artifacts/hide/prediction_audit.json`（sha256 `9dc4eab9…486a`）。

### 4.4 失败归因诊断（只诊断，未调参）

`scripts/diag_hide_prediction_horizon.py`（24 对，逐 20 帧边界）：

| 步 | 环境帧 | 预测 XY 中位 | P90 | latent 余弦 | 运动方向余弦 |
|---|---|---|---|---|---|
| 0 | 0 | 0.213 | 0.494 | 1.000 | — |
| 1 | 20 | 0.646 | 1.119 | 0.945 | 0.459 |
| 2 | 40 | 0.928 | 1.790 | 0.846 | 0.445 |
| 3 | 60 | 1.084 | 2.476 | 0.741 | 0.433 |
| 4 | 80 | 1.755 | 3.186 | 0.570 | 0.302 |
| 5 | **100** | **1.602** | **3.417** | **0.577** | **0.397** |

解读：

1. **不是读出器坏了**：真实状态误差 0.21–0.27 稳定，1 步预测仍有 0.945 的 latent 余弦，
   说明宏观动作编码与时间对齐基本正确（若错位，1 步即崩溃）。
2. **是预测器长视界漂移**：latent 余弦 0.945 → 0.577，并被「零位移基线」追平
   （cos(start,true)=0.544，预测 0.528）——在第 5 步，预测携带的导航信息**不超过「不动」先验**。
3. 即使把审计收紧到 1 步，方向余弦 0.459 仍低于 0.5 阈值；不存在「换个视界就能过门禁」的解释空间。

## 5. 工程冒烟（形状 / 梯度 / 开销，真实 checkpoint）

`scripts/smoke_hide_grad.py`（随机候选，只验证接线，不代表成功实验）：

| 项 | L3 | L2 |
|---|---|---|
| 代价形状 | `[2, 400]` | `[2, 100]` |
| 预测长度 / horizon | 6 / 5 | 3 / 2 |
| 投影后预测形状 | `[2,400,6,384]` | `[2,100,2,384]`（L3 空间） |
| 动作梯度 | 有限且非零（2.6e5 量级） | 有限且非零（4.2e4 量级） |
| 冻结 head 参数梯度 | 无 | 无 |
| `enabled=false` 适配器 | 返回同一 tensor（`is` 级） | 同 |
| `get_cost` 额外耗时 | 0.84× | 0.79× |

产物 `artifacts/hide/gpu_smoke.json`（sha256 `bd868041…fe14`）。

**fail-closed 验证**：以真实（NO-GO）审计文件尝试安装 →

```
ValueError: Prediction audit on frozen L3 latent must PASS before enabling HiDe
hide_nav_enabled: False | L1 has adapter: False
```

即坏/陈旧/缺失的 projector、scale、audit 一律抛错，不会静默回退。

## 6. 未执行的阶段与理由

| 阶段 | 状态 | 理由 |
|---|---|---|
| B4 成本尺度标定 | 跳过 | 契约要求「B3 失败即停」；用失配预测器标定尺度会造成「尺度正确但语义错误」的假象 |
| C GPU 2 任务开启冒烟 | 跳过 | 同上（安装本应 fail-closed） |
| D 12×2 闭环 Pilot | **跳过** | 门禁 III 未过，结论已可判定；不消耗开发集，也不做 α / 记录头搜索 |
| 官方 50 任务 | 跳过 | 从未进入 |

## 7. 复现命令

```bash
cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME=/root/rivermind-data/hjepa_home PYTHONPATH=. MUJOCO_GL=egl
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TRAIN="$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5"

python scripts/export_hide_live_train.py --checkpoint "$CKPT" --dataset "$TRAIN" \
  --output artifacts/hide/live_train_states.npz --max-episodes 150 --every 10 --batch-size 16
python scripts/train_hide_nav.py --input artifacts/hide/live_train_states.npz \
  --output artifacts/hide/nav_projector.pt --epochs 30 --batch-size 256 --seed 42
python scripts/make_hide_prediction_pairs.py --checkpoint "$CKPT" --dataset "$TRAIN" \
  --head artifacts/hide/nav_projector.pt \
  --output artifacts/hide/prediction_pairs_train_heldout.npz
python scripts/audit_hide_predictions.py --pairs artifacts/hide/prediction_pairs_train_heldout.npz \
  --head artifacts/hide/nav_projector.pt --output artifacts/hide/prediction_audit.json
python scripts/diag_hide_prediction_horizon.py --checkpoint "$CKPT" --dataset "$TRAIN" \
  --pairs artifacts/hide/prediction_pairs_train_heldout.npz \
  --head artifacts/hide/nav_projector.pt --output artifacts/hide/prediction_horizon_diag.json
python scripts/smoke_hide_grad.py --checkpoint "$CKPT" \
  --head artifacts/hide/nav_projector.pt --output artifacts/hide/gpu_smoke.json
```

## 8. 由此得到的结论（不构成调参许可）

1. 在**当前冻结 checkpoint**上，「为高层补充二维导航代价」的假设前提不成立：
   L3 的 5 步预测在 100 帧上失去导航语义，任何基于它的导航项都无法被解释为「朝目标推进」。
2. 该失败**与读出器质量无关**（真实状态 0.27 中位），也与代码接线无关
   （no-op 逐字节一致、梯度有限、形状与原始样本数一致）。
3. 因此 HiDe-HJEPA V1 在当前 checkpoint 上应判定 NO-GO；
   下一步若仍要走功能性解耦，必须先解决「L3 预测在规划视界内的导航保真度」
   （例如缩短 L3 代价视界并重新审视 horizon 配置，或改进预测器），
   而不是在开发集上搜索 α 或更换读出器。
4. 按照 plan61 的既定原则：**任何未通过预先固定门禁的机制，都不进入闭环成功率比较。**
