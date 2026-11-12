# AntMaze L3 Baseline Trace Audit

日期：2026-11-12  
诊断分支：`research/ant-trace-audit-v1`  
基线提交：`a5b3e2e`（原始 38/50 运行所在提交，分支 `research/feedback-execution-v1`）

---

## 0. 结论摘要

1. **Trace logger 已落地并验证**：默认完全 no-op，开启后只观察已有数据，不消耗随机数、不改变动作、不额外推理；43 个单元测试全部通过；两次同配置短跑逐步 trace 的 SHA256 完全一致。
2. **门禁失败**：原 Worker 1 的 8/13 无法重放（得到 6/13，成功任务集合不同）。按任务书要求，**机制归因立即暂停**。
3. **根因已定位**：不是 logger，而是评测本身的隐藏随机性——`xy_neutral` 初始状态通过 `settled_info_from_xy()` 的 **5 步 `action_space.sample()` settle** 得到，而 `env.reset(seed=...)` 不会给 action space 播种。两个独立进程即使配置完全相同，起始 qpos 最大差约 **1.0**。
4. **修复已加入且默认关闭**：`+eval.env_seed=<int>` 同时播种 `reset()` 与每个子环境的 action-space RNG；实测两次 3 任务评测逐位一致。
5. 可复现重放（`env_seed=1000`）为 **8/13**；其失败任务为 15、16、19、22、24。该结果用于描述性诊断，**不是**原始 38/50 失败任务的复现，两者不能互相替代。

---

## 1. 运行身份

| 项 | 原始 50 任务评测 | 本次审计重放 |
|---|---|---|
| 分支 | `research/feedback-execution-v1` | `research/ant-trace-audit-v1` |
| 提交 | `a5b3e2e` | `ec3b60d` + `8e56e7d` + 本报告提交 |
| 入口 | `parallel_eval.py` → `eval.py` | `eval.py`（Worker 1 配置） |
| 配置 | `ant_l3` | `ant_l3` |
| Checkpoint | `ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt` | 同左 |
| 任务 | `eval_ant_l3_par/w0..w3`（13/13/12/12） | tasks 13–25，seed=55，13 环境 |
| Solver 参数 | `n_steps=90`，samples L1/L2=100、L3=400，`batch_size=2` | 同左 |
| 结果 | 38/50 = 76.0% | 8/13 = 61.5% |

> 注意：任务书中写的“原 batch_size=16”与实际产物不符；`eval_ant_l3_par/w*/eval_config.yaml` 记录的是 `batch_size=2`，本次重放严格使用 2。

---

## 2. PHASE 0：已有产物审计

已存在（四个 worker 均有）：

- `metrics.yaml`：per-task `episode_successes`、`steps_to_success`、`wall_clock_to_success`、`evaluation_time`；**`seeds: null`**。
- `planning_compute.json`：每次 GD solve 的 `level / horizon / num_samples / n_iters / wall_clock`，共 95 条/worker。
- `eval_config.yaml`：完整配置（含 `output.dir`、slice 路径、solver 参数）。

缺失且无法从上述文件恢复：

- 逐步 `xy / goal_xy / qpos / qvel`；
- 规划调用与 `steps_taken / call_idx` 的对齐（`planning_compute.json` 不含任务或步数索引）；
- 环境 variation / settle 状态。

因此 PHASE 0 可复用的只有“各层求解统计 + 任务级结果”，不足以做轨迹级归因。

---

## 3. PHASE 1：opt-in Trace Logger

实现：`h_jepa/antmaze_trace.py`  
启用：`+eval.trace_dir=trace`（相对路径挂在输出目录下）；未设置时 `install_trace()` 返回 `None`，不包装任何对象。  
接入点：`h_jepa/planning_eval.py`（两条评测路径；tracing 与 `chunk_size` 分块互斥，显式报错）。

### 记录内容

| 频率 | 字段 |
|---|---|
| 每个环境步 | `t, env, task, xy, goal_xy, qpos, qvel, action, success, terminated, truncated` |
| 每次规划（按层） | `level, call_idx, steps_taken, horizon, num_samples, n_eps, n_iters, wall_clock` |
| 每次规划 | `goal_level, goal_in_upper_space, goal_norm_last, goal_vec64` |
| 每次规划 | `pred_shape, pred_norm_last, pred_vec64, pred_cos_prev`（与同一层上一次调用的余弦） |
| 每次规划 | 选中动作（L1 为完整 plan；L2/L3 为 macro action） |
| 每次规划 | `cost_batches`：按 solver 内层 batch 拆分的 GD cost 曲线 + 最终 per-env cost |

约束落实：

- 关闭时零开销；
- 不调用任何 RNG（单测用 `np.random`/`torch.rand` 前后取值验证）；
- 不改变动作流（单测逐步对比）；
- 不做额外模型前向；cost 直接复用 solver 已有的 `get_cost` 结果；
- 每个求解只做一次 `.cpu()`（latent 摘要、cost 曲线），避免逐步 GPU 同步；
- 逐步 JSONL 增量写入，结束时原子写 per-task JSON 与 `trace_done.json`。

产物：

```
<output.dir>/trace/
  meta.json          # worker/seed/env_seed/task_ids/maze_map/maze_unit/git commit
  steps.jsonl        # 4563 行 = 13 env × 351 步
  planning.jsonl     # 95 条 = 每层每次规划
  tasks/task_XXX.json
  trace_done.json
```

测试：`h_jepa/tests/test_antmaze_trace.py`（3 个），全仓库 `43 passed`。

---

## 4. PHASE 2：门禁结果（FAILED）

### 4.1 原 Worker 1 对比（同 seed、同 batch、同 slice、`chunk_size=null`，仅加日志）

| 检查 | 结果 |
|---|---|
| `episode_successes` | ❌ 原始 `[F,T,F,F,T,T,T,T,F,F,T,T,T]` vs 重放 `[F,F,F,T,T,T,F,T,F,F,T,F,T]` |
| `steps_to_success` | ❌ 逐任务不同 |
| `seeds` | ✅ 均为 `null` |
| `planning_compute` 记录数 / level / horizon | ✅ 95 条，结构一致 |
| `planning_compute.n_iters` | ❌ 不同（早停与数值漂移无关：根因见 4.2） |

原始 8/13，重放 6/13。**门禁不通过，按任务书要求暂停机制归因。**

### 4.2 根因：settle 步骤使用了未播种的 action-space RNG

`ant_l3.yaml` 使用 `start_state_mode: xy_neutral`、`goal_state_mode: xy_neutral`。  
`World._apply_xy_neutral_eval_state()` 调用 `LocomazeEnv.settled_info_from_xy()`：

```python
self.env.set_state(qpos, qvel)
self.env.set_xy(xy)
for _ in range(5):
    self.env.step(self.action_space.sample())   # <-- action space RNG
```

而 `env.reset(seed=...)` 只播种环境的 `np_random`，**不会**播种 `action_space`。因此：

- 每个评测进程的 5 个 settle 动作不同 → settle 后的 `qpos/qvel` 不同；
- 后续 `set_state()` 使用的是 settle 后的状态；
- 规划器是确定性的（同输入 + 同 seed → 同动作），但输入从第 0 步就不同。

实测（同一任务、固定目标、两个独立进程）：

| 设置 | settled qpos 是否一致 | 最大差 |
|---|---|---|
| `reset(seed=None)` ×2 | ❌ | 1.075 |
| `reset(seed=1234)` ×2 | ❌ | 0.954 |
| `reset(seed=1234)` + `action_space.seed(1234)` ×2 | ✅ | 0.0 |

这解释了：`metrics.yaml` 的 `seeds: null`、原 38/50 无法逐任务复现、重放成功率 46%–62% 波动。

### 4.3 修复：`+eval.env_seed=<int>`（默认关闭）

`stable_worldmodel/world.py::evaluate_from_dataset(..., env_seed=None)`：

```python
if env_seed is not None:
    seeds = [int(env_seed) + i for i in range(self.num_envs)]
    for i, env in enumerate(self.envs.unwrapped.envs):
        env.action_space.seed(int(env_seed) + i)
```

- 缺省 `None` → 行为与原来完全一致；
- 设置后 `metrics.yaml` 会记录 `seeds: [base, base+1, ...]`；
- 分块路径按 `env_seed + chunk_start` 偏移，与 planner seed 的分块语义一致；
- `parallel_eval.py` 后续应给每个 worker 传不同的 `eval.env_seed`（如 `1000 + off`）。

### 4.4 确定性证明（DET_OK）

两次完全相同的小规模评测（tasks 13–15，`eval_budget=60`，`env_seed=1000`，完整 solver 参数）：

| 检查 | 结果 |
|---|---|
| success / success_rate / steps_to_success | ✅ 全等 |
| `steps.jsonl` SHA256 | ✅ `c0cd053d…a667b1` ×2 |
| planning 动作 / n_iters / cost 曲线 | ✅ 逐项相同 |
| `metrics.seeds` | ✅ `[1000, 1001, 1002]` |

证据文件：`determinism_check.json`。

---

## 5. 可复现重放的描述性诊断

**重要限定**：以下分析基于 `eval_ant_l3_trace_w1_seeded`（`env_seed=1000`）。它与原始 38/50 不是同一个初始状态集合，**不能**用来说明原始失败任务的机制。仅作为“logger 能提供什么证据”的示例，以及后续确定性基线的起点。

### 5.1 任务级结果（seeded，8/13）

| task | 结果 | 类别 | 格子距离 d0→dmin→dfin | 最长停滞 | 最早异常 | L3 余弦中位/反转 |
|---|---|---|---|---|---|---|
| 13 | ✅ 328 步 | success | 3→1→1 | 6 | — | 0.97 / 0 |
| 14 | ✅ 348 步 | success | 3→0→0 | 6 | — | 0.98 / 0 |
| 15 | ❌ | A | 3→3→4 | 9 | step 229 距离回升 | 0.94 / 1 |
| 16 | ❌ | B | 3→0→0 | 9 | step 320 距离回升 | 0.97 / 0 |
| 17 | ✅ 321 步 | success | 3→0→0 | 10 | — | 0.98 / 0 |
| 18 | ✅ 182 步 | success | 3→1→1 | 4 | — | 0.98 / 0 |
| 19 | ❌ | B | 3→3→3 | **40 @187** | step 187 停滞 | 0.99 / 0 |
| 20 | ✅ 290 步 | success | 3→1→1 | 8 | — | 0.98 / 0 |
| 21 | ✅ 279 步 | success | 3→0→0 | 6 | — | 0.98 / 0 |
| 22 | ❌ | A | 3→3→5 | 8 | 无 | 0.91 / 0 |
| 23 | ✅ 235 步 | success | 3→0→0 | 10 | — | 0.96 / 0 |
| 24 | ❌ | B | 3→3→3 | **52 @280** | step 280 停滞 | 0.97 / 0 |
| 25 | ✅ 328 步 | success | 3→0→0 | 8 | — | 0.97 / 0 |

### 5.2 五个失败案例

- **task 19 / 24（B，局部运动失败）**：整段 350 步都停留在起始格。19 的 `xy` 从 (20.86, 4.71) 仅移动到 (18.75, 4.75)，24 从 (19.86, 11.09) 到 (20.13, 11.31)；两者 `distinct_cells = 1`。高层子目标余弦几乎恒为 1.0（0.99 / 0.97），**没有规划抖动**。最早异常是停滞开始（187 / 280 步），此前在格子内小幅移动。身体高度最低 0.25–0.26，不属于“摔倒”（<0.2）判定。
- **task 15 / 22（A，路线选择）**：分别只访问 2 和 3 个格子，最终最短路径距离反而增大（3→4，3→5）；`mean_disp` 0.07–0.08，说明身体在动，但没有沿可通行拓扑接近目标。L3 余弦中位 0.94 / 0.91，15 有一次反转。
- **task 16（B，进入目标格但未达容差）**：轨迹在 step ~315 和 ~340 两次进入目标格（BFS 距离 0），但 `min |xy-goal| = 1.57`，始终未进入 1.0 的成功容差；最终停在目标格内。属于终点精度/局部控制问题，不是高层选路错误。

案例图：`figures/case_task_015.png`、`016`、`019`、`022`、`024`（左：迷宫 XY 轨迹 + 规划调用位置；中：BFS 距离；右：各层子目标余弦）。

### 5.3 跨运行一致性（仅描述，不归因）

| task | 原始 w1 | 无种子重放 | seeded 重放 |
|---|---|---|---|
| 13 | ❌ | ❌ | ✅ |
| 14 | ✅ | ❌ | ✅ |
| 15 | ❌ | ❌ | ❌ |
| 16 | ❌ | ✅ | ❌ |
| 17 | ✅ | ✅ | ✅ |
| 18 | ✅ | ✅ | ✅ |
| 19 | ✅ | ❌ | ❌ |
| 20 | ✅ | ✅ | ✅ |
| 21 | ❌ | ❌ | ✅ |
| 22 | ❌ | ❌ | ❌ |
| 23 | ✅ | ✅ | ✅ |
| 24 | ✅ | ❌ | ❌ |
| 25 | ✅ | ✅ | ✅ |

三次运行都失败：**15、22**。原始 + seeded 失败：16。19、24 只在两次重放中失败（原始成功）→ 强烈提示初始 settle 状态决定结果。

### 5.4 目前**不能**下的结论

- 不能把 15/22 的“路线选择”外推到原始 38/50 的 12 个失败；
- 不能宣称 C（高层规划不稳定）是主要机制：本样本 L3 余弦中位 0.91–0.99，反转 0–1 次；
- 不能比较“失败任务 vs 成功任务”的规划统计显著性：13 个任务、单一 env seed，且初始状态分布未受控；
- L3 子目标是 latent 的桶均值摘要，**未解码为 XY**，不能当作迷宫路标解释。

---

## 6. 决定与下一步（按任务书暂停）

本轮到此停止，不进入方法开发、不调参、不启用 RAGC/ARN/Adaptive/Feedback-N。

建议的最小下一步（需要你确认）：

1. 用 `eval.env_seed` 把官方 50 任务评测改为**可复现协议**（4 worker，`env_seed = 1000 + off`），保留 `chunk_size=null`；
2. 至少跑 2 个 env seed（如 1000、2000），得到每个任务的稳定失败集合；
3. 只在“跨 env seed 稳定失败”的任务上做机制归因；若失败集合仍随 seed 大幅变化，应先报告协议方差，而不是改 H-JEPA；
4. 之后才考虑针对 B（局部运动）或 A（选路）的最小干预，并用独立开发任务集验证。

---

## 7. 复现命令

```bash
# 带 trace 的 Worker 1 重放（可复现）
cd h_jepa
PYTHONPATH=. python eval.py --config-name ant_l3 \
  seed=55 output.dir=eval_ant_l3_trace_w1_seeded eval.num_eval=13 \
  +eval.chunk_size=null +eval.env_seed=1000 \
  load_eval_trajs_path=/tmp/eval_slices_xx45h58h/slice_1.pt \
  policy=$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
  solver.solvers.level1.batch_size=2 solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 +eval.trace_dir=trace

# 分析（迷宫 BFS 距离、停滞、子目标变化、图）
PYTHONPATH=. python scripts/analyze_antmaze_trace.py \
  --trace .../eval_ant_l3_trace_w1_seeded/trace \
  --out .../eval_ant_l3_trace_w1_seeded/trace/analysis

# 门禁
PYTHONPATH=. python scripts/compare_trace_replay.py \
  --original-dir .../eval_ant_l3_par/w1 \
  --replay-dir .../eval_ant_l3_trace_w1
```

---

## 8. 交付物

| 文件 | 内容 |
|---|---|
| `h_jepa/antmaze_trace.py` | opt-in logger（默认 no-op） |
| `h_jepa/planning_eval.py` | 接入 `eval.trace_dir` / `eval.env_seed` |
| `stable_worldmodel/world.py` | `env_seed` 同时播种 reset 与 action space（默认关闭） |
| `h_jepa/tests/test_antmaze_trace.py` | 3 个无侵入性单测 |
| `h_jepa/scripts/analyze_antmaze_trace.py` | 轨迹/迷宫/子目标分析 + 图 |
| `h_jepa/scripts/compare_trace_replay.py` | 重放门禁 |
| `audits/antmaze_l3_trace_audit/summary_seeded.json` | 可复现重放的 per-task 指标 |
| `audits/antmaze_l3_trace_audit/summary_unseeded_replay.json` | 无种子重放的 per-task 指标 |
| `audits/antmaze_l3_trace_audit/gate_original_vs_unseeded.json` | 门禁失败证据 |
| `audits/antmaze_l3_trace_audit/determinism_check.json` | 确定性证明 |
| `audits/antmaze_l3_trace_audit/figures/` | overview + 5 个失败案例图 |
