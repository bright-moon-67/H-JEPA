# AntMaze L3 开发集审计（PHASE 0–3）

日期：2026-11-13  
分支：`research/ant-trace-audit-v1`  
基线：`6c315c2`（上轮审计收尾）+ 本轮 `1cd38e0`（全局任务 seed、开发集、门禁）

---

## 0. 结论摘要

- PHASE 0：`eval.env_seed` 已按**全局任务 ID** 派生（`base + global_task_id`），`parallel_eval.py` 支持 `--env-seed-base` 并输出 manifest；tracer 记录 checkpoint/planner seed、batch 分组、初始状态与 goal 像素哈希、标准化/实际动作。
- PHASE 1：全预算（350 步）门禁 **17/17 通过**。trace off/on 的 success、步数、`n_iters` 一致；两次 trace-on 的逐步轨迹 SHA256、规划动作与 cost 曲线完全一致；初始 settle 状态可逐位复算。
- PHASE 2：12 个独立开发任务，settle seed 1000 → **8/12（66.7%）**，seed 2000 → **9/12（75.0%）**。
- PHASE 3：**7 个始终成功、2 个始终失败、3 个翻转（25%）**。稳定失败为 task 04（A，错误路线）与 task 09（B，终点精度）；翻转任务 1/3/10 表明初始物理状态可显著改变结果。L3 子目标余弦均值约 0.95、反转 0 次 → 本开发集**没有 C 类“规划不稳定”证据**，且高余弦与错误路线可以同时存在。

---

## 1. PHASE 0：确定性修复补全

### 1.1 全局任务 ID 映射

`stable_worldmodel/world.py::evaluate_from_dataset(..., env_seed=...)`：

```python
global_ids = ep_idx_arr  # payload["episodes_idx"]
seeds = [env_seed + gid for gid in global_ids]
for i, env in enumerate(self.envs.unwrapped.envs):
    env.action_space.seed(seeds[i])
```

- 同一任务无论落在哪个 worker / slice，`env_seed` 相同 → 初始物理状态相同；
- `reset(seed=seeds)` 也使用同一组值；
- 缺省 `None` 保持旧行为。

`planning_eval.py` 的分块路径不再对 `env_seed` 叠加 chunk 偏移（避免重复加全局 ID）。

### 1.2 `parallel_eval.py`

新增 `--env-seed-base`。启用后每个 worker 传 `+eval.env_seed=<base>`，并按全局任务 ID 生效。评测结束写出 `manifest.json`，包含：

- 每个 worker 的 `tasks`（全局 ID）、`planner_seed`、`env_seed_by_task`
- `batch_size`、`git_commit`、配置名、checkpoint 与任务文件路径

### 1.3 Trace 元数据补全

`meta.json` 新增/记录：

- `checkpoint_sha256`（流式哈希）
- `planner_seeds`（每层 `torch_gen.initial_seed()`）
- `batch_size_by_level`、`env_to_batch`
- `env_seed`、`env_seed_by_task`

`steps.jsonl` 的 `t=0` 记录额外包含 `hashes`：`qpos/qvel/xy/goal_xy/pixels/goal_pixels` 的 SHA256。每个环境步新增 `action_norm`（由 applied action 经 forward scaler 重建，供标准化动作比对）。

### 1.4 单元测试

`pytest tests`：43 passed（含 3 个 tracer 无侵入性测试）。

---

## 2. 独立开发任务集

生成器：`h_jepa/scripts/generate_antmaze_dev_tasks.py`  
文件：`h_jepa/audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt`  
SHA256：`0b954edce93c451d32553525d265b91ec9e85e31c19c2ac944280803e96383fa`

- 12 个任务，起点从全部可通行格均匀采样，目标格 BFS 距离固定为 3（与官方 `expert_grid_d3` 同分布）；
- 生成 seed `20261113`，显式排除官方 50 任务中出现的（起点,终点）组合（含反向）；
- 不依赖 OGBench expert 策略：`xy_neutral` 协议下起点会被重新 settle、goal 会按 `goal_xy` 重新渲染，专家轨迹片段只需提供首/末帧 XY；
- 任务对（`start_ij -> goal_ij`）：

| task | start | goal |
|---|---|---|
| 00 | (2,2) | (3,4) |
| 01 | (5,3) | (4,5) |
| 02 | (4,1) | (6,2) |
| 03 | (6,2) | (4,1) |
| 04 | (5,1) | (6,3) |
| 05 | (4,5) | (2,4) |
| 06 | (3,4) | (2,2) |
| 07 | (1,2) | (3,3) |
| 08 | (5,3) | (3,4) |
| 09 | (2,5) | (3,3) |
| 10 | (3,4) | (5,3) |
| 11 | (2,4) | (4,5) |

---

## 3. PHASE 1：全预算确定性门禁

配置：dev tasks 0–2，`eval_budget=350`，`seed=42`，`env_seed=1000`（每任务 1000/1001/1002），`batch_size=2`，原 `ant_l3` solver；trace-off 一次，trace-on 两次。

结果（`phase1_gate.json`，17/17 通过）：

| 检查 | 结果 |
|---|---|
| off vs on：success / steps / seeds | ✅ 一致（1/3 成功，task 02 在 331 步成功） |
| off vs on：`planning_compute` 记录数 / level / horizon / `n_iters` | ✅ 完全一致（95 条） |
| on_a vs on_b：success / steps | ✅ 一致 |
| on_a vs on_b：`steps.jsonl` SHA256 | ✅ 相同 |
| on_a vs on_b：规划动作 / cost 曲线 | ✅ 逐项相同 |
| meta：checkpoint hash / planner seeds / batch / env seeds / git commit | ✅ 相同 |
| 初始物理状态离线复算 | ✅ qpos/qvel 哈希与 trace `t=0` 完全一致 |

> 门禁脚本第一版用裸 `gym.make` 复算 settle 状态，得到不同 qpos（最大差 ~1.0）。原因是 eval 走 `MegaWrapper → SyncWorld → VariationWrapper` 链，settle 使用的 action space 与裸 `gym.make` 链并非同一对象/调用序列。改为与 `World` 完全相同的包装链后逐位一致。这也说明：**确定性复现必须复刻完整包装链**，不能只复刻 `reset(seed)`。

---

## 4. PHASE 2：12 任务 × 2 个 settle seed

两次运行除 `eval.env_seed` 外完全一致（seed=42、350 步、`batch_size=2`、同一 checkpoint、同一任务文件）：

| run | env_seed | 成功 | 成功率 | 成功步数均值 |
|---|---|---|---|---|
| `eval_dev12_env1000` | 1000（每任务 1000..1011） | 8/12 | 66.7% | 237.6 |
| `eval_dev12_env2000` | 2000（每任务 2000..2011） | 9/12 | 75.0% | 241.1 |

逐任务结果（`comparison_env_seeds.md`）：

| task | start→goal | seed 1000 | seed 2000 | 翻转 | 类别 1000/2000 | dmin | dfin | min\|xy−g\| | 最长停滞 | L3 cos |
|---|---|---|---|---|---|---|---|---|---|---|
| 00 | (2,2)→(3,4) | ✅ 244 | ✅ 223 | — | success | 0/0 | 0/0 | 0.99/0.99 | 7/11 | 0.96/0.92 |
| 01 | (5,3)→(4,5) | ✅ 285 | ❌ | success→fail | success/B | 0/1 | 0/1 | 0.99/4.54 | 9/40 | 0.97/0.96 |
| 02 | (4,1)→(6,2) | ✅ 288 | ✅ 227 | — | success | 0/0 | 0/0 | 0.91/0.99 | 8/10 | 0.99/0.98 |
| 03 | (6,2)→(4,1) | ❌ | ✅ 265 | fail→success | B/success | 2/0 | 2/0 | 7.34/0.94 | 87/6 | 0.97/0.98 |
| 04 | (5,1)→(6,3) | ❌ | ❌ | — | A/A | 3/3 | 4/4 | 8.80/8.56 | 8/7 | 0.98/0.97 |
| 05 | (4,5)→(2,4) | ✅ 197 | ✅ 250 | — | success | 0/0 | 0/0 | 0.97/0.99 | 5/8 | 1.00/0.99 |
| 06 | (3,4)→(2,2) | ✅ 163 | ✅ 255 | — | success | 0/0 | 0/0 | 0.99/0.96 | 5/8 | 0.99/0.99 |
| 07 | (1,2)→(3,3) | ✅ 318 | ✅ 282 | — | success | 0/0 | 0/0 | 0.99/0.93 | 6/5 | 0.99/0.99 |
| 08 | (5,3)→(3,4) | ✅ 178 | ✅ 242 | — | success | 0/0 | 0/0 | 0.98/0.96 | 11/7 | 0.96/0.96 |
| 09 | (2,5)→(3,3) | ❌ | ❌ | — | B/B | 0/1 | 1/1 | 2.19/3.38 | 10/11 | 0.97/0.97 |
| 10 | (3,4)→(5,3) | ❌ | ✅ 218 | fail→success | B/success | 0/0 | 0/0 | 2.33/0.97 | 6/6 | 0.99/0.99 |
| 11 | (2,4)→(4,5) | ✅ 228 | ✅ 208 | — | success | 0/0 | 0/0 | 0.96/0.90 | 6/7 | 0.98/0.96 |

汇总：

- **始终成功（7/12）**：0, 2, 5, 6, 7, 8, 11
- **始终失败（2/12）**：4（A/A）、9（B/B）
- **翻转（3/12 = 25%）**：1（成功→失败）、3（失败→成功）、10（失败→成功）
- 失败类别：seed 1000 = B×3 + A×1；seed 2000 = B×2 + A×1
- 规划统计（两次运行几乎一致，说明算力预算协议稳定）：
  - L1：36 次 solve，平均迭代 53.6 / 55.4，horizon ∈ {1,2,3,4}
  - L2：31 次 solve，平均迭代 33.1 / 32.8，horizon=2
  - L3：28 次 solve，平均迭代 40.6 / 36.9，horizon ∈ {2,3,4,5}
  - L3 子目标余弦：均值 0.954 / 0.947，最小 0.487 / 0.377，**反转 0 次**

---

## 5. PHASE 3：五个案例

案例图见 `figures/case_task_*`（左：迷宫轨迹与规划点；中：BFS 距离；右：各层子目标余弦）。

### 5.1 task 04（稳定 A：路线选择）

两个 seed 都失败，`dmin=3`、`dfin=4`（比初始更远），`min|xy−goal| ≈ 8.6–8.8`，无长停滞，L3 余弦 0.97–0.98。Ant 在错误走廊中持续移动，高层子目标却非常稳定。**这是“稳定的错误路线”**：高余弦不能证明路线正确。

### 5.2 task 09（稳定 B：终点精度）

两个 seed 都进入目标格附近（`dmin=0/1`），但 `min|xy−goal| = 2.19/3.38 > 1.0`，最终停在目标格内；停滞仅 10–11 步。属于**终点精度/局部控制**问题，不是选路错误。

### 5.3 task 01（success→fail：初始状态敏感）

seed 1000 在 285 步成功；seed 2000 只到 `min|xy−goal|=4.54`，出现 40 步停滞（起始格附近），L3 余弦仍为 0.96。**高层意图稳定，但低层运动未能执行**。

### 5.4 task 03（fail→success：局部运动失败）

seed 1000 出现 87 步停滞、`min|xy−goal|=7.34`；seed 2000 在 265 步成功。同一任务、同一 planner seed，仅 settle 初始姿态不同，结果翻转。**这是初始物理状态影响可执行性的直接证据**。

### 5.5 task 10（fail→success：终点精度）

seed 1000 进入目标格但 `min|xy−goal|=2.33`，seed 2000 成功（218 步）。与 task 09 同类，但对初始状态敏感。

---

## 6. 研究解读（谨慎）

1. **初始物理状态确实会改变任务结果**：12 个任务中 3 个（25%）在仅改变 settle seed 时翻转；task 03 从 87 步停滞变为成功，task 01 反之。这支持“AntMaze 的失败不仅是路线规划问题，也包含初始运动状态鲁棒性”的假设。
2. **稳定失败只有 2 个**：task 04（A，稳定错误路线）与 task 09（B，终点精度）。它们是目前最值得做机制归因的对象，但 **2 个 seed、12 个任务不足以宣称统计稳定**，只能算“可重复观察”。
3. **没有 C 类（高层规划不稳定）证据**：两个 seed 的 L3 余弦均值 0.95 左右、反转 0 次，即使 task 04 走错走廊时仍然稳定。这再次说明“子目标稳定”与“导航正确”是两件事。
4. **B 类内部至少有两种子机制**：局部停滞（task 01 seed 2000、task 03 seed 1000）与终点精度（task 09 两个 seed、task 10 seed 1000）。当前 A/B/C/D 分类粒度不足以区分，后续应拆分。
5. **对后续实验的硬约束**：任何方法比较都必须在相同 `env_seed`、相同任务、相同 planner seed 与 batch 分组下做配对；否则 25% 的翻转率会淹没真实效应。

---

## 7. 决定

- 本轮到此停止：未开发新算法、未调参、未使用官方 50 任务选择方法。
- 若继续，建议的下一个最小实验（需批准）：在同样的 12 个开发任务上增加 settle seed（例如 3000、4000），只统计 task 04 与 task 09 的配对结果，并把 B 类拆成“停滞/终点精度”两个子类；仍不进入方法开发。

---

## 8. 复现与产物

命令（两次运行只差 `+eval.env_seed`）：

```bash
cd h_jepa
PYTHONPATH=. python eval.py --config-name ant_l3 seed=42 \
  output.dir=eval_dev12_env1000 eval.num_eval=12 +eval.chunk_size=null \
  +eval.env_seed=1000 eval.eval_budget=350 \
  load_eval_trajs_path=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
  policy=$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
  solver.solvers.level1.batch_size=2 solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 +eval.trace_dir=trace
```

| 文件 | 内容 |
|---|---|
| `audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.{pt,sha256,json}` | 独立开发任务与哈希 |
| `audits/antmaze_dev_audit/phase1_gate.json` | PHASE 1 全预算门禁（17 项） |
| `audits/antmaze_dev_audit/metrics_env{1000,2000}.yaml` | 两次运行指标 |
| `audits/antmaze_dev_audit/summary_env{1000,2000}.json` | per-task 诊断 |
| `audits/antmaze_dev_audit/comparison_env_seeds.{json,md}` | 配对比较 |
| `audits/antmaze_dev_audit/figures/` | 两个 overview + 5 个案例 |
| `scripts/generate_antmaze_dev_tasks.py` | 任务生成器 |
| `scripts/compare_dev_gate.py` | PHASE 1 门禁 |
| `scripts/compare_dev_seeds.py` | 跨 seed 配对比较 |

---

## 9. 禁止项遵守

本轮未修改 checkpoint、未改 GradientSolver、未迁移 RAGC/ARN、未训练 RL、未运行反馈策略、未扫描超参数、未用官方 50 任务选择算法、未根据 seed 结果追加样本。

