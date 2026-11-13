# CLFB V1：跨层可达性引导的 L3 束规划

日期：2026-11-13  
分支：`research/ant-clfb-v1`（基线 `48ef1c2`）  
状态：Stage A 通过（15/15）；**Stage B NO-GO**（net −2，destroyed 5，wall 1.64×）。

---

## 1. 方法（固定配置）

CLFB 不训练、不改世界模型、不改执行周期，只在 L3 的 400 个**已优化**候选中改变最终选择：

1. `GradientSolver.solve(return_candidate_pool=True)` 额外返回按最终 cost 排序的 Top-M（M=32）候选动作、cost 与索引；原始 `actions` 仍是 Top-1。
2. 用冻结 L3 模型对 Top-M 动作做一次 rollout，得到完整预测潜变量轨迹。
3. 在潜变量轨迹（去掉 context frame）上做 greedy farthest-first，选 K=4 条：
   - 必含原始最优（索引 0）；
   - 其余最大化与已选轨迹的余弦距离。
4. 对每条候选，取 L3 预测的第一个子目标与对应宏动作，按 `HierarchicalSolver` 完全相同的时序/anchoring 语义构造 L2 目标。
5. 独立 L2 探测：每条候选 16 samples × 30 GD steps，使用原 `LevelCostModel.get_cost`、原 optimizer 类与 LR、独立 `torch.Generator`；不调用 live L2 `.solve()`，不修改 `.init` / `.torch_gen` / `solve_records`，不累积模型参数梯度。
6. 归一化排名评分：$S_i = \tfrac12 r_3(i) + \tfrac12 r_2(i)$，$r_3$ 为 Top-M 内成本排名，$r_2$ 为 K 条内 L2 探测 cost 排名；取最小者；平局/非法探测保留原始 Top-1。
7. 若切换，**同时**替换 L3 选中动作与其预测潜变量；L2/L1 仍走原完整求解流程。

不使用真实 XY、迷宫 BFS、仿真 success、RAGC/ARN head。探测不改变正式 solver 的 RNG 或状态。

## 2. Stage A：工程门禁（15/15 通过）

配置：dev tasks 0–2，`eval_budget=350`，`seed=42`，`env_seed=1000`，`batch_size=2`，trace on。

| 检查 | 结果 |
|---|---|
| `beam_width=1` vs baseline：success / 步数 / planning `n_iters` | ✅ |
| `beam_width=1` vs baseline：逐步 trace SHA256 | ✅ |
| `force_original`（probe 全跑，强制原方案）vs baseline：success / 步数 / `n_iters` / 逐步 SHA256 | ✅ |
| `force_original` 记录：全部 `switched=false`，cost 有限，4 条 beam 互异 | ✅ |
| `beam_width=1` 不产生选择记录 | ✅ |
| 真实模型冒烟：Top-32 pool、4 beam、L2 probe cost、任务映射、切换逻辑 | ✅（1 任务，2/4 次调用切换，beam 最小潜变量距离 0.306） |

证据：`stage_a_gate.json`、`smoke_records.json`。

已知开销（3 任务、350 步）：probe 总 wall 302.4s；CLFB 760.2s vs baseline 451.2s = **1.685×**（3 环境、`batch_size=2` 时探测的 GPU 利用率偏低；12 环境下的比值见 Stage B）。

## 3. Stage B：24 Episode 配对 Pilot（结果）

固定开发集：`audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt`，12 任务 × env_seed 1000/2000；原 checkpoint、planner seed=42、task 顺序与 `batch_size=2` 分组；CLFB 配置为 `top_m=32, beam_width=4, probe 16×30, seed=420042`。Baseline = 17/24。

| seed | baseline | CLFB | rescued | destroyed | wall (base→CLFB) | 比值 |
|---|---|---|---|---|---|---|
| 1000 | 8/12 | 7/12 | 3, 10 | 1, 2, 11 | 1444s → 2321s | 1.607× |
| 2000 | 9/12 | 8/12 | 9 | 7, 8 | 1401s → 2293s | 1.637× |
| **合计** | **17/24** | **15/24** | **3** | **5** | — | **1.637× (max)** |

- 净成功数：**−2**（目标 ≥ +2）
- 破坏原成功任务：**5**（目标 ≤ 1）
- 规划计算开销：**1.637×**（目标 ≤ 1.5×）
- 高层切换：**250 次**（56 次规划调用 × 12 环境 = 672 个决策，37.2% 被切换）；两个 seed 各 125 次
- 切换时的模型代价变化（均值）：
  - L3 cost：**+112.3**（如预期，放弃了 L3 最优）
  - L2 probe cost：**−5.62**（代理可达性确实变好）
  - L1 最终规划 cost：**40.95 → 43.90**（实际执行层反而更差）
- 逐任务切换次数（env1000）：task 0:10, 1:13, 2:16, 3:11, 4:5, 5:14, 6:7, 7:7, 8:9, 9:8, 10:11, 11:14

复现命令（两次仅 `eval.env_seed` 与输出目录不同）：

```bash
cd h_jepa
PYTHONPATH=. python eval.py --config-name ant_l3 seed=42 \
  output.dir=eval_clfb_dev12_env1000 eval.num_eval=12 +eval.chunk_size=null \
  +eval.env_seed=1000 eval.eval_budget=350 \
  load_eval_trajs_path=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
  policy=$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt \
  solver.solvers.level1.batch_size=2 solver.solvers.level2.batch_size=2 \
  solver.solvers.level3.batch_size=2 +eval.trace_dir=trace \
  +clfb.enabled=true +clfb.top_m=32 +clfb.beam_width=4 \
  +clfb.probe_samples=16 +clfb.probe_steps=30 +clfb.seed=420042
```

## 4. 结论：NO-GO

三项预设门槛全部未通过：

| 门槛 | 要求 | 实测 | 结果 |
|---|---|---|---|
| 净成功数 | ≥ +2 | −2 | ❌ |
| 破坏原成功 | ≤ 1 | 5 | ❌ |
| 规划开销 | ≤ 1.5× | 1.637× | ❌ |

按预先约定，**停止 CLFB V1，不调 beam width / top-M / 排名权重 / probe 步数或样本数**，也不在官方 50 任务上重跑。

机制层面的观察（仅作记录，不作为调参依据）：

1. L2 可达性代理确实在切换决策上变好（−5.62），但闭环结果更差（rescued 3 vs destroyed 5）——**模型内可达性 ≠ 真实可执行性**。
2. 37.2% 的决策被切换，切换过于频繁；L1 最终 cost 反而升高，说明被选中的 L3 方案在真实执行层并不更容易。
3. CLFB 在 task 04、09 这类稳定失败上没有针对性逻辑（遵守约束），也未救回它们；被救回的是 3、10、9，被破坏的是 1、2、11、7、8。
4. 计算开销主要来自 4 条 beam 的 16×30 L2 探测（probe wall 合计 1837s），在 `batch_size=2` 下 GPU 利用率低。

代码保持默认关闭（`clfb` 缺省即原路径），Stage A 已证明 `beam_width=1` 与 `force_original` 均逐动作复现 baseline。

## 5. 预设门槛（已用于判定）

- 净成功数 ≥ +2
- 破坏原成功任务 ≤ 1
- 规划计算开销 ≤ 1.5×（以同任务端到端 wall time 比值计）

任一不满足 → NO-GO，停止该机制，不调 beam width / top-M / 权重 / probe 步数或样本数。

## 5. 禁止项遵守

未重训 H-JEPA、未启用 RAGC/ARN、无 RL、未改执行周期、未改原 GradientSolver 参数、未针对 task 04/09 写逻辑、未运行官方 50 任务。
