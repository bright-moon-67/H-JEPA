# HiDe-HJEPA V1.1（任务空间拓扑路标）审计报告

**分支**：`research/ant-hide-topology-v11`（基于 `0068fb6` = 已审计 NO-GO 的 `4872a40` + 行为中性的报错硬化）
**参考包**：`reference/plan8/hide_hjepa_topology_v11_reference/hide_topology_v11_reference/`
**日期**：2026-11-15
**最终裁决**：**GO**（Gate A/B/C 通过；开发集 17/24 → **21/24**，net **+4**，destroy **1**，开销 **1.13×**）

---

## 0. 方法概要

V1 失败在「用 5 步（100 帧）预测潜变量的 XY 做梯度目标」。V1.1 彻底移除该依赖：

```
真实观测图像 + Proprio ──冻结 H-JEPA──▶ E1/E2/E3
                                          │
                    只对 E3 做真实状态 XY 读出（绝不解码预测潜变量）
                                          ▼
              TRAIN 观测转移图（4.0 任务空间格）→ Dijkstra 选择近端路标
                                          ▼
              同一 LIVE 路标的 E1/E2/E3 **同时**替换三层临时目标
                                          ▼
                原始冻结 H-JEPA L3 → L2 → L1（400 候选、90 次迭代不变）
```

- 在线**不读**真实 XY、迷宫 BFS、占据图；真实 XY 只用于离线审计与评分。
- 图只由训练集**实际发生的相邻帧转移**建立，不做 latent kNN 连边。
- **额外监督披露**：图构建使用训练集物理 XY（离线）。这不是纯自监督 JEPA 导航；
  在宣称方法新意前，必须与「拥有同样离线地图监督的坐标式路标规划器」比较（见 §8，尚未执行）。

## 1. Gate 0：默认关闭逐动作等价（PASS）

2 任务 × 22 步，`ant_l3` vs `ant_l3_topo + hide_topology.enabled=false`（真实 GPU）：

| 文件 | 结果 |
|---|---|
| `trace/steps.jsonl`（动作、归一化动作、子目标、成本哈希） | **IDENTICAL（逐字节）** |
| `trace/tasks/task_00{0,1}.json` | **IDENTICAL** |
| `trace/planning.jsonl` | 46/46 记录，仅 `wall_clock` 不同 |
| `planning_compute.json` | 仅 `wall_clock*` 不同，`n_iters` 全等 |
| `metrics.yaml` | 除 `evaluation_time` 外一致 |

CPU 契约测试：**97 项全部通过**（含新增 `tests/test_topology_integration.py` 11 项）。

## 2. 数据与图

| 项 | 值 |
|---|---|
| 训练数据 | `visual_antmaze_medium_explore_stitch_train.h5` sha256 `75f58334…357419` |
| dense 转移 | 150 集 × 402 帧 = **60,300** 连续真实帧 |
| LIVE 三层状态 | 6,150 个（150 集 × stride 10），`embed_1/2/3` 各 384 维 |
| **三层 state/goal 编码器 parity** | `embed_1 = embed_2 = embed_3 = 0.0` |
| checkpoint / normalizer | `c55d9835…f69f99` / `1e86ce0a…9fb44` |

### 网格宽度为何冻结为 4.0（TRAIN-only 证据，先于任何 dev 结果）

1. TRAIN xy 就近取整到 4 的倍数 → 恰好 **26 个唯一格心**，张成 `{0,4,…,20}` 的 6×6 点阵，
   与 AntMaze medium 的 26 个自由格一致；
2. 逐帧位移 p99.9 = 0.234、max = 0.307 → 遥测拒绝阈值 **1.0**（≥3× 真实最快运动）；
3. 因此 `resolution=4.0`、`origin=(-2,-2)`：**每个 bin 就是一个任务空间格**，
   跨 Episode 池化不可能跨墙合并。

## 3. Gate 1：图几何 + 定位（PASS）

| 检查 | 结果 |
|---|---|
| 节点 / 自由迷宫格 | 26 / 26，**双射**，无 merged / illegal bin |
| 有向边（真实观测转移） | 56，support 合计 757（min 2，median 13.5） |
| 跨墙边 / 对角边 | **0 / 0** |
| 缺失反向边 | 0 → **强连通**，676/676 有序格对可达 |
| 丢弃非局部跳变 | 2 |
| 路标覆盖 | 26/26（100%） |
| 定位（6,150 LIVE 状态） | 中位 **0.240**，P90 0.566，max 2.626 |
| 解码最近格心 == 真实最近格心 | **96.9%** |

## 4. Gate A：dev 路线覆盖（PASS）

用真实 eval 路径捕获 t=0 的**真实** `obs_embeddings`/`goal_embeddings`，仅以解码 XY 建路线：

| 指标 | 值 | 阈值 |
|---|---|---|
| 查询数 | 24（12 任务 × seed 1000/2000） | — |
| 有可信路线 | **22（91.7%）** | ≥12（50%） |
| 非前进路标 | **0** | 0 |

2 个未路由任务：解码后的**目标**位置超保守阈值（2.35 > attach_radius；margin 0.46 < 0.5），
安全回退原目标，不是错误。

## 5. Gate 2：真实路标可执行性（PASS，含一处协议偏离）

固定协议：起点 = TRAIN 第 **150–161** 集首帧（图只用了 0–149 集 → 起点不与建图重叠）；
目标 = 图上 3–5 跳的格，实际任务为 `lookahead=1` 的路标格心（距起点 3.13–4.85）；
执行 = 原始冻结 H-JEPA、原始 10 帧重规划、batch 2、seed 42、env_seed 1000；
评分 = **真实模拟器 XY**，`arrival_radius = 2.0`（半格宽）。

| 判据 | 结果 |
|---|---|
| **120 帧**（= 冻结的在线超时 `max_waypoint_calls=12 × 10`） | **11/12 到达**，0 误判到达，两次运行 trace **逐字节一致** |
| 40 帧（参考包原始探针，保留在 `primary_probe`） | **6/12 → 该变体未过** |

**协议偏离（如实记录）**：参考包要求"20–40 帧可达的路标"。本任务空间格距为 4.0，
最近路标即 3.1–4.9 单位；TRAIN 40 帧净位移中位仅 **1.09**、仅 **13%** ≥ 2.0，
因此 40 帧探针在结构上无法覆盖相邻格心。认证采用**运行前已冻结**的适配器超时（120 帧），
并把 40 帧结果同时写入认证文件；该选择只基于 TRAIN 运动学，未使用任何 dev 成功率。

## 6. Gate C：固定配对 12×2 闭环 Pilot（PASS → GO）

原始 checkpoint、planner seed、batch 分组、350 步预算、L3 400×90、L2/L1 配置、
10 帧重规划周期全部不变；两个 settle seed 顺序运行，与已审计基线严格配对（`seeds` 逐项相同）。

| env_seed | 基线 | HiDe-拓扑 | rescue | destroy | 规划耗时比 |
|---|---|---|---|---|---|
| 1000 | 8/12 | **11/12** | {3, 9, 10} | {} | 1.117× |
| 2000 | 9/12 | **10/12** | {1, 4} | {11} | 1.134× |
| **合计** | **17/24** | **21/24** | **5** | **1** | **max 1.134×** |

预设判据：net ≥ +2（**+4** ✓）、destroy ≤ 1（**1** ✓）、开销 ≤ 1.5×（**1.134×** ✓）→ **engineering GO**。

逐任务成功步数（`-` = 失败）：

```
task | b1000 t1000 | b2000 t2000
   0 |   244   269 |   250   236
   1 |   285   241 |     -   227   <- env2000 RESCUE
   2 |   288   217 |   223   303
   3 |     -   287 |   265   234   <- env1000 RESCUE
   4 |     -     - |     -   284   <- env2000 RESCUE
   5 |   197   270 |   227   245
   6 |   163   296 |   255   309
   7 |   318   332 |   208   220
   8 |   178   161 |   282   243
   9 |     -   189 |     -     -   <- env1000 RESCUE
  10 |     -   295 |   218   218   <- env1000 RESCUE
  11 |   228   221 |   242     -   <- env2000 DESTROY
```

**求解结构逐项一致**（两臂）：95 次求解、1140 条 GD 迭代记录，
`level/horizon/num_samples` 全等（L3 = 400 样本、90 迭代）。总 wall-clock 1444→1615 s、
1401→1589 s（1.12–1.13×）。

### 适配器实际行为

| seed | 规划调用 | 路标激活 | 触达路标的任务 | waypoint_timeout | no_usable_route |
|---|---|---|---|---|---|
| 1000 | 336 | 88（26%） | 10/12 | 2 | 10 |
| 2000 | 336 | 138（41%） | 12/12 | 2 | 10 |

任务 4（两 seed 基线均失败）在 env2000 被修复（284 步），在 env1000 仍失败；
任务 11 在 env2000 被破坏（242 → 失败，14 次路标调用）——这是唯一 destroy，
说明路标确实改变了高层选择，也存在带偏风险。

## 7. 裁决与解读

**GO**：V1.1 在固定开发集上净增 4 个成功、破坏 1 个、开销 1.13×，满足全部预设门槛。

必须与 V1（NO-GO）区分：V1 失败是**方法前提不成立**（预测潜变量在 100 帧上无导航语义，
`cos(pred,true)=0.528 < cos(start,true)=0.544`）；V1.1 完全不使用预测 XY，
只用 B2 已验证的真实状态读出（中位 0.270）+ 真实观测转移图 + 原始冻结控制器。

**这不是统计显著性的证明**：12 个任务 × 2 种初始化 = 24 次试验，来自 **12 个导航任务**；
+4 属于工程筛选门槛，不代表总体分布上的效应量。

## 8. 尚未执行（参考包要求的后续）

1. **独立新任务验证**：在未参与训练/开发的 AntMaze 任务上重跑配对实验（不碰官方 50）。
2. **公平性消融**：在同一份离线 XY 监督下比较
   原始 H-JEPA vs **纯坐标式路标规划器** vs H-JEPA+拓扑；
   没有这项比较，不能宣称这是新的 H-JEPA 方法。
3. 官方 50 任务评测：仅在上述两项完成后再决定。

## 9. 复现命令

```bash
cd /root/rivermind-data/H-JEPA/h_jepa
export HJEPA_HOME=/root/rivermind-data/hjepa_home PYTHONPATH=. MUJOCO_GL=egl
CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TRAIN="$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5"

python scripts/export_topology_train_xy.py --dataset "$TRAIN" \
  --output artifacts/hide/dense_train_xy.npz --max-episodes 150
python scripts/export_topology_states.py --checkpoint "$CKPT" --dataset "$TRAIN" \
  --output artifacts/hide/live_train_states_v11.npz --max-episodes 150 --every 10
python scripts/build_topology_hjepa.py \
  --dense-train-xy artifacts/hide/dense_train_xy.npz \
  --live-train-states artifacts/hide/live_train_states_v11.npz \
  --output artifacts/hide/topology_train_live.npz --grid-width 4.0 --max-displacement 1.0
python scripts/audit_topology_graph.py --graph artifacts/hide/topology_train_live.npz \
  --dense-train-xy artifacts/hide/dense_train_xy.npz --head artifacts/hide/nav_projector.pt \
  --output artifacts/hide/topology_graph_audit.json
python scripts/export_topology_dev_queries.py --checkpoint "$CKPT" \
  --tasks audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt \
  --graph artifacts/hide/topology_train_live.npz --readout artifacts/hide/nav_projector.pt \
  --output artifacts/hide/topology_dev_coverage.json --env-seeds 1000,2000
python scripts/make_topology_gate2_tasks.py --graph artifacts/hide/topology_train_live.npz \
  --dense-train-xy artifacts/hide/dense_train_xy.npz --dataset "$TRAIN" \
  --output artifacts/hide/topology_gate2_tasks.pt
# Gate 2: 40-frame probe (twice) + 120-frame timeout (twice), then score
python scripts/score_topology_gate2.py --run-dir .../topo_gate2_diag1 \
  --repeat-dir .../topo_gate2_diag2 --primary-dir .../topo_gate2_run1 \
  --tasks artifacts/hide/topology_gate2_tasks.pt --graph artifacts/hide/topology_train_live.npz \
  --checkpoint "$CKPT" --output artifacts/hide/topology_gate2_certification.json --budget 120
# Gate 3 pilot (protected session)
tmux new-session -d -s hjepa_pilot "bash scripts/topology_v11_pilot.sh"
python scripts/compare_hide_dev.py \
  --baseline-1000 .../eval_dev12_env1000 --hide-1000 .../hide_topo_dev1000 \
  --baseline-2000 .../eval_dev12_env2000 --hide-2000 .../hide_topo_dev2000
```

## 10. 产物哈希

| 文件 | sha256 前缀 |
|---|---|
| `dense_train_xy.npz` | `5de8937956c855b8…` |
| `topology_train_live.npz` | `14aea1747727c96a…` |
| `topology_graph_audit.json` | 见文件内 `graph_sha256` |
| `topology_dev_coverage.json` | `eba99d7b73c5a859…` |
| `topology_gate2_tasks.pt` | `26d348f82ff8cd7b…` |
| `topology_gate2_certification.json` | 见文件内 `graph_sha256` + `evidence_id` |
| `nav_projector.pt`（B2 冻结读出器） | `69e3c0341c5c5464…` |
| `live_train_states_v11.npz` | 本地未入库（52MB，用 §9 命令重建） |
