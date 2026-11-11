# 自适应重规划 —— Step 1（log-only + 标定）状态记录

> 状态：**代码实现完成、链路已验证；全量标定因机器性能异常暂停**
> 日期：2026-11-11
> 对应实验：Push-T / `pusht_hjepa_l2` / seed 42 / 50 任务

## 1. 背景与目标

原始评测中，`pusht_l2` 的重规划周期固定为 **10 个环境步**
（`receding_horizon: 2` × `level1.action_block: 5`，见 `stable_worldmodel/policy.py` 的
`flatten_receding_horizon`）。

改进思路：**先执行第一个 5 步动作块，用世界模型的预测误差决定是否继续执行第二个块**。

$$
e_t=\frac{\lVert \hat z^{(1)}_{t+5}-z^{(1)}_{t+5}\rVert_2^2}{D\,\sigma_z^2+\epsilon},
\qquad
\xi_t=\begin{cases}5,&e_t>\tau\\ 10,&e_t\le\tau\end{cases}
$$

其中 $\hat z^{(1)}_{t+5}=F^{(1)}(z_t^{(1)},a_t^{(1)})$，$z^{(1)}_{t+5}=E^{(1)}(o_{t+5})$。

Step 1 只做 **`log_only`**：计算并记录 $e_t$，**不改变任何决策**，用于标定 $\tau$。

## 2. 关键事实核对

| 事实 | 依据 |
|---|---|
| 重规划周期确为 10 环境步 | `pusht_l2.yaml`：`receding_horizon: 2` × `level1.action_block: 5` |
| Level-1 时间步长 = 5 | `config/train/base/pusht.yaml`：`data.dataset.level1.frameskip: 5` |
| 一次 `predict` 恰好跨 5 环境步 | 与 `action_block: 5` 对齐，无需 5 次自回归 |
| $\hat z_{t+5}$ 可零成本获得 | `GradientSolver._rollout_final_actions` → `predictions['predicted_embed_0']` |

## 3. 代码改动

| 文件 | 说明 |
|---|---|
| `stable_worldmodel/policy.py` | `get_action` 拆分为 `_prepare_info` + `_next_action` 两个钩子，便于子类插入检查 |
| `h_jepa/adaptive_policy.py`（新增） | `AdaptiveWorldModelPolicy`：计算 $e_t$、写 JSONL、`log_only` 下不决策 |
| `h_jepa/hierarchical_solver.py` | 把 level-1 的 `predicted_embed_0` 透传给 policy（复用已有 rollout，零额外算力） |
| `h_jepa/planning_eval.py` | 依据 `adaptive` 配置选择策略类 |
| `h_jepa/config/eval/pusht_l2_adaptive.yaml`（新增） | 标定运行配置 |
| `h_jepa/scripts/calibrate_adaptive.py`（新增） | 从 JSONL 估计 $\sigma_z^2$、$e_t$ 分位数、各候选 $\tau$ 的触发率与额外规划调用 |

日志字段（`$HJEPA_HOME/adaptive_logs/pusht_l2_seed42.jsonl`，每条检查一行）：

```
check, t, call, env, D, mse, z_sq, e, score, trigger
```

## 4. 已验证

- **单环境标定运行成功**：1/1 任务成功，61.2 s，17 次求解**中位 2.20 s** —— 与基线同量级，说明改动本身无性能退化。
- **JSONL 正常产出**：16 条样本，$e_t$ 落在 `0.0088 – 0.042`。
- **`z_sq ≈ 0.99`**：SIGReg 正则使 latent 近似单位方差，因此 `norm: global` 下 $e_t \approx \text{mse}$，归一化天然成立。

## 5. 阻塞问题（暂停原因）

16 环境（`chunk_size=16`）下标定运行出现严重掉速：

| 运行 | 单次求解耗时 |
|---|---|
| 基线 50 任务，chunk16 | 68 次，中位 **3.51 s**，最大 17.07 s |
| `calib_smoke`（1 环境） | 17 次，中位 **2.20 s** |
| 标定 16 环境（adaptive） | 单次 **113.3 s / 34.1 s**（policy call 1 未跑完） |
| **同期基线对照**（16 环境） | 9 分钟内**一次求解都没完成** |

**判断**：同期基线对照同样慢，因此大概率**不是** adaptive 代码问题，而是机器级资源争用
（怀疑此前 4 进程并行评测的残留进程占用 GPU/CPU）。
停止时残留进程已消失、GPU 空闲，故该结论**尚未证实**。

**次生问题**：`nohup` 重定向导致 Python stdout 缓冲，日志严重滞后，放大了误判 —— 后续一律使用 `python -u`。

## 6. 恢复步骤

1. 先跑 16 环境**基线对照**（`pusht_l2`，`chunk_size=16`），确认单次求解回到 ~3 s 量级；
   若仍 > 20 s，说明主机仍有争用，先检查 `nvidia-smi` / `top`。
2. 重跑标定（上次 `calib_log/` 仅有 `eval_config.yaml`、无 chunk，需重跑）：

   ```bash
   cd h_jepa
   python -u eval.py --config-name pusht_l2_adaptive seed=42 output.dir=calib_log \
     policy="$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt"
   ```

3. 分析阈值：

   ```bash
   python scripts/calibrate_adaptive.py "$HJEPA_HOME/adaptive_logs/pusht_l2_seed42.jsonl"
   ```

4. 得到 $\tau$ 后切到决策模式（`log_only=false`）。在此之前需处理两个已知坑：
   - `planning_call_index` 用 `steps_taken // 10` 计算调用序号，**提前重规划会使 `decreasing_horizon_schedule` 的衰减表错乱**，应改用 `policy._policy_call_count`；
   - 向量化环境共享同一 action buffer，**per-env 异步触发需要 `trigger_mask`**；v1 建议先用 `chunk_size=1` 验证收益。

## 7. 基线成绩（对照）

`pusht_hjepa_l2 / seed42 / 50 任务 = 50.0%`，372.5 s（`eval_l2/metrics.yaml`），
成功任务平均 74.24 步。
