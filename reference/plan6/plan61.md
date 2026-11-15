# HiDe-HJEPA：面向 AntMaze 的规划与控制功能解耦方案

建议新分支：`research/ant-hide-decouple-v1`

我建议正式停止 PAHRL V1 的扩展，在已验证的 `research/ant-trace-audit-v1` 分支上重新开展 HiDe 方向，而不是继续从 PAHRL 分支叠加功能。

这次研究的核心目标很明确：

> 让 H-JEPA 的高层优化“导航目标与路线进展”，让低层优化“身体运动与动作执行”，减少高层导航目标被无关身体姿态影响的问题。

但有一个必须认识到的事实：原始 H-JEPA 本身已经是 L3→L2→L1 分层规划架构。仅仅把现有代码封装成 `Planner` 和 `Controller` 两个类，不构成真正的 HiDe 改进。

HiDe 原论文 Learning Functionally Decomposed Hierarchies for Continuous Control Tasks With Path Planning 强调的是不同层级使用不同的任务状态、动作空间及目标语义。它的高层使用环境布局信息和规划价值图，低层通过本体状态学习目标条件控制；两层在原论文中还会联合训练。

![](https://www.google.com/s2/favicons?domain=https://files.ait.ethz.ch\&sz=32)

AIT 文件库+1

因此，我们的第一版应采用冻结世界模型 + 轻量导航表示 + 分层目标函数解耦。这是 HiDe-inspired 改造，并非完整复现 HiDe。

## 一、项目目标与不变条件

当前确定性 AntMaze L3 开发集基线

# 17/24

70.8%

12 个开发任务 × 环境种子 1000、2000；原官方 50 任务曾达到 38/50，但其旧版初始化不能作为严格配对复现实验。

本次必须保持以下条件不变：

* 冻结原始 AntMaze L3 H-JEPA Checkpoint，包括三个 Encoder、Predictor 及动作编码模块。

* 保持 L3 400 候选、90 次优化，L2/L1 原始求解参数及 Horizon Schedule。

* 保持每 10 个环境步重规划一次、每 Episode 最多 350 步。

* 不启用 PAHRL、CLFB、RAGC、ARN 或 Feedback-N。

* 训练辅助模块只使用训练集，不使用开发集和官方测试集优化参数。

* 在线算法不读取真实 `xy`、墙体占据图、BFS 路线或环境奖励来选择动作。

最终唯一的主要优化目标仍是闭环成功率。 导航表示质量、代价下降和路标稳定性只作为前置门禁与故障诊断。

## 二、总体架构：高层导航目标与低层运动控制分离

当前图像 + Proprio + 最终目标图像

原始 H-JEPA 输入不变

冻结的 H-JEPA Encoder 与 Predictor

产生原始 L1 / L2 / L3 表征及预测

新增 Navigation Projector

从完整 L3 Latent 提取与导航位置相关的任务空间表示

L3 导航规划

重点优化导航进展，弱化关节姿态干扰

L2 局部跟踪

跟踪上层导航意图，保留完整预测状态

L1 运动控制：完全保留原始规划器

图像、身体状态、动力学和实际关节动作

原始 AntMaze 闭环环境

### 最核心的改变

目前 H-JEPA 的高层与低层使用不同时间尺度的 Latent，但这些 Latent 来自同一套包含图像和本体感觉的表征体系。L3 求解的目标仍是与指定 Goal Latent 尽量接近。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv+1

我们要引入一个明确的导航任务表示：

znav=Pϕ(z(3))z_{\mathrm{nav}}=P_\phi(z^{(3)})znav=Pϕ(z(3))

其中：

* z(3)z^{(3)}z(3) 是冻结 H-JEPA 的 L3 状态表征；

* PϕP_\phiPϕ 是新增的小型导航投影器；

* znavz_{\mathrm{nav}}znav 重点表示机器人在环境中的导航位置，而非完整身体状态。

第一版建议用 2 维导航坐标作为训练监督，因为 AntMaze 的实际目标就是二维位置到达。

这里要强调：训练时允许使用训练轨迹的 XY 作为监督标签，但推理时只能输入图像与正常的 Proprio，绝对不读取评估环境的真实 XY。

这与直接把 `xy` 作为规划器输入有本质区别。

同时，不能简单删除 L3 输入中的 Proprio，也不能把 384 维 Latent 硬截成某几个维度。那会破坏冻结世界模型的输入分布和训练语义。

正确的 V1 是保持世界模型完整，只对规划目标的表示空间进行解耦。

## 三、核心算法：Navigation Projector + 双层目标解耦

### 1. 学习导航表示，而不是重新训练世界模型

从现有训练数据中获取：

(zt(3),pt),pt=(xt,yt)(z_t^{(3)},p_t),\qquad p_t=(x_t,y_t)(zt(3),pt),pt=(xt,yt)

训练一个很小的投影网络：

p^t=Pϕ(zt(3))\hat p_t=P_\phi(z_t^{(3)})p^t=Pϕ(zt(3))

建议 V1 使用：

Python

运行

```
class NavigationProjector(torch.nn.Module):
    def __init__(self, latent_dim=384):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, z):
        return self.net(z)
```

监督损失为：

Lnav=E\(∥Pϕ(zt(3))−pt∥22\)\mathcal L_{\mathrm{nav}} = \mathbb E\left\( \\|P\_\\phi(z\_t^{(3)})-p\_t\\|\_2^2 \\right\)Lnav=E\(∥Pϕ​(zt(3)​)−pt​∥22​\)

这里的 `384` 是当前 L3 表征维度。模型输入必须与在线 `_encode_all_levels()` 的输出保持一致。

但基于 PAHRL 的失败经验，不能直接缓存 HDF5 的 L3 Latent 后就把投影器拿去在线使用。

建议采用以下固定数据协议：

|
数据

|

用途

|

上限

|
| --- | --- | --- |
|

训练 Episode 中实时重渲染的状态

|

训练导航投影器

|

4,000 个状态

|
|

不重叠训练 Episode 中实时重渲染的状态

|

选择训练 Epoch / 验证

|

1,000 个状态

|
|

再独立留出的训练来源 Episode

|

最终离线测试

|

1,000 个状态

|
|

相同物理状态对应的原始 HDF5 图像

|

衡量渲染域偏移

|

上述状态的一部分

|

按 Episode 而不是随机帧切分，避免同一轨迹相邻帧同时进入训练和验证。

训练最多 30 Epoch，固定一个训练随机种子；可额外训练线性 Probe 作诊断，但不进行隐藏层宽度、学习率或损失权重搜索。

原始 H-JEPA 的仓库本身已有 AntMaze probing 配置，包含 `xy`、`qpos`、`qvel`、`proprio` 字段，可以复用其数据组织和归一化习惯。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub

### 2. 不只检验真实状态，还要检验预测状态

这一点非常重要。

我们真正要优化的是 H-JEPA Predictor 产生的未来 Latent，而不仅是 Encoder 处理的真实图像。

因此需要检验：

Pϕ(z^t+h(3))与pt+hrealP_\phi(\hat z_{t+h}^{(3)}) \quad\text{与}\quad p_{t+h}^{\mathrm{real}}Pϕ(z^t+h(3))与pt+hreal

是否保持一致的导航意义。

可以用训练集内留出的真实动作序列，按照 L3 的正确时间跨度进行预测，再比较预测导航位移与实际发生的导航位移。

不能把 L3 的一步直接当作一个环境步；它包含由较低层时间步长和动作块构成的宏时间尺度。

离线门禁建议预先固定为：

* 实时渲染测试集上，导航位置误差中位数低于环境成功半径，90% 分位误差低于两个成功半径；

* 对真实未来状态和预测未来状态，至少能可靠判断前进方向以及是否朝目标取得进展；

* 相同位置、不同姿态的导航投影应保持接近；

* 训练与推理使用完全相同的 Checkpoint、Normalizer 和编码路径。

如果真实状态能准确定位，但预测 Latent 无法保持导航位置意义，就停止直接用该投影器指导梯度优化。 否则梯度求解器可能利用投影器的误差，产生预测导航代价很低、实际动作却无效的方案。

### 3. L3：改变导航代价，不改变世界模型预测

原始 L3 代价记作：

J3base=d3(z^t+H(3),zg(3))J_3^{\mathrm{base}}= d_3(\hat z_{t+H}^{(3)},z_g^{(3)})J3base=d3(z^t+H(3),zg(3))

新增导航代价：

J3nav=∥Pϕ(z^t+H(3))−Pϕ(zg(3))∥22J_3^{\mathrm{nav}} = \left\| P_\phi(\hat z_{t+H}^{(3)}) - P_\phi(z_g^{(3)}) \right\|_2^2J3nav=Pϕ(z^t+H(3))−Pϕ(zg(3))22

V1 不建议完全扔掉原始 Latent Cost，因为仅优化 XY 距离可能重新制造“直线穿墙”的倾向。

因此首版采用有保护的软解耦：

J3HiDe=J3base+α3 s3J3nav\boxed{ J_3^{\mathrm{HiDe}} = J_3^{\mathrm{base}} + \alpha_3\,s_3 J_3^{\mathrm{nav}} }J3HiDe=J3base+α3s3J3nav

其中 α3=0.25\alpha_3=0.25α3=0.25 是预先固定的研究起点，s3s_3s3 是仅通过训练数据计算的原始代价与导航代价尺度比值，不能用开发集成功率调节。

这个设计让 L3 开始明确强调导航位置，同时保留原模型的其他约束。它是渐进式功能解耦，不是完全独立的 HiDe 高层规划器。

### 4. L2：跟踪导航目标，而不是强迫所有身体状态与 L3 匹配

当前 `HierarchicalSolver.solve()` 会取 L3 的第一个预测 Latent，作为 L2 要跟踪的上层子目标；L2 通过 `LevelCostModel._project_to_goal_space()` 把自身预测投影到 L3 表征空间后计算误差。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

这里正是第二个适合解耦的位置。

对于 L2：

J2HiDe=J2base+α2 s2∥Pϕ(Proj⁡2→3(z^(2)))−Pϕ(zsubgoal(3))∥22\boxed{ J_2^{\mathrm{HiDe}} = J_2^{\mathrm{base}} + \alpha_2\,s_2 \left\| P_\phi(\operatorname{Proj}_{2\to3}(\hat z^{(2)})) - P_\phi(z_{\mathrm{subgoal}}^{(3)}) \right\|_2^2 }J2HiDe=J2base+α2s2Pϕ(Proj2→3(z^(2)))−Pϕ(zsubgoal(3))22

同样建议先固定 α2=0.25\alpha_2=0.25α2=0.25。

而 L1 保持完全不变：它仍然接收原始完整状态与原始 L2 预测的完整 Latent，继续处理关节运动、姿态和精细动作。

这样可以形成较清晰的职责分工：

|
层级

|

主要负责

|

V1 的修改

|
| --- | --- | --- |
|

L3

|

朝任务目标进行高层导航规划

|

在原始 Cost 上加入导航任务空间 Cost

|
|

L2

|

将高层意图转化为局部可执行运动

|

在跟踪 L3 子目标时加入导航一致性 Cost

|
|

L1

|

原始身体控制与动作执行

|

不变

|

关键点：L3→L2 仍传递完整的预测 Latent，不传递一个孤立的二维 XY。 Navigation Projector 只在比较目标代价时使用，不破坏 H-JEPA 已训练的 Latent Dynamics 接口。

## 四、具体需要修改哪些代码？

已核对当前公开分支的代码结构，主要修改位置如下。

|
文件

|

操作

|

内容

|
| --- | --- | --- |
|

`h_jepa/hide_nav.py`

|

新增

|

`NavigationProjector`、导航代价、模型及数据哈希校验

|
|

`h_jepa/scripts/train_hide_nav.py`

|

新增

|

离线训练导航投影器，只更新新增 Head

|
|

`h_jepa/scripts/audit_hide_nav.py`

|

新增

|

真实状态、预测状态、渲染域一致性评估

|
|

`h_jepa/hierarchical_solver.py`

|

小幅修改

|

给 L3/L2 的 `LevelCostModel` 接入可选导航代价

|
|

`h_jepa/planning_eval.py`

|

小幅修改

|

从配置读取并加载冻结的 Navigation Projector

|
|

`h_jepa/config/eval/ant_l3_hide.yaml`

|

新增

|

完全独立的实验配置

|
|

`h_jepa/tests/test_hide_*.py`

|

新增

|

No-op、形状、梯度、时间索引及跨层目标一致性测试

|
|

`h_jepa/audits/antmaze_hide_v1/`

|

新增

|

实验 Manifest、指标、失败案例和最终裁决

|

### 最重要的函数接入点

当前源码的 `LevelCostModel.get_cost()` 流程为：

Python

运行

```
info_dict = self.rollout(info_dict, action_candidates)
pred_emb = info_dict["predicted_embed_0"]

pred_for_cost, action_for_cost = self._project_to_goal_space(...)

cost_info = dict(info_dict)
cost_info["predicted_embed_0"] = pred_for_cost

cost = self.model.criterion(cost_info)
```

建议把新导航目标计算放在 `pred_for_cost` 已经确定之后，使用同一时间位置的预测表征和目标表征。

导航代价的核心计算可以参考：

Python

运行

```
def navigation_cost(pred_for_cost, goal_embed, nav_projector):
    # AntMaze V1: cost_last_n = 1
    # pred_for_cost: [B, N, T, D]
    # goal_embed:    [B, G, D]

    pred_nav = nav_projector(pred_for_cost[..., -1, :])
    goal_nav = nav_projector(goal_embed[:, -1, :]).detach()

    # [B, N]
    return ((pred_nav - goal_nav[:, None, :]) ** 2).sum(dim=-1)
```

这段是算法核心参考代码，不是已经验证过的完整源码补丁。实际实现要核实 `GradientSolver` 的候选维度、`cost_last_n`、L2 向 L3 投影时的时间对齐，以及 Goal Anchoring。

特别要保证：

* `enabled=false` 时原始 `criterion()` 的代码路径和返回值完全不变。

* `enabled=true` 时冻结 Navigation Projector 参数，但不能把整个规划 Cost 放在 `torch.no_grad()` 内，否则梯度无法回传至候选动作。

* L3 的 Horizon 为 1、被原始调度跳过时，不得强行生成导航子目标。

* L2 只有在本次确实跟踪 L3 预测目标时才使用导航跟踪项；否则沿用原始 Cost。

* 目标空间投影必须与原始 `upper_anchored`、`goal_embed_0_in_upper_space` 逻辑一致，绝不单独重写时间索引。

这些约束来自你当前已实现的跨层目标传递逻辑。

![](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

GitHub+1

## 五、分阶段实施与验收标准

建议分成四个阶段。只允许一个预先固定的 V1 方案进入 24 Episode 闭环对照，不在开发集上搜索多个权重或架构。

1. Stage 0：训练并验证导航表示

   使用训练来源状态，通过与评估一致的环境包装链实时渲染，训练冻结 H-JEPA 上的二维导航投影器。

   输出：`hide_nav.pt`、`hide_nav_metrics.json`、Checkpoint/Normalizer SHA256 和数据 Episode 清单。

   通过条件： 真实状态位置估计可靠，训练/在线渲染域表现一致，预测 Latent 的导航方向具有实际预测意义。

   失败处理： 不继续修改 Planner；记录原因并停止该 V1。

2. Stage 1：实现 L3/L2 目标函数解耦

   增加可选的 `NavigationCostAdapter`，只改变目标代价，不改变原始 Solver 和 Encoder。

   固定 α3=α2=0.25\alpha_3=\alpha_2=0.25α3=α2=0.25，代价尺度只在训练数据上确定。

   必须完成以下工程测试：关闭时逐动作等价；开启时梯度有限且可回传；L2/L3 时域对齐；L3 被跳过时行为正确；不同环境之间不共享状态；不修改真实最终 Goal。

   失败处理： 先修接口或形状错误，不允许降低原始规划参数来绕开问题。

3. Stage 2：严格配对的 24 Episode Pilot

   使用原有独立 12 任务开发集和两个环境初始化种子。

   首先检查前两个任务的开启/关闭差异、目标代价数值范围、梯度范数和每次规划耗时；这是烟雾测试，不用于挑选成功任务。

   接着运行完整 24 Episode。保留原始每 10 步重规划、350 步预算、模型和 Solver 随机种子、Batch 分组及确定性 Settle 协议。

   唯一主要指标：原始任务闭环成功率。

4. Stage 3：独立泛化验证

   只有开发阶段满足预先设定的工程门槛，才在新的、未参与训练和开发的 AntMaze 起终点任务上运行验证。

   在此之前，不使用原官方 50 任务集做调参。

   如果独立验证也显示稳定收益，再考虑加入 HiDe 论文中的自适应子目标范围或更明确的环境布局先验。

### 闭环实验怎么比较？

|
组别

|

L3 Cost

|

L2 Cost

|

L1

|
| --- | --- | --- | --- |
|

B0 原始 H-JEPA

|

原始

|

原始

|

原始

|
|

H1 HiDe-HJEPA V1

|

原始 + 导航代价

|

原始 + 导航跟踪代价

|

原始

|

不建议第一轮同时加入十几个消融组。若 H1 取得收益，再开展只修改 L3、只修改 L2 的消融，以分析收益来源。

预先冻结的 Go / No-Go 决策

开发集净成功增量

# ≥ +2

原成功任务被破坏

# ≤ 1

总规划计算开销

# ≤ 1.5×

原始关闭模式

# 完全等价

这是研究投入的工程门槛，不代表 +2/24 已具有统计显著性。必须经过独立任务复验。

## 六、建议的配置和运行方式

建立 `config/eval/ant_l3_hide.yaml`，在原始 AntMaze L3 配置基础上添加以下字段：

YAML

```
hide:
  enabled: false

  # Frozen navigation representation
  projector_path: null
  projector_dim: 2

  # Navigation-cost integration
  alpha_l3: 0.25
  alpha_l2: 0.25
  apply_to_l2: true
  scale_stats_path: null

  # Keep original solver and execution schedule
  preserve_original_horizon: true
  preserve_original_execution: true

  # Diagnostics
  log_navigation_cost: true
  log_original_cost: true
  log_gradient_norm: true
```

关闭配置必须完全复现原始 H-JEPA。开启时才加载投影器，并验证投影器文件对应的世界模型及 Normalizer 哈希。

测试命令框架：

Bash

```
cd /root/rivermind-data/H-JEPA/h_jepa

export HJEPA_HOME=/root/rivermind-data/hjepa_home
export PYTHONPATH=.

CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"
TASKS=audits/antmaze_dev_set/dev_grid_d3_n12_seed20261113.pt

python eval.py --config-name ant_l3_hide \
  policy="$CKPT" seed=42 \
  load_eval_trajs_path="$TASKS" \
  eval.num_eval=12 eval.eval_budget=350 \
  +eval.env_seed=1000 \
  hide.enabled=true \
  hide.projector_path=artifacts/hide/nav_projector.pt \
  hide.scale_stats_path=artifacts/hide/cost_scales.json \
  output.dir=eval_hide_dev_env1000
```

第二次使用 `env_seed=2000` 和独立输出目录。

注意： 以上是目标命令接口，需要 Agent 实现新增配置后才能运行；实际 `batch_size`、`chunk_size` 和并行 Worker 设置必须从既有确定性 Baseline Manifest 原样继承，不可使用新默认值。

## 七、需要保留哪些诊断指标？

除成功率外，只要求额外记录最能解释失败的指标：

|
指标

|

用来判断什么

|
| --- | --- |
|

Navigation Projector 在真实/预测状态上的误差

|

导航表示是否可信

|
|

L3 原始 Cost 与导航 Cost 的相对变化

|

是否产生与原目标冲突的优化方向

|
|

L3→L2 导航目标差异

|

高层指令是否被局部规划有效跟踪

|
|

实际 XY 到任务目标距离变化

|

预测的导航进展是否真实发生

|
|

墙壁附近停滞、碰撞和反复改变方向

|

是否因为二维导航代价而忽视障碍

|
|

成功任务 Rescue / Destroy

|

新方法究竟修复了谁、破坏了谁

|
|

规划耗时与额外网络开销

|

是否值得付出计算成本

|

实际 `xy` 只允许评估日志离线分析，不得在执行规划时参与 Cost 或动作选择。

还有一个必须预防的风险：Navigation Projector 即使能准确解码真实 L3 状态，也可能被梯度规划器利用，在不真实的预测 Latent 区域得到虚假的导航进展。因此 Stage 0 的预测状态验证不能省略。

## 八、可直接交给编码 Agent 的实施指令

下面这段可以作为独立任务直接提交：

编辑

# HiDe-HJEPA V1：AntMaze 规划与控制功能解耦实验

**目标：** 在冻结 AntMaze L3 H-JEPA 世界模型的前提下，通过导航任务空间投影器及 L3/L2 目标函数解耦，提高真实闭环成功率。

**代码基础：** `research/ant-trace-audit-v1`，提交 `48ef1c2`。

**新分支：** `research/ant-hide-decouple-v1`。

## 实施任务

1. 创建干净实验分支，不合并 PAHRL、CLFB、RAGC 或 Feedback-N 修改。

2. 复核 `HierarchicalSolver.solve()`、`LevelCostModel.get_cost()`、`_project_to_goal_space()`、`_encode_all_levels()` 和现有 `ant_l3.yaml`，绘制 L3→L2→L1 的状态及目标 Tensor 形状。

3. 新增 `NavigationProjector`，输入冻结 L3 Latent，输出二维导航位置。用训练来源、在线同域重渲染状态训练。按 Episode 分割训练、验证和测试。

4. 验证导航投影器在真实状态、Goal 编码及模型预测状态上的一致性。若预测状态不可靠，停止，不进行规划器集成。

5. 在 `LevelCostModel.get_cost()` 中增加默认关闭的导航 Cost Adapter。L3 优化任务导航目标，L2 增加导航子目标跟踪项；L1 保持完全不变。导航 Cost 必须对候选动作保持可微。

6. 保留原始 H-JEPA Cost 作为正则项，使用预先固定的 `alpha_l3=0.25` 和 `alpha_l2=0.25`。成本尺度只能在训练数据上校准。

7. 不修改 H-JEPA Checkpoint、GradientSolver 候选数/迭代数、Horizon Schedule、10 步动作执行周期和 350 步预算。

8. 添加默认关闭、显式关闭、完整 Goal Anchoring、Horizon 为 1、Batch 隔离、梯度有限性、模型/Normalizer 哈希一致性等测试。

9. 确认关闭模式下与原始 Baseline 逐动作一致，测试报告及源码提交通过后，才执行完整开发集闭环 Pilot。

10. 使用已有 12 任务 × 2 环境种子，严格配对原始 17/24 基线，比较 H1 与 B0。

11. 记录成功任务 Rescue、Destroy、净增量、真实环境位移、导航 Cost、停滞及墙体碰撞、实际计算时间。

12. 预设 GO 标准：NET ≥ +2，Destroy ≤ 1，额外计算 ≤ 1.5×。未达到则 NO-GO，不进行权重搜索。

13. 通过开发门槛后才进行新任务独立验证；不在开发期使用原官方 50 任务。

## 明确禁止

* 禁止删除或置零冻结模型原有的 Proprio 输入。

* 禁止用真实评估 XY、占据图或 BFS 作为规划输入。

* 禁止使用 PAHRL 的失配离线 Latent 缓存直接训练新 Head。

* 禁止在规划 Cost 中使用 `torch.no_grad()` 切断候选动作梯度。

* 禁止把局部 Cost 改善、导航解码准确度当作闭环成功率改善。

* 禁止未经验证就修改优化次数、执行频率或增加其他规划器。

## 交付

新分支 Commit、Projector Checkpoint、训练数据 Manifest、导航投影离线验证报告、No-op 与单元测试结果、24 Episode 配对成功表、计算开销与明确 Go/No-Go 结论。

任何关键门禁失败，立即提交实际证据并停止下一阶段；不要通过持续调参寻找开发集收益。

## 九、与 HiDe 原论文的关系及后续方向

这一版主要借鉴 HiDe 的功能性状态空间解耦、面向任务的目标空间和低层专注运动控制，而不是完整复制它的高层 MVProp 价值传播网络和联合训练 Actor-Critic。

这一区别很重要：原论文的高层通常能够访问占据栅格等环境先验，还使用动态 Agent-centric Attention 来限制子目标范围。你当前的 H-JEPA 则主要通过图像和本体感觉进行规划，不能默认拥有完整迷宫地图。

![](https://www.google.com/s2/favicons?domain=https://files.ait.ethz.ch\&sz=32)

AIT 文件库+1

如果 V1 在独立闭环实验中有效，V2 最值得研究的是：

让高层根据低层实际可执行能力，学习自适应的导航子目标范围。

这才更进一步接近 HiDe 的动态 Attention 思想，也能够针对“高层目标合理但低层难以执行”的情况开展研究。

但在 V1 成功之前，不要加入这个模块。

## 最终决策

我的建议是正式推进 HiDe-HJEPA V1，并以 Navigation Projector + L3/L2 Cost 解耦作为唯一主实验。

先证明在冻结世界模型下，明确区分导航目标与身体状态确实有收益；再讨论是否值得进一步学习子目标可达范围和地图感知。

这次不要把主要研究资源花在重新设计复杂 Planner 上。最关键的改动只有两个：高层优化什么，以及低层如何理解高层的目标。 如果这两个问题解决后闭环成功率仍然没有提高，就说明在当前 Checkpoint 上，功能解耦并不是主要瓶颈，应当及时停止。
