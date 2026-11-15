PAHRL → H-JEPA：具体实施方案

我建议创建独立分支 research/ant-pahrl-graph-v1，以已验证的 research/ant-trace-audit-v1、提交 48ef1c2 为基础。

第一版暂称 RG-HJEPA（Reachability-Graph-Guided H-JEPA）。

它不是复现 PAHRL 的完整强化学习系统，而是提取论文中最适合你的核心机制：

真实状态记忆 → 可信局部可达关系 → 全局图搜索 → 冻结 H-JEPA 分层执行。

PAHRL 原文依赖层级 Q 函数估计时间距离，并用 Dijkstra 生成路标；我们先用真实训练轨迹中已经发生的状态转换作为图边证据，暂不训练 Q 网络。

一、实施分为四个关口
阶段	核心任务	是否需要完整闭环评估
Stage 0	从 AntMaze 训练集提取冻结 L3 Latent，建立真实轨迹状态图	不需要
Stage 1	验证图连通性、跨墙错误连接和路标的局部可达性	少量局部测试
Stage 2	在原始 L3 目标位置接入图路标，保留全部原始 L2/L1 规划	先做无侵入性门禁
Stage 3	使用 12 任务 × 2 个 Settle Seed 做配对闭环 Pilot	24 个 Episode

三个原则保持不变：不修改 H-JEPA Checkpoint，不改变 10 步执行周期，不使用在线真实 XY 或迷宫 BFS 给规划器提供特权信息。

尤其不能跳过 Stage 1。单凭两个视觉 Latent 相似就连接图边，可能制造 PAHRL 论文已经指出的“隔墙捷径”。这是整套方案最容易失效的地方。

二、完整参考代码包

PAHRL-inspired H-JEPA V1

13 个文件，包含建图、离线路标、H-JEPA 接入补丁、测试及 Agent 任务书

下载 PAHRL-HJEPA V1 参考代码 ZIP

CPU 单元测试：14/14 通过

核心代码结构：

pahrl_hjepa_reference/
├── README.md
├── AGENT_TASK.md
├── apply_pahrl.py
├── patch_spec.json
└── h_jepa/
    ├── pahrl_memory.py
    ├── pahrl_waypoints.py
    ├── scripts/
    │   ├── export_pahrl_ant_states.py
    │   ├── build_pahrl_graph.py
    │   ├── propose_pahrl_links.py
    │   ├── calibrate_pahrl_attachments.py
    │   └── audit_pahrl_graph.py
    └── tests/
        ├── test_pahrl_memory.py
        └── test_pahrl_calibration.py
三、具体怎么执行

第一步：创建研究分支并安装代码。

cd /root/rivermind-data/H-JEPA

git switch research/ant-trace-audit-v1
git switch -c research/ant-pahrl-graph-v1

将 ZIP 解压到 /tmp/pahrl_hjepa_reference，然后：

SRC=/tmp/pahrl_hjepa_reference

cp "$SRC/h_jepa/"pahrl_*.py h_jepa/
cp "$SRC/h_jepa/scripts/"*.py h_jepa/scripts/
cp "$SRC/h_jepa/tests/"*.py h_jepa/tests/

python "$SRC/apply_pahrl.py" "$PWD"

cd h_jepa
PYTHONPATH=. python -m pytest \
  tests/test_pahrl_memory.py \
  tests/test_pahrl_calibration.py -q

安装脚本只修改 hierarchical_solver.py 和 planning_eval.py，默认不启用图规划。若源码与目标版本不匹配，会拒绝自动修改。

第二步：提取训练集的 L3 状态表示。

export HJEPA_HOME=/root/rivermind-data/hjepa_home
export PYTHONPATH=.

CKPT="$HJEPA_HOME/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt"

python scripts/export_pahrl_ant_states.py \
  --checkpoint "$CKPT" \
  --dataset "$HJEPA_HOME/visual_antmaze_medium_explore_stitch_train.h5" \
  --output artifacts/pahrl/states_train.npz \
  --every 10 \
  --max-episodes 300 \
  --batch-size 32

这个脚本复用仓库原有的图像归一化、Proprio 处理和逐层编码路径，而不是重新定义一套特征提取方式。源码中的 _encode_all_levels() 也正是原始规划器获取 L1/L2/L3 状态表示的位置。

第三步：构建第一版有向状态记忆图。

python scripts/build_pahrl_graph.py \
  --states artifacts/pahrl/states_train.npz \
  --output artifacts/pahrl/graph_observed_only.npz \
  --max-observed-gap 10

这里的边只代表训练轨迹中真实观察到的转换，不能因为两个节点 Latent 相近就自动增加捷径。

如果图不连通，包里还提供了跨轨迹候选连接的生成脚本；但这些候选必须经过冻结 H-JEPA 的局部真实执行验证，才能成为正式图边。

四、最重要的工程接入点

参考补丁把 Graph Waypoint 插入 HierarchicalSolver.solve() 中，位于 L3 Goal 编码完成后、逐层规划开始之前。

核心逻辑相当于：

graph_goal, graph_mask = controller.target_embeddings(
    obs_embeddings["embed_3"],
    goal_embeddings["embed_3"],
    steps_taken=steps_taken,
    call_idx=call_idx,
)

goal_embeddings["embed_3"] = torch.where(
    graph_mask[:, None, None],
    graph_goal,
    goal_embeddings["embed_3"],
)

这样原始 L3 规划器就会以图路标作为当前规划目标，随后仍走原来的 L2/L1 求解路径。

特别重要： 只有 L3 实际参与规划时才允许替换；L3 Horizon 降至 1、被原调度跳过时，必须继续使用真实任务 Goal。现有代码确实存在跳过高层后传递真实目标、以及将上层真实 Goal 锚定给下层的逻辑，不能随意绕过。

五、什么时候允许真正运行成功率实验？

必须先经过两个门禁。

门禁 A：图是否真的可用？

检查图连通性、训练数据覆盖、起终点附着、跨墙错误连接和局部路标到达率。若多数开发任务都无法获得可信路线，就停止，不强行上线。

门禁 B：接入是否完全正确？

Graph 禁用时必须逐动作复现 Baseline；没有有效路径时必须无损回退；真实 Checkpoint、Normalizer 与图文件必须匹配；不得改变 Solver RNG、Horizon Schedule 或原始执行周期。

参考包目前没有自动实现 AntMaze 完整的局部可达性仿真验证，因为这一步必须使用你服务器上已审计过的真实 World 包装链和确定性 Settle 协议。它需要 Agent 在 GPU 环境接入并验证，不能把 CPU 测试通过等同于闭环代码已经验证完成。

门禁均通过后，才做固定的 24 Episode 对照：

固定 AntMaze 开发集基线

17/24

70.8%

进入下一阶段的工程标准

净增加至少 2 个成功、破坏原成功任务不超过 1 个、规划计算开销不超过 1.5 倍。

两个 Settle Seed 共享同一批 12 个导航任务，统计上不能当成 24 个独立任务。若未达标，停止本方案，不进行参数搜索。

六、后续真正值得发展的创新点

第一版的图边来自真实训练轨迹和经过局部验证的连接，还不是 PAHRL 原文的“分层 Q 函数时间距离”。

如果 V1 能够提高闭环成功率，我认为最值得扩展的第二版才是：

用 H-JEPA 的多时间尺度预测学习策略条件下的时间可达性，以此改善图连边和路标选择。

这能将 PAHRL 的核心思想与 H-JEPA 本身的层级世界模型更深入地结合，而不是简单外挂一个 Dijkstra 搜索器。

不过，现在不应预先实现这一步。

我的最终建议：先交给 Agent 按代码包中的 AGENT_TASK.md 完成 Stage 0 建图和 Stage 1 局部可达性门禁。只有证明图确实覆盖可执行路线，才接入 L3 进行闭环成功率比较。

这既是对 PAHRL 主要思想比较忠实的借鉴，也符合我们之前确定的研究原则：每一个新机制，都必须最终通过真实闭环成功率来证明价值。