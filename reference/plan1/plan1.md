# Action-Response Network：基于你当前 H-JEPA 仓库的实施方案

这次严格限定在上次提出的 §2.1 反事实动作响应学习 和 §2.2 Push-T 中的动作作用差异，暂时不实现 Event Boundary、事件级世界模型或新的规划器。

## 先确定工程边界

我核对了公开仓库的以下代码：

* hjepa_utils.py ：模型构建与训练 Forward。

* models/hjepa.py ：层次编码接口。

* models/jepa.py ：单层 JEPA 编码、预测与 Rollout。

* base/pusht.yaml ：Push-T 数据字段及时间跨度。

* pusht_hjepa_l2.yaml ：双层模型配置。

由代码确认的基础约束是：

|
项目

|

当前仓库

|
| --- | --- |
|

Level-1 输入

|

RGB 图像，不使用 Proprio 作为编码输入

|
|

图像尺寸

|

224 × 224

|
|

Level-1 Latent

|

192 维

|
|

原始环境动作

|

2 维

|
|

Level-1 时间跨度

|

5 个环境步

|
|

Level-1 动作块

|

5 × 2 = 10 维

|
|

Level-2 Stride

|

2，即 10 个环境步

|
|

数据字段

|

包含 `pixels`、`action`、`state` 等

|
|

训练接口

|

`create_world_model(cfg)`、`model.get_level(1)`、`JEPA.encode(...)`

|

尤其要注意：虽然 HDF5 包含 `state`，它仍然只记录真实发生过的动作结果，不能直接作为多动作反事实数据集。 不能随意修改某条专家动作，然后继续使用原来的下一帧作为标签。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+2

我建议第一阶段采用 冻结原始 H-JEPA，仅训练独立的 Action-Response Network 的方案。这能最大程度保持现有训练、规划与评估结果不变，也方便之后验证新表示是否确实学到了额外信息。

## 一、§2.1：反事实动作响应学习（Action-Response Learning）

### 1. 简单思路

原始 H-JEPA 学习：

F(1)(zt,at)→z^t+1F^{(1)}(z_t,a_t)\rightarrow\hat z_{t+1}F(1)(zt,at)→z^t+1

即给定当前状态和动作，预测未来状态。

我们新增的网络学习另一件事：

从完全相同的物理状态出发，执行两组不同动作，未来会产生什么差异？

设两组动作序列为：

A=(a0,…,a4),B=(b0,…,b4)A=(a_0,\ldots,a_4),\qquad B=(b_0,\ldots,b_4)A=(a0,…,a4),B=(b0,…,b4)

都包含 5 个 Push-T 环境动作。

通过真实仿真获得：

oA=Sim⁡(st,A)o_A=\operatorname{Sim}(s_t,A)oA=Sim(st,A)

oB=Sim⁡(st,B)o_B=\operatorname{Sim}(s_t,B)oB=Sim(st,B)

使用冻结的 Level-1 Encoder 构造监督目标：

rA,B∗=sg⁡\(E(1)(oA)−E(1)(oB)\)\boxed{ r^*_{A,B} = \operatorname{sg} \left\(E^{(1)}(o\_A)-E^{(1)}(o\_B)\\right\) }rA,B∗=sg\(E(1)(oA​)−E(1)(oB​)\)

训练网络：

Rψ(zt,A,B)≈rA,B∗\boxed{ R_\psi(z_t,A,B)\approx r^*_{A,B} }Rψ(zt,A,B)≈rA,B∗

这样网络不是回答“执行 A 会到哪里”，而是回答“相对于执行 B，执行 A 会造成什么不同”。

这是一种在同一初始状态下，通过动作干预获得的反事实差异学习。它不等价于仅根据两条不相关轨迹计算未来状态差异。

### 2. 结合仓库的工程实施顺序

这里有一个很重要的决定：

暂时不要把新损失直接插入 `hjepa_forward()`。

第一版只增加独立训练流程。

代码模块及数据流

① Push-T Simulator

`stable_worldmodel/envs/pusht/env.py`

从相同 Seed 与相同历史动作恢复共同起点，分别执行 A、B，保存三张图像。

② Frozen Level-1 Encoder

`model.get_level(1).encode(...)`

提取 z0,zA,zBz_0,z_A,z_Bz0,zA,zB，并构造 rA,B∗=zA−zBr^*_{A,B}=z_A-z_BrA,B∗=zA−zB。

③ ActionResponseNet

输入 z0,A,Bz_0,A,Bz0,A,B，预测未来表示差异。

④ Independent Training

仅更新 ActionResponseNet，保留原始 H-JEPA 的全部参数。

我已经将这些模块组织成一套独立代码。

完整代码包

下载 H-JEPA Action-Response Starter 

包含 5 个 Python 文件与一份 AI Agent 实施文档。纯 PyTorch 单元测试已执行，4/4 通过；真实 Push-T 仿真及 Checkpoint 集成测试需要在你的 H-JEPA 环境中运行。

解压后，将对应文件合并到你仓库的 `h_jepa/` 目录。

建议首先让 AI Agent 执行 `git rev-parse HEAD` 记录所使用的 Commit，然后建立独立分支：

Bash

```
git switch -c research/action-response-v1
```

### 3. 详细代码与关键实现

#### 3.1 `h_jepa/action_response.py`

这是主要网络结构，完整代码已经放入/root/rivermind-data/H-JEPA/reference/plan1/hjepa_action_response_starter/h_jepa 。

我建议不要一开始直接设计任意的三输入 MLP，而采用一个具有反对称约束的结构：

Rψ(z,A,B)=ϕψ(z,A)−ϕψ(z,B)\boxed{ R_\psi(z,A,B)=\phi_\psi(z,A)-\phi_\psi(z,B) }Rψ(z,A,B)=ϕψ(z,A)−ϕψ(z,B)

这样具有两个天然性质：

Rψ(z,A,A)=0R_\psi(z,A,A)=0Rψ(z,A,A)=0

Rψ(z,A,B)=−Rψ(z,B,A)R_\psi(z,A,B)=-R_\psi(z,B,A)Rψ(z,A,B)=−Rψ(z,B,A)

含义是：

* 两个动作完全相同，动作差异必然为零。

* 交换动作 A、B，预测效应的方向应该反过来。

核心代码：

Python

运行

```
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResponseConfig:
    state_dim: int = 192
    action_dim: int = 2
    horizon: int = 5
    hidden_dim: int = 384


class ActionResponseNet(nn.Module):
    def __init__(self, cfg: ResponseConfig):
        super().__init__()
        self.cfg = cfg

        input_dim = (
            cfg.state_dim
            + cfg.horizon * cfg.action_dim
        )

        self.phi = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.state_dim),
        )

    def potential(self, z0, actions):
        # z0: [B, 192]
        # actions: [B, 5, 2]
        B = z0.shape[0]

        expected = (
            B, self.cfg.horizon, self.cfg.action_dim
        )
        assert tuple(actions.shape) == expected

        x = torch.cat(
            [z0, actions.reshape(B, -1)],
            dim=-1,
        )
        return self.phi(x)

    def forward(self, z0, acts_a, acts_b):
        return (
            self.potential(z0, acts_a)
            - self.potential(z0, acts_b)
        )
```

为什么没有直接接入 H-JEPA Predictor？

因为当前要验证的是一个独立的科学问题：

经过反事实监督，模型能否学习到真实环境中动作效应的差异？

如果一开始就让它依赖原来的 Predictor，再把它用于新规划器，后续失败很难归因。

第一阶段将网络独立出来，能更清楚地观察它学到了什么。

#### 3.2 生成真正的反事实数据

这是整个 §2.1 最容易犯错的地方。

你的 Push-T 仿真器提供 `_set_state(state)`，但源码中它主要恢复：

* Agent 位置。

* T 形块位置和角度。

* Agent 速度。

它没有完整恢复 T 形块的线速度、角速度及全部接触相关的物理状态，而且调用后还会执行一次 Pymunk 物理步。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

因此不要这样生成数据：

Python

运行

```
# 不推荐：可能没有恢复完整物理状态
env.unwrapped._set_state(state)
end_a = rollout(actions_a)

env.unwrapped._set_state(state)
end_b = rollout(actions_b)
```

这样两条轨迹未必真正从同一个完整物理状态出发。

我采用的方式是确定性历史重放：

同一个 Seed+同一段 Roll-in Actions→共同起点\text{同一个 Seed} + \text{同一段 Roll-in Actions} \rightarrow \text{共同起点}同一个 Seed+同一段 Roll-in Actions→共同起点

然后分叉执行 A、B。

对应的核心代码：

Python

运行

```
def run_branch(env, seed, prefix, branch_actions):
    # 从同一个初始随机种子恢复
    obs, _ = env.reset(seed=int(seed))

    # 完整重放相同的历史动作
    for a in prefix:
        obs, _, _, _, _ = env.step(a)

    # 到达共同的反事实起点
    start_state = obs["state"].copy()
    start_rgb = env.render().copy()

    contacts = 0

    # 仅从这一刻开始改变动作
    for a in branch_actions:
        obs, _, _, _, info = env.step(a)
        contacts += int(info.get("n_contacts", 0))

    return {
        "start_state": start_state,
        "start_rgb": start_rgb,
        "end_state": obs["state"].copy(),
        "end_rgb": env.render().copy(),
        "contacts": contacts,
    }


A = run_branch(env, seed, prefix, actions_a)
B = run_branch(env, seed, prefix, actions_b)

# 强制验证两个分支真正拥有相同起点
np.testing.assert_allclose(
    A["start_state"],
    B["start_state"],
    rtol=0,
    atol=1e-5,
)

assert np.array_equal(
    A["start_rgb"],
    B["start_rgb"],
)
```

其中 `prefix` 必须是预先保存的、数值完全一致的原始环境动作序列。

这比简单重置七维状态更加严格。

但即使这样，也必须通过上述断言验证当前 Pymunk 版本和环境实现确实能确定性重放；不能仅凭相同 Seed 就假定物理状态一定一致。

### 3.3 保存的数据格式

每个 Counterfactual Pair 建议保存：

|
字段

|

Shape

|

含义

|
| --- | --- | --- |
|

`rgb0`

|

`[224,224,3]`

|

两个动作分支的共同起点

|
|

`rgb_a`

|

`[224,224,3]`

|

执行 A 后的图像

|
|

`rgb_b`

|

`[224,224,3]`

|

执行 B 后的图像

|
|

`acts_a`

|

`[5,2]`

|

原始环境动作 A

|
|

`acts_b`

|

`[5,2]`

|

原始环境动作 B

|
|

`state0`

|

`[7]`

|

共同起点的物理状态

|
|

`state_a` / `state_b`

|

`[7]`

|

两种执行的最终物理状态

|
|

`contacts_a` / `contacts_b`

|

标量

|

仿真接触计数

|
|

`anchor_seed`

|

标量

|

用于重放与复现

|

其中物理状态与接触计数只用于数据检查和事后分析。

它们不应该进入第一版 Action-Response Network 的训练输入或监督目标。

这样可以保持未来跨任务迁移的可能性。

### 3.4 严格对齐原始 Encoder 的调用

在你的仓库中，Level 1 的状态编码不是简单调用：

Python

运行

```
model.encoder(image)
```

而是使用已有 JEPA 接口：

Python

运行

```
model.get_level(1).encode(...)
```

原始 `JEPA.encode` 接受带时间维的输入，并将结果存储在 `embed_0` 中。

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1

正确的编码示例：

Python

运行

```
@torch.no_grad()
def encode_rgb(model, images, device):
    # images: uint8 [B, 224, 224, 3]

    x = torch.from_numpy(images.copy())
    x = x.to(device=device, dtype=torch.float32)

    # NHWC -> NCHW
    x = x.permute(0, 3, 1, 2) / 255.0

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    # JEPA 要求输入具有时间维 T
    x = x.unsqueeze(1)  # [B, 1, 3, 224, 224]

    out = model.get_level(1).encode(
        {"pixels": x},
        key="pixels",
    )

    z = out["embed_0"][:, 0]

    assert z.shape == (len(images), 192)

    return z
```

ZIP 中的实际实现支持分批编码，避免一次将全部图像放入显存。

这里还有一个关键约束：

原始 H-JEPA Checkpoint 必须处于 `eval()` 模式，全部参数冻结。

否则可能一边训练 Action-Response Network，一边改变它的监督目标表示。

### 3.5 训练过程

训练时：

Python

运行

```
z0 = encode_rgb(model, rgb0, device)
z_a = encode_rgb(model, rgb_a, device)
z_b = encode_rgb(model, rgb_b, device)

target_response = (z_a - z_b).detach()

pred_response = response_net(
    z0,
    actions_a,
    actions_b,
)

loss = torch.nn.functional.mse_loss(
    pred_response,
    target_response,
)
```

注意这里的 `actions_a`、`actions_b` 是 `[B,5,2]` 的原始 Push-T 环境动作。

原始 H-JEPA 训练时使用的动作归一化和 Action Embedding，是另一条数据路径。

目前这个独立网络可以直接使用原始环境动作，因为 Push-T 的动作范围已受到限制。将来与原始 Predictor 做公平比较时，才需要严格复用它的 Action Normalizer。

### 4. 不需要长训的测试方法

建议按以下顺序运行。

第一步：运行纯网络单元测试

Bash

```
cd H-JEPA/h_jepa

PYTHONPATH=. python -m pytest -q \
  tests/test_action_response.py
```

已经实际测试通过：

4 passed

约 2.3 秒

测试覆盖相同动作输出零、交换动作时响应取反、错误 Shape 报错，以及小样本拟合后损失下降。

第二步：生成少量真实反事实样本

Bash

```
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_train.npz \
  --pairs 64 \
  --seed 42

python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_val.npz \
  --pairs 16 \
  --seed 1042
```

这一步会自动验证两个分支起点一致，并报告产生接触的样本数量。

第三步：只训练 20 个梯度步骤

Bash

```
python -u scripts/train_action_response.py \
  --train /tmp/pusht_cf_train.npz \
  --val /tmp/pusht_cf_val.npz \
  --ckpt "$HJEPA_HOME/ckpts/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt" \
  --out /tmp/action_response_head.pt \
  --steps 20 \
  --batch-size 8
```

这里的 20 Steps 只是验证：

* 真实 Checkpoint 能正确加载。

* Encoder 接口和数据维度正确。

* Loss 能正常反向传播。

* 网络参数确实被更新。

* 验证集指标能计算。

不能把 20 Steps 的结果作为模型学习能力的最终结论。

特别建议同时比较：

MSE⁡(Rψ,r∗)\operatorname{MSE}(R_\psi,r^*)MSE(Rψ,r∗)

和零响应基线：

MSE⁡(0,r∗)\operatorname{MSE}(0,r^*)MSE(0,r∗)

如果新网络只是预测接近零的向量，也可能在动作几乎没有改变物体的样本上得到很低的误差。

所以仅仅看到 Loss 下降还不够。

### 5. §2.1 必须提醒 AI Agent 的错误

下面是建议设置为必须检查的实现约束。

* 不能把不同初始状态的两条轨迹当成反事实配对。 反事实的关键是同一个完整物理起点。

* 不能修改动作后沿用原轨迹的未来图像。 这会生成错误监督标签。

* 不能直接用 `_set_state(state7)` 代替完整物理恢复。 当前环境的七维观测不足以完整描述所有隐藏动力学。

* 不能把 `[B,5,2]` 动作序列写成 `[B,2,5]` 或单个二维 Macro-action。

* 不能混用真实环境动作、经过 Normalizer 的动作和 JEPA 内部 Action Embedding。

* 不能让原始 H-JEPA 的 Encoder 在训练 Response Head 时继续更新。

* 不能把 `state`、`n_contacts` 或 T 块角度偷偷加入网络输入。 当前它们只允许用于采样与诊断。

* 不能仅凭一个训练集 Loss 下降就宣称学会动作因果效应。 需要独立的同状态反事实测试集。

# 二、§2.2：Push-T 中的动作作用机制差异

这一部分不是再训练一个新模型，而是验证：

§2.1 训练得到的 Action-Response Network，是否真的对 Push-T 中不同的动作作用情境敏感。

它直接关系到我们以后能否用 Action-Response Signature 发现事件边界。

## 1. 简单思路

我建议首先构造三种具有代表性的情境：

A. 自由移动

Agent 尚未接近方块

主要考察 Agent 自身位移差异

B. 接近方块

Agent 可能即将发生接触

考察小动作变化是否影响接触结果

C. 已有推动交互

Agent 与方块发生物理作用

考察方块平移及旋转的差异

但这里有一个科学上很重要的细节：

这三种情境只用于生成不同分布的样本，不应被直接当成模型已经发现的三个动力学类别。

例如，Collector 使用随机运动或朝方块方向的动作来构建起点，并不保证每个标记为 `push` 的样本都真正发生了接触。

必须根据真实仿真结果验证。

## 2. 最重要的测试：同样的动作，在不同状态下是否产生不同响应？

要验证这一点，最好固定两组候选动作 A 和 B，只改变初始状态。

例如：

A=\((0.6,0),…,(0.6,0)\)A= \((0.6,0),\\ldots,(0.6,0)\)A=\((0.6,0),…,(0.6,0)\)

B=\((0.6,0.35),…,(0.6,0.35)\)B= \((0.6,0.35),\\ldots,(0.6,0.35)\)B=\((0.6,0.35),…,(0.6,0.35)\)

两组动作各持续 5 个环境步。

然后在不同情境中分别执行它们。

![](data\:image/svg+xml;charset=utf-8,%3Csvg%20font-family%3D%22-apple-system-body%2C%20ui-sans-serif%2C%20-apple-system%2C%20system-ui%2C%20%26quot%3BSegoe%20UI%26quot%3B%2C%20Helvetica%2C%20%26quot%3BApple%20Color%20Emoji%26quot%3B%2C%20Arial%2C%20sans-serif%2C%20%26quot%3BSegoe%20UI%20Emoji%26quot%3B%2C%20%26quot%3BSegoe%20UI%20Symbol%26quot%3B%22%20font-weight%3D%22400%22%20data-d-component%3D%22svg%22%20fill%3D%22currentColor%22%20style%3D%22color%3Argb\(13%2C%2013%2C%2013\)%22%20viewBox%3D%220%200%20344%20144%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Ctext%20x%3D%227%22%20y%3D%2215%22%20font-size%3D%2211%22%20fill%3D%22currentColor%22%20font-weight%3D%22600%22%3ESame%20actions%20A%20and%20B%3C%2Ftext%3E%3Crect%20x%3D%225%22%20y%3D%2229%22%20width%3D%22105%22%20height%3D%2230%22%20rx%3D%225%22%20fill%3D%22%23e3edf0%22%2F%3E%3Ctext%20x%3D%2257%22%20y%3D%2248%22%20fill%3D%22%233b6575%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3EFree%20motion%3C%2Ftext%3E%3Cpath%20d%3D%22M115%2044%20H162%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%22165%22%20y%3D%2231%22%20width%3D%2273%22%20height%3D%2226%22%20rx%3D%225%22%20fill%3D%22%23f2e4d8%22%2F%3E%3Ctext%20x%3D%22201%22%20y%3D%2248%22%20fill%3D%22%2398613c%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3EA%20vs%20B%3C%2Ftext%3E%3Cpath%20d%3D%22M241%2044%20H260%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%225%22%20y%3D%2268%22%20width%3D%22105%22%20height%3D%2230%22%20rx%3D%225%22%20fill%3D%22%23e3edf0%22%2F%3E%3Ctext%20x%3D%2257%22%20y%3D%2287%22%20fill%3D%22%233b6575%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3ENear%20object%3C%2Ftext%3E%3Cpath%20d%3D%22M115%2083%20H162%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%22165%22%20y%3D%2270%22%20width%3D%2273%22%20height%3D%2226%22%20rx%3D%225%22%20fill%3D%22%23f2e4d8%22%2F%3E%3Ctext%20x%3D%22201%22%20y%3D%2287%22%20fill%3D%22%2398613c%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3EA%20vs%20B%3C%2Ftext%3E%3Cpath%20d%3D%22M241%2083%20H260%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%225%22%20y%3D%22107%22%20width%3D%22105%22%20height%3D%2230%22%20rx%3D%225%22%20fill%3D%22%23e3edf0%22%2F%3E%3Ctext%20x%3D%2257%22%20y%3D%22126%22%20fill%3D%22%233b6575%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3EInteraction%3C%2Ftext%3E%3Cpath%20d%3D%22M115%20122%20H162%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%22165%22%20y%3D%22109%22%20width%3D%2273%22%20height%3D%2226%22%20rx%3D%225%22%20fill%3D%22%23f2e4d8%22%2F%3E%3Ctext%20x%3D%22201%22%20y%3D%22126%22%20fill%3D%22%2398613c%22%20font-size%3D%2210%22%20text-anchor%3D%22middle%22%3EA%20vs%20B%3C%2Ftext%3E%3Cpath%20d%3D%22M241%20122%20H260%22%20stroke%3D%22%2384969f%22%20stroke-width%3D%221.4%22%2F%3E%3Crect%20x%3D%22263%22%20y%3D%2248%22%20width%3D%2278%22%20height%3D%2263%22%20rx%3D%226%22%20fill%3D%22%23e2eee8%22%2F%3E%3Ctext%20x%3D%22302%22%20y%3D%2269%22%20text-anchor%3D%22middle%22%20fill%3D%22%2339775d%22%20font-size%3D%2210%22%3EDifferent%3C%2Ftext%3E%3Ctext%20x%3D%22302%22%20y%3D%2282%22%20text-anchor%3D%22middle%22%20fill%3D%22%2339775d%22%20font-size%3D%2210%22%3Eresponse%3C%2Ftext%3E%3Ctext%20x%3D%22302%22%20y%3D%2295%22%20text-anchor%3D%22middle%22%20fill%3D%22%2339775d%22%20font-size%3D%2210%22%3Esignatures%3C%2Ftext%3E%3C%2Fsvg%3E)希望学到的是状态依赖的动作效应，而不仅是看到某个动作向右，就始终输出相同的响应向量。

核心假设：

Rψ(zfree,A,B)≠Rψ(zinteraction,A,B)R_\psi(z_{\mathrm{free}},A,B) \neq R_\psi(z_{\mathrm{interaction}},A,B)Rψ(zfree,A,B)=Rψ(zinteraction,A,B)

但注意，这个不等式成立本身还不足以证明学到了有意义的动力学机制。

因为输入状态不同，网络当然可能输出不同向量。

真正需要检验的是：这种差异能否预测真实物理执行中 Agent、物体平移和物体旋转的变化。

## 3. §2.2 的具体代码与工程测试

代码包中已经实现 `--fixed-probes` 选项。

执行：

Bash

```
python -u scripts/collect_action_response_pusht.py \
  --out /tmp/pusht_cf_fixed.npz \
  --pairs 24 \
  --seed 2042 \
  --fixed-probes
```

这样每个 Anchor 使用完全相同的两组候选动作。

然后运行：

Bash

```
python -u scripts/analyze_action_response_pusht.py \
  --data /tmp/pusht_cf_fixed.npz
```

该脚本根据真实仿真状态计算三类物理效应。

Agent 位移响应差异：

Δagent=∥paA−paB∥2\Delta_{\mathrm{agent}} = \|p_a^A-p_a^B\|_2Δagent=∥paA−paB∥2

物体平移响应差异：

Δblock=∥pbA−pbB∥2\Delta_{\mathrm{block}} = \|p_b^A-p_b^B\|_2Δblock=∥pbA−pbB∥2

物体旋转响应差异：

Δrot=∣wrap⁡(θbA−θbB)∣\Delta_{\mathrm{rot}} = |\operatorname{wrap}(\theta_b^A-\theta_b^B)|Δrot=∣wrap(θbA−θbB)∣

主要实现：

Python

运行

```
import numpy as np


def wrapped_angle_delta(a, b):
    return np.arctan2(
        np.sin(a - b),
        np.cos(a - b),
    )


# state_a, state_b: [N,7]
# 0:2 Agent xy
# 2:4 Block xy
# 4   Block orientation
# 5:7 Agent velocity

agent_effect = np.linalg.norm(
    state_a[:, :2] - state_b[:, :2],
    axis=1,
)

block_effect = np.linalg.norm(
    state_a[:, 2:4] - state_b[:, 2:4],
    axis=1,
)

rotation_effect = np.degrees(
    np.abs(
        wrapped_angle_delta(
            state_a[:, 4],
            state_b[:, 4],
        )
    )
)

contact_pair = (
    contacts_a + contacts_b
) > 0
```

这三个指标不是网络的训练目标，而是评价网络是否具有物理解释性的工具。

建议按 `free`、`approach`、`push` 三种采样情境统计平均值，并额外按真实是否发生接触分组。

### 推荐增加一个学习后检查

在完整代码包基础上，可以让 Agent 再写一个很小的分析脚本：

1. 加载训练好的 `ActionResponseNet`。

2. 使用冻结 Encoder 提取每个 Anchor 的 z0z_0z0。

3. 输入固定动作 A、B。

4. 计算 r^=Rψ(z0,A,B)\hat r=R_\psi(z_0,A,B)r^=Rψ(z0,A,B)。

5. 与真实目标 r∗=zA−zBr^*=z_A-z_Br∗=zA−zB 比较。

6. 按是否发生接触、物体是否明显移动、是否发生旋转分组报告误差。

其中最重要的指标是：

Eresponse=∥Rψ(z0,A,B)−(zA−zB)∥2\boxed{ E_{\mathrm{response}} = \|R_\psi(z_0,A,B)-(z_A-z_B)\|^2 }Eresponse=∥Rψ(z0,A,B)−(zA−zB)∥2

同时报告真实效应的模长：

∥zA−zB∥\|z_A-z_B\|∥zA−zB∥

因为若真实效应本来几乎为零，那么一个始终输出零的模型也可能获得很小误差。

## 4. §2.2 最好进行哪些快速检查？

|
检查

|

合格表现

|

不合格时重点排查

|
| --- | --- | --- |
|

固定动作正确性

|

所有样本中的 A 相同、B 相同

|

数据生成过程中动作被重新采样

|
|

共同起点

|

每对起点图像完全一致

|

Reset / Roll-in 重放错误

|
|

动作确实不同

|

A、B 至少一个维度不同

|

随机扰动或 Clip 导致动作相同

|
|

物理作用覆盖

|

有自由运动样本，也有实际接触样本

|

起点采样过于单一

|
|

旋转效应覆盖

|

至少部分样本有可观测的方向差异

|

接触位置或动作组合不足

|
|

响应方向

|

交换 A、B 后输出取反

|

网络结构实现错误

|
|

Response Prediction

|

独立验证误差优于零响应基线

|

表征、数据或模型学习不足

|

注意：不应强制要求每个 `push` 样本都发生旋转。 这会混淆采样意图与真实物理结果。

如果 24 个固定探针样本中接触和旋转覆盖不足，可以先改变采样起点和动作探针，而不是急着修改模型。

这仍属于快速数据诊断，不需要长时间训练。

## 5. §2.2 尤其需要防止的 AI Agent 错误

最重要的是下面三个。

错误一：把接触计数当成模型必须学习的类别标签。

当前阶段真正要学习的是未来状态响应差异，而不是一个 `contact / no-contact` 分类器。

如果直接训练接触类别分类器，就把研究问题换成了另一项任务。

错误二：用不同的动作测试不同情境，然后声称差异来自状态。

如果自由运动情境使用动作 A，接触情境使用动作 B，那么响应差异可能只是动作不同造成的。

要研究状态依赖性，必须固定 A、B，只改变起点。

错误三：只比较 Latent Response，不检查真实物理状态变化。

JEPA 可能主要编码 Agent 的视觉变化，而对方块旋转不够敏感。

因此，即使 Latent Prediction Error 很小，也不能直接认为模型捕获了重要的接触机制。

必须检查它是否能够区分具有不同物理后果的动作对。

# 三、给 AI Agent 的实施顺序与验收标准

我建议你让 Agent 分四个小 PR 或四次独立提交完成，而不是一次性写完所有代码。

Commit 1：Counterfactual Data Collector

只加入 Push-T 数据生成脚本。验证两分支共同起点一致、输出图片正确、物理状态差异合理。

此阶段禁止修改 JEPA、Planner 和 Policy。

Commit 2：ActionResponseNet

新增独立 PyTorch Module，验证相同动作响应为零、反对称性、梯度和张量维度。

此阶段不需要加载任何 H-JEPA Checkpoint。

Commit 3：Frozen Encoder Integration

加载现有 Push-T L2 Checkpoint，提取 Level-1 Latent，训练 Response Head。

先运行 1 Step，再运行 20 Steps。验证梯度与验证集指标。

Commit 4：Physical Response Diagnostics

固定两组 Action Probes，在不同初始状态下收集数据，分析真实物理效应与 Latent Response。

这一步产出以后用于研究 Event Boundary 的第一份机制证据。

这里我特别建议：每个 Commit 都运行现有的快速单元测试，并检查 `git diff`，确保没有无意间修改原始 H-JEPA 的训练与规划行为。

## 四、可直接交给 AI Agent 的任务说明

```
任务：在 bright-moon-67/H-JEPA 仓库中实现
Action-Response Network 的第一阶段（§2.1 与 §2.2）。

请先阅读随附 README_AGENT.md，并核对以下源文件：

h_jepa/models/hjepa.py
h_jepa/models/jepa.py
h_jepa/hjepa_utils.py
h_jepa/data.py
h_jepa/config/train/base/pusht.yaml
h_jepa/config/train/pusht_hjepa_l2.yaml
stable_worldmodel/envs/pusht/env.py

目标：

1. 从相同完整仿真状态出发，执行两组不同的
   5-step Push-T 动作，生成真正的 Counterfactual Pair。

2. 使用现有冻结的 Level-1 Encoder 得到
   z0、zA、zB，构造差异监督 target=zA-zB。

3. 实现独立的 ActionResponseNet：
   R(z,A,B)=phi(z,A)-phi(z,B)。

4. 仅训练 ActionResponseNet，不更新原始 H-JEPA。

5. 用相同 A/B 动作、不同初始状态检查响应的
   State Dependence，并使用物理状态做事后分析。

必须遵守：

- 不修改原有 Planner、Policy 或 hjepa_forward。
- 不复用原始专家轨迹的终点作为修改后动作的标签。
- 不把七维 Push-T state 当作完整 Pymunk 快照。
- 不混淆原始环境动作和归一化后的 JEPA 动作。
- 不把接触标签、物体姿态等加入模型输入。
- 不在没有测试的情况下声称代码可运行。
- 不启动大规模长训或完整 50-task 规划评估。

工作顺序：

A. 检查源仓库 Commit、路径、接口与张量 Shape。
B. 运行现有单元测试。
C. 生成 4 个 Counterfactual Pairs 做数据 Smoke Test。
D. 生成 64/16 个 Train/Val Pairs。
E. 用已有 Checkpoint 进行 1-step 和 20-step 训练测试。
F. 生成固定动作探针并分析物理状态差异。
G. 报告所有代码修改、测试输出与尚未验证的部分。

验收要求：

- 相同起点严格验证通过。
- R(z,A,A)=0。
- R(z,A,B)=-R(z,B,A)。
- Encoder 输出为 [B,192]。
- 所有 Backbone 参数保持冻结。
- Loss 为有限数且能够反向传播。
- 报告 Validation MSE 与 Zero-effect MSE。
- 不改变原有 H-JEPA Baseline 的执行逻辑。
```

## 五、现阶段应该如何判断这项工作做成功了？

我建议区分三个层次。

|
层次

|

验收目标

|

当前状态

|
| --- | --- | --- |
|

工程正确性

|

数据配对、网络约束、接口、梯度都正确

|

纯 PyTorch 4/4 测试已通过；仿真集成待跑

|
|

表征有效性

|

能预测未见过的动作干预差异，优于零响应及必要的模型基线

|

待实验

|
|

物理机制敏感性

|

能够区分自由移动、接触推动等情境下不同动作的物理后果

|

待实验

|

第一个层次只证明代码实现没有明显错误。

第二个层次证明网络确实学到了非平凡的动作响应。

第三个层次才真正支持我们之前提出的研究假设：动作的因果效应结构有可能成为比固定时间间隔更合适的世界模型抽象依据。

我建议你当前只要求 AI Agent 完成这三个层次的验证，不要提前把 Response Head 接入高层规划。

还有一个必须保留的科学基线：未来正式研究时，应比较新网络与原始冻结 World Model 对 A、B 分别预测后再相减的结果。否则不能证明反事实专项训练优于原模型已有的动作区分能力。这个比较需要按源仓库正确加载 Action Normalizer，因此我没有在本次 Starter 中用未经核对的近似实现替代它。

最终要验证的不是“又训练了一个能预测 Latent Difference 的 MLP”，而是“额外学习动作干预差异，是否能揭示原始 H-JEPA 在不同动力学情境下未能可靠利用的可控性信息”。

这是 §2.1 和 §2.2 应当完成的真正研究目标，也是后续事件分层机制能否成立的前提。
