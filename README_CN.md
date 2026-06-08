# A-Simple-DSMoE（简易 DeepSeek-MoE 实现）

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.8.0+-green.svg)](https://www.mindspore.cn/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

基于 **[MindSpore](https://www.mindspore.cn/)** 框架实现的 **DeepSeek-MoE** 模型核心原理的**简化版本**。本项目**专为学习和学术研究设计**，不适用于工业部署。

---

## 目录

- [DeepSeek-MoE 是什么？](#deepseek-moe-是什么)
- [MindSpore 是什么？](#mindspore-是什么)
- [项目目标](#项目目标)
- [文件结构](#文件结构)
- [安装步骤](#安装步骤)
  - [CPU](#cpu)
  - [GPU](#gpu)
  - [Ascend](#ascend)
- [快速开始](#快速开始)
- [许可证](#许可证)

---

## DeepSeek-MoE 是什么？

**DeepSeek-MoE** 是 DeepSeek-AI 在论文 *"DeepSeek-MoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"*（2024）中提出的混合专家模型架构，相比传统 MoE 架构有多项关键创新：

### 核心创新

| 创新点 | 说明 |
|---|---|
| **细粒度专家分割** | 将标准 FFN 专家拆分为更细粒度的子单元，在保持总参数量不变的同时增加专家数量，提升路由灵活性。 |
| **共享专家隔离** | 指定部分专家为"共享专家"——对每个输入 token 都激活，负责捕获通用知识；其余"路由专家"负责学习特定模式。 |
| **动态路由** | 可学习的门控网络为每个输入 token 动态选择最相关的 Top-K 个路由专家，结合共享专家的输出产生最终结果。 |
| **负载均衡损失** | 辅助损失函数，促使 token 在各路由专家间均匀分布，防止专家坍塌，提高训练稳定性。 |

### 为什么选择 DeepSeek-MoE？

- **专家专业化**：细粒度专家使得模型可以更精确地专业化，提升模型质量。
- **计算高效**：每个 token 只激活少量专家，以远低于稠密模型的计算成本实现相近的模型容量。
- **可扩展性**：可自然扩展到数千亿参数（如 DeepSeek-V2、DeepSeek-V3）。

### 本项目实现

本项目提供一个简洁、极简的实现，捕捉 DeepSeek-MoE 的核心思想：

- **`dsmoe/model/expert.py`** — MoE 子模块，包含 `ExpertLayer`、`MoELayer`（含共享专家和路由专家隔离）和 `MoELoss`（负载均衡损失）
- **`dsmoe/model/ds_moe.py`** — 核心模型，包含 `RMSNorm`、`MultiHeadLatentAttention`（MLA）、`TransformerBlock` 和 `DSMoE`（完整 DeepSeek-MoE 模型）
- **`examples/`** — 可运行的示例脚本（预留）
- **`docs/`** — 文档（预留）

---

## MindSpore 是什么？

**[MindSpore](https://www.mindspore.cn/)** 是 **华为** 开发的开源深度学习框架，为 **端、边、云** 全场景 AI 应用提供统一的训练和推理能力。

### 核心特性

- **自动微分**：基于计算图的源码级自动微分。
- **动静态图统一**：支持 PyNative（动态图）和 Graph（静态图）两种模式。
- **Ascend 原生支持**：对华为昇腾 NPU 提供一流支持，同时兼容 GPU 和 CPU。
- **大模型支持**：内置数据并行、模型并行、流水线并行策略。
- **丰富生态**：MindSpore Lite、MindSpore Serving、MindInsight 等工具。

MindSpore 的 `nn.Cell` 抽象非常适合实现模块化的 MoE 架构和高效稀疏计算。

---

## 项目目标

1. **教育性**：为学生和 AI 爱好者提供一个易懂的 DeepSeek-MoE 实现。
2. **简洁清晰**：聚焦核心思想，不引入工业级复杂度和优化负担。
3. **跨平台**：支持 CPU、GPU 和 Ascend NPU 三种后端。
4. **可扩展**：模块化设计，方便扩展新的专家类型、路由策略或微调方法。

---

## 文件结构

```
A-Simple-DSMoE/
├── .github/                  # GitHub 配置（预留）
├── .gitignore                # Git 忽略规则
├── LICENSE                   # Apache 2.0 开源许可证
├── README.md                 # 项目说明（英文）
├── README_CN.md              # 项目说明（中文）
│
├── dsmoe/                    # 核心源码包
│   ├── __init__.py           # 包导出声明
│   └── model/                # 模型子包
│       ├── __init__.py       # 模型导出声明
│       ├── expert.py         # ExpertLayer、MoELayer、MoELoss
│       └── ds_moe.py         # RMSNorm、MultiHeadLatentAttention、TransformerBlock、DSMoE
│
├── docs/                     # 文档（预留）
├── examples/                 # 示例脚本（预留）
└── scripts/                  # 训练/推理脚本（预留）
```

---

## 安装步骤

### 环境要求

- Python 3.9 或更高版本
- pip（Python 包管理器）

### CPU

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore
pip install jieba matplotlib numpy
```

### GPU（CUDA）

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore-cuda11
pip install jieba matplotlib numpy
```

> 其他 CUDA 版本请参考 [MindSpore 安装指南](https://www.mindspore.cn/install)。

### Ascend（NPU）

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore-ascend
pip install jieba matplotlib numpy
```

> Ascend 版本安装前需先安装 CANN 工具包和驱动固件，详见[官方指南](https://www.mindspore.cn/install)。

### 验证安装

```python
import mindspore
print(mindspore.__version__)
print(mindspore.get_context("device_target"))
```

---

## 快速开始

```python
import mindspore as ms
from mindspore import Tensor
from dsmoe.model.ds_moe import DSMoE

# 创建 DeepSeek-MoE 模型
model = DSMoE(
    vocab_size=10000,
    d_model=2048,
    nhead=16,
    num_layers=6,
    num_experts=16,
    n_shared_experts=2,
    top_k=6,
    dim_feedforward=2048
)

# 前向传播
input_ids = Tensor([[1, 2, 3, 4, 5]], ms.int32)
output, aux_loss = model(input_ids)
print(f"输出形状: {output.shape}")
print(f"辅助损失: {aux_loss}")
```

可运行的示例脚本将在 `examples/` 目录中提供（即将推出）。

---

## 许可证

本项目基于 **Apache License 2.0** 开源。详见 [LICENSE](LICENSE) 文件。

---

## 参考资料

- [DeepSeek-MoE: Towards Ultimate Expert Specialization (arXiv, 2024)](https://arxiv.org/abs/2401.06066)
- [MindSpore 官方网站](https://www.mindspore.cn/)
- [MindSpore 安装指南](https://www.mindspore.cn/install)
