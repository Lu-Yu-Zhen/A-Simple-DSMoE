# A-Simple-DSMoE

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.8.0+-green.svg)](https://www.mindspore.cn/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

A **simplified implementation** illustrating the core principles of the **DeepSeek-MoE** model, built on the **[MindSpore](https://www.mindspore.cn/)** framework. This project is designed **exclusively for learning and academic research** — not for industrial deployment.

---

## Table of Contents

- [What is DeepSeek-MoE?](#what-is-deepseek-moe)
- [What is MindSpore?](#what-is-mindspore)
- [Project Goals](#project-goals)
- [File Structure](#file-structure)
- [Installation](#installation)
  - [CPU](#cpu)
  - [GPU](#gpu)
  - [Ascend](#ascend)
- [Quick Start](#quick-start)
- [License](#license)

---

## What is DeepSeek-MoE?

**DeepSeek-MoE** is a Mixture-of-Experts architecture proposed in the paper *"DeepSeek-MoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"* (2024) by DeepSeek-AI. It introduces several key innovations over traditional MoE architectures:

### Key Innovations

| Innovation | Description |
|---|---|
| **Fine-Grained Expert Segmentation** | Splits standard FFN experts into smaller, more fine-grained units, keeping the total parameter count the same but increasing the number of experts (and thus the routing flexibility). |
| **Shared Expert Isolation** | Designates certain experts as "shared" — activated for every input token to capture common knowledge, while the remaining "routed" experts specialize in specific patterns. |
| **Dynamic Routing** | A learned gating network dynamically selects the top-K most relevant routed experts for each input token, combined with the shared experts to produce the final output. |
| **Load-Balancing Loss** | An auxiliary loss that encourages uniform token distribution across routed experts, preventing expert collapse and improving training stability. |

### Why DeepSeek-MoE?

- **Expert Specialization**: Fine-grained experts allow more precise specialization, leading to better model quality.
- **Computation Efficiency**: By activating only a small fraction of experts per token, the model achieves the capacity of a much larger dense model at a fraction of the inference cost.
- **Scalability**: The architecture scales naturally to hundreds of billions of parameters (e.g., DeepSeek-V2, DeepSeek-V3).

### This Implementation

This project provides a clean, minimal implementation that captures the essential ideas of DeepSeek-MoE:

- **`dsmoe/model/expert.py`** — MoE submodule containing `ExpertLayer`, `MoELayer` (with shared & routed expert isolation), and `MoELoss` (load-balancing loss)
- **`dsmoe/model/ds_moe.py`** — Core model with `RMSNorm`, `MultiHeadLatentAttention` (MLA), `TransformerBlock`, and `DSMoE` (full DeepSeek-MoE model)
- **`examples/`** — Runnable demo scripts (reserved)
- **`docs/`** — Documentation (reserved)

---

## What is MindSpore?

**[MindSpore](https://www.mindspore.cn/)** is an open-source deep learning framework developed by **Huawei**, designed for all-scenario AI applications across **device, edge, and cloud**.

### Key Features

- **Automatic Differentiation**: Source-to-source auto-diff based on computational graphs.
- **Dynamic & Static Graphs**: Supports both PyNative (eager) and Graph (static) modes.
- **Ascend Native**: First-class support for Huawei Ascend NPUs, plus GPU and CPU backends.
- **Large Model Support**: Built-in data/model/pipeline parallelism for distributed training.
- **Rich Ecosystem**: MindSpore Lite, MindSpore Serving, MindInsight, etc.

MindSpore's `nn.Cell` abstraction makes it straightforward to implement modular MoE architectures with efficient sparse computation.

---

## Project Goals

1. **Educational**: Provide an easy-to-understand implementation of DeepSeek-MoE for students and AI enthusiasts.
2. **Minimal & Clean**: Focus on the core ideas without production-level complexity or optimization overhead.
3. **Cross-Platform**: Support CPU, GPU, and Ascend NPU backends.
4. **Extensible**: Modular design that can be extended with new expert types, routing strategies, or fine-tuning methods.

---

## File Structure

```
A-Simple-DSMoE/
├── .github/                  # GitHub configuration (reserved)
├── .gitignore                # Git ignore rules
├── LICENSE                   # Apache 2.0 License
├── README.md                 # Project documentation (English)
├── README_CN.md              # Project documentation (Chinese)
│
├── dsmoe/                    # Core package
│   ├── __init__.py           # Package exports
│   └── model/                # Model subpackage
│       ├── __init__.py       # Model exports
│       ├── expert.py         # ExpertLayer, MoELayer, MoELoss
│       └── ds_moe.py         # RMSNorm, MultiHeadLatentAttention, TransformerBlock, DSMoE
│
├── docs/                     # Documentation 
├── examples/                 # Example scripts 
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### CPU

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore
pip install jieba matplotlib numpy
```

### GPU (CUDA)

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore-cuda11
pip install jieba matplotlib numpy
```

> For other CUDA versions, see [MindSpore Installation Guide](https://www.mindspore.cn/install).

### Ascend (NPU)

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore-ascend
pip install jieba matplotlib numpy
```

> For Ascend, install the CANN toolkit and driver firmware first. See the [official guide](https://www.mindspore.cn/install).

### Verify Installation

```python
import mindspore
print(mindspore.__version__)
print(mindspore.get_context("device_target"))
```

---

## Quick Start

```python
import mindspore as ms
from mindspore import Tensor
from dsmoe.model.ds_moe import DSMoE

# Create a DeepSeek-MoE model
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

# Forward pass
input_ids = Tensor([[1, 2, 3, 4, 5]], ms.int32)
output, aux_loss = model(input_ids)
print(f"Output shape: {output.shape}")
print(f"Auxiliary loss: {aux_loss}")
```

See the `examples/` directory for runnable demos (coming soon).

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## References

- [DeepSeek-MoE: Towards Ultimate Expert Specialization (arXiv, 2024)](https://arxiv.org/abs/2401.06066)
- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindSpore Installation Guide](https://www.mindspore.cn/install)
