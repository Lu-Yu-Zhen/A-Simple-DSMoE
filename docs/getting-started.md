# Getting Started / 快速入门

> Bilingual documentation — English sections are followed by Chinese translations.

---

## 1. Installation / 安装

### Prerequisites / 环境要求

- Python 3.9+
- MindSpore 2.8.0+
- pip

See the main [README.md](../README.md) for detailed installation instructions across CPU, GPU, and Ascend backends.

### Quick Install / 快速安装

```bash
git clone https://github.com/Lu-Yu-Zhen/A-Simple-DSMoE.git
cd A-Simple-DSMoE
pip install mindspore
pip install jieba matplotlib numpy   # optional: data processing & visualization
```

---

## 2. Module Import / 模块导入

> **IMPORTANT**: Before running the code, you need to fix a relative import in `dsmoe/model/ds_moe.py`:
>
> Change `from model.expert import MoELayer` → `from .expert import MoELayer`
>
> This ensures the import works correctly within the `dsmoe.model` package.

> **重要提示**：运行前需修复 `dsmoe/model/ds_moe.py` 中的相对导入：
>
> 将 `from model.expert import MoELayer` 改为 `from .expert import MoELayer`
>
> 确保在 `dsmoe.model` 包内的导入正确工作。

---

## 3. Basic Usage / 基本使用

### 3.1 Create a Single MoELayer / 创建单个 MoELayer

```python
import mindspore as ms
from mindspore import Tensor

# After fixing the import, you can import directly:
from dsmoe.model.expert import MoELayer

moelayer = MoELayer(
    dim_model=1024,
    shared_ffn_dim=1024,
    router_ffn_dim=512,
    num_experts=16,
    top_k=2,
    dropout=0.1
)

x = Tensor.randn(2, 16, 1024, dtype=ms.float32)  # (B, L, D)
output, aux_loss = moelayer(x)

print(f"MoELayer output shape: {output.shape}")   # (2, 16, 1024)
print(f"Auxiliary loss: {aux_loss}")
```

### 3.2 Create a TransformerBlock / 创建单个 TransformerBlock

```python
from dsmoe.model.ds_moe import TransformerBlock

block = TransformerBlock(
    dim_model=1024,
    n_heads=16,
    shared_ffn_dim=1024,
    router_ffn_dim=512,
    num_experts=16,
    top_k=2,
    dropout=0.1
)

x = Tensor.randn(2, 16, 1024, dtype=ms.float32)
output, aux_loss = block(x)

print(f"TransformerBlock output shape: {output.shape}")  # (2, 16, 1024)
print(f"Auxiliary loss: {aux_loss}")
```

### 3.3 Build the Full DSMoE Model / 构建完整 DSMoE 模型

```python
from dsmoe.model.ds_moe import TransformerBlock, DSMoE

# Configure a single block template
block_template = TransformerBlock(
    dim_model=1024,
    n_heads=16,
    shared_ffn_dim=1024,
    router_ffn_dim=512,
    num_experts=16,
    top_k=2,
    dropout=0.1
)

# Build the full model with 6 stacked blocks
model = DSMoE(
    transformer_block=block_template,
    num_blocks=6
)

print(f"Total parameters: {model.total_params()}")
print(f"Number of blocks: {len(model)}")

# Forward pass
x = Tensor.randn(2, 16, 1024, dtype=ms.float32)
output, total_aux_loss = model(x)

print(f"DSMoE output shape: {output.shape}")        # (2, 16, 1024)
print(f"Total auxiliary loss: {total_aux_loss}")
```

### 3.4 Use the Combined Loss / 使用组合损失

```python
from dsmoe.model.expert import MoELoss

# Assume logits, labels, and aux_loss are from your forward pass
logits = Tensor.randn(2, 16, 10000, dtype=ms.float32)  # (B, L, vocab_size)
labels = Tensor.randint(0, 10000, (2, 16), dtype=ms.int32)

# Get aux_loss from the model
_, aux_loss = model(x)

# Compute combined loss
loss_fn = MoELoss(aux_loss=aux_loss, alpha=0.1)
loss = loss_fn(logits, labels)

print(f"Total loss: {loss}")
```

---

## 4. Running Scripts / 运行脚本

Navigate to the project root and ensure the import fix is applied, then:

```bash
# Example: run a quick forward test
python -c "
import mindspore as ms
from mindspore import Tensor
from dsmoe.model.ds_moe import TransformerBlock, DSMoE

# Fix import by running from the project root
block = TransformerBlock(dim_model=512, n_heads=8, num_experts=8, top_k=2)
model = DSMoE(block, num_blocks=3)
x = Tensor.randn(1, 8, 512, ms.float32)
out, loss = model(x)
print('Output shape:', out.shape, 'Aux loss:', loss)
"
```

> **Note**: The model currently operates on continuous hidden states (not token IDs). To use token IDs, you need to add a `nn.Embedding` layer as input and a `nn.Dense` as output projection.

> **注意**：当前模型处理的是连续隐藏状态（而非 token ID）。如需使用 token ID，需要添加 `nn.Embedding` 层作为输入和 `nn.Dense` 作为输出投影。

---

## 5. Visualizing the Model / 可视化模型

```python
# Print all trainable parameters
model.print_trainable_params()

# Display all TransformerBlock instances
model.show_cells()

# Get total parameter count
print(f"Total params: {model.total_params()}")
```

---

## 6. Project Structure Reference / 项目结构参考

```
A-Simple-DSMoE/
├── dsmoe/
│   ├── __init__.py
│   └── model/
│       ├── __init__.py
│       ├── expert.py         # ExpertLayer, MoELayer, MoELoss
│       └── ds_moe.py         # RMSNorm, MultiHeadLatentAttention,
│                             # TransformerBlock, DSMoE
├── docs/
│   ├── API.md                # API reference
│   ├── architecture.md       # Architecture overview
│   └── getting-started.md    # This guide
├── examples/                 # Example scripts (reserved)
└── scripts/                  # Training scripts (reserved)
```

---

## 7. FAQ / 常见问题

**Q: Why does the model output have the same shape as input?**

A: `DSMoE` operates on continuous hidden states. It does not include an embedding layer or output projection. The input should be `(B, L, dim_model)` — pre-computed hidden states or embeddings.

**Q: 为什么模型输出和输入形状相同？**

A：`DSMoE` 处理的是连续隐藏状态，不包含 Embedding 层和输出投影。输入应为 `(B, L, dim_model)` 的预计算隐藏状态或嵌入向量。

---

**Q: How to use the model with token IDs?**

A: Wrap the model with an embedding layer and output projection:

```python
class TextDSMoE(nn.Cell):
    def __init__(self, vocab_size, dim_model, ...):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim_model)
        self.dsmoe = DSMoE(...)
        self.output = nn.Dense(dim_model, vocab_size)

    def construct(self, input_ids):
        x = self.embed(input_ids)
        x, aux_loss = self.dsmoe(x)
        logits = self.output(x)
        return logits, aux_loss
```

**Q: 如何用 token ID 使用模型？**

A：用 Embedding 层和输出投影包装模型即可（见上方代码）。
