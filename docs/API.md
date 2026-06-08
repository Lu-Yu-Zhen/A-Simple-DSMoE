# API Reference

> Bilingual documentation — English sections are followed by Chinese translations.

---

## Package Structure / 包结构

```
dsmoe/
├── __init__.py          # Package entry
├── model/               # Model subpackage
│   ├── __init__.py
│   ├── expert.py        # ExpertLayer, MoELayer, MoELoss
│   └── ds_moe.py        # RMSNorm, MultiHeadLatentAttention, TransformerBlock, DSMoE
```

---

## dsmoe.model.expert

### `ExpertLayer`

A single feed-forward expert network with two fully-connected layers and GELU activation.

单个前馈专家网络，包含两个全连接层和 GELU 激活函数。

```python
ExpertLayer(dim_model: int = 1024, ffn_dim: int = 1024)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `dim_model` | `int` | `1024` | Input & output dimension / 输入输出维度 |
| `ffn_dim` | `int` | `1024` | Hidden dimension of the FFN / FFN 隐藏层维度 |

**Forward / 前向传播:**

```
x → fc1 → GELU → fc2 → output
```

- **Input**: `(B, L, dim_model)` or `(N, dim_model)`
- **Output**: Same shape as input / 与输入形状相同

---

### `MoELayer`

Mixture-of-Experts layer with shared expert isolation and top-K routed expert gating. Includes an auxiliary load-balancing loss.

混合专家层，包含共享专家隔离和 Top-K 路由门控，附带辅助负载均衡损失。

```python
MoELayer(
    dim_model: int = 1024,
    shared_ffn_dim: int = 1024,
    router_ffn_dim: int = 512,
    num_experts: int = 16,
    top_k: int = 2,
    dropout: float = 0.1
)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `dim_model` | `int` | `1024` | Model hidden dimension / 模型隐藏层维度 |
| `shared_ffn_dim` | `int` | `1024` | Shared expert FFN dimension / 共享专家 FFN 维度 |
| `router_ffn_dim` | `int` | `512` | Routed expert FFN dimension / 路由专家 FFN 维度 |
| `num_experts` | `int` | `16` | Number of routed experts / 路由专家数量 |
| `top_k` | `int` | `2` | Number of experts activated per token / 每个 token 激活的专家数 |
| `dropout` | `float` | `0.1` | Dropout rate / Dropout 比例 |

**Forward / 前向传播:**

- **Input**: `(B, L, dim_model)`
- **Output**: `Tuple[Tensor, Tensor]` — `(moe_output: (B, L, dim_model), aux_loss: scalar)`

**Internal methods / 内部方法:**

| Method / 方法 | Description / 说明 |
|---|---|
| `_gate_forward(x_flat)` | Compute router gate probabilities / 计算路由门控概率 |
| `_top_k_selection(gate_prob)` | Select top-K experts / 选择 Top-K 专家 |
| `_get_mask(topk_indices, N)` | Generate one-hot dispatch mask / 生成 one-hot 分发掩码 |
| `_normalize_dispatch_weights(mask, gate_prob)` | Normalize dispatch weights / 归一化分发权重 |
| `_shared_expert(x_flat)` | Forward through shared expert / 共享专家前向 |
| `_expert_forward(x_flat, dispatch_weights, N, origin_shape)` | Weighted sum over all experts / 各专家输出的加权求和 |
| `_aux_loss_fn(mask, gate_prob)` | Load-balancing auxiliary loss / 负载均衡辅助损失 |

The auxiliary loss encourages uniform token distribution across experts:

辅助损失促使 token 在各专家间均匀分布：

```
aux_loss = sum(mean(mask) * mean(gate_prob)) * num_experts
```

---

### `MoELoss`

Combined loss: main cross-entropy loss + weighted auxiliary load-balancing loss.

组合损失：主交叉熵损失 + 加权的辅助负载均衡损失。

```python
MoELoss(aux_loss: Tensor, alpha: float = 0.1)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `aux_loss` | `Tensor` | — | Auxiliary loss from `MoELayer` / 来自 MoELayer 的辅助损失 |
| `alpha` | `float` | `0.1` | Weight of auxiliary loss / 辅助损失权重 |

**Forward / 前向传播:**

```
loss = CrossEntropyLoss(logits, labels) + alpha * aux_loss
```

- **Input**: `logits: (B, L, vocab_size)`, `labels: (B, L)`
- **Output**: Scalar loss / 标量损失值

---

## dsmoe.model.ds_moe

### `RMSNorm`

Root Mean Square Layer Normalization — a computationally cheaper alternative to LayerNorm.

RMS 层归一化 — 比 LayerNorm 计算更高效的归一化方法。

```python
RMSNorm(normalize_shape: Union[int, Tuple], eps: float = 1e-6)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `normalize_shape` | `int` or `Tuple` | — | Shape for normalization / 归一化形状 |
| `eps` | `float` | `1e-6` | Small constant for numerical stability / 防止除零的小常数 |

**Forward / 前向传播:**

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

---

### `MultiHeadLatentAttention`

Multi-head Latent Attention (MLA) — an efficient attention mechanism from DeepSeek-V2 that uses low-rank compression to reduce KV cache memory.

多头潜在注意力（MLA）— DeepSeek-V2 提出的高效注意力机制，通过低秩压缩降低 KV 缓存显存占用。

```python
MultiHeadLatentAttention(
    dim_model: int = 1024,
    n_heads: int = 16,
    q_lora_rank: int = 128,
    kv_lora_rank: int = 128,
    qh_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    rope_theta: float = 10000.0,
    max_position_embeddings: int = 32000
)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `dim_model` | `int` | `1024` | Model hidden dimension / 模型隐藏层维度 |
| `n_heads` | `int` | `16` | Number of attention heads / 注意力头数 |
| `q_lora_rank` | `int` | `128` | Query low-rank compression dimension / Query 低秩压缩维度 |
| `kv_lora_rank` | `int` | `128` | KV low-rank compression dimension / KV 低秩压缩维度 |
| `qh_nope_head_dim` | `int` | `128` | No-PE dimension per head / 每头无位置编码维度 |
| `qk_rope_head_dim` | `int` | `64` | RoPE dimension per head / 每头 RoPE 维度 |
| `v_head_dim` | `int` | `128` | Value dimension per head / 每头 Value 维度 |
| `rope_theta` | `float` | `10000.0` | RoPE base frequency / RoPE 基频 |
| `max_position_embeddings` | `int` | `32000` | Maximum sequence length / 最大序列长度 |

**Internal projection matrices / 内部投影矩阵:**

| Matrix / 矩阵 | Shape / 形状 | Purpose / 用途 |
|---|---|---|
| `W_dq` | `(dim_model, q_lora_rank)` | Query down-projection (LoRA-style) / Query 低秩下投影 |
| `W_uq` | `(q_lora_rank, n_heads × qh_nope_head_dim)` | Query up-projection / Query 升维上投影 |
| `W_qr` | `(dim_model, qk_rope_head_dim)` | Query RoPE projection / Query RoPE 投影 |
| `W_dkv` | `(dim_model, kv_lora_rank)` | Shared KV down-projection / KV 共享下投影 |
| `W_uk` | `(kv_lora_rank, n_heads × qh_nope_head_dim)` | Key up-projection / Key 升维上投影 |
| `W_uv` | `(kv_lora_rank, n_heads × v_head_dim)` | Value up-projection / Value 升维上投影 |
| `W_kr` | `(dim_model, qk_rope_head_dim)` | Key RoPE projection / Key RoPE 投影 |
| `W_o` | `(n_heads × qh_nope_head_dim, dim_model)` | Output projection / 输出投影 |

**Forward / 前向传播:**

- **Input**: `(B, L, dim_model)`
- **Output**: `(B, L, dim_model)`

**Internal methods / 内部方法:**

| Method / 方法 | Description / 说明 |
|---|---|
| `_get_position_ids(seq_len)` | Generate position indices / 生成位置索引 |
| `_apply_rotary_emb(x, position_ids)` | Apply rotary position embedding (RoPE) / 施加旋转位置编码 |

---

### `TransformerBlock`

A single Transformer block combining Multi-Head Latent Attention (MLA) and MoE Feed-Forward Network.

单个 Transformer 块，组合多头潜在注意力（MLA）和 MoE 前馈网络。

```python
TransformerBlock(
    dim_model: int = 1024,
    n_heads: int = 16,
    shared_ffn_dim: int = 1024,
    router_ffn_dim: int = 512,
    num_experts: int = 16,
    top_k: int = 2,
    q_lora_rank: int = 128,
    kv_lora_rank: int = 128,
    qh_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    rope_theta: float = 10000.0,
    max_position_embeddings: int = 32000,
    dropout: float = 0.1
)
```

(Parameter descriptions are shared with `MultiHeadLatentAttention` and `MoELayer`; refer to those sections for details.)

**Forward flow / 前向流程:**

```
x → RMSNorm → MLA → Dropout → + residual → RMSNorm → MoE FFN → Dropout → + residual → output
```

- **Input**: `(B, L, dim_model)`
- **Output**: `Tuple[Tensor, Tensor]` — `(output: (B, L, dim_model), aux_loss: scalar)`

**Internal methods / 内部方法:**

| Method / 方法 | Description / 说明 |
|---|---|
| `_mla_block(x)` | MLA sub-block with Dropout / MLA 注意力子块 |
| `_moe_ffn_block(x)` | MoE FFN sub-block with Dropout / MoE 前馈子块 |

---

### `DSMoE`

The full DeepSeek-MoE model: a stack of `TransformerBlock`s.

完整的 DeepSeek-MoE 模型：多层 `TransformerBlock` 堆叠。

```python
DSMoE(transformer_block: TransformerBlock, num_blocks: int = 12)
```

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| `transformer_block` | `TransformerBlock` | — | A configured TransformerBlock instance / 一个配置好的 TransformerBlock 实例 |
| `num_blocks` | `int` | `12` | Number of stacked blocks / 堆叠层数 |

**Forward / 前向传播:**

- **Input**: `(B, L, dim_model)` (hidden states, not token IDs / 隐藏状态，而非 token ID)
- **Output**: `Tuple[Tensor, Tensor]` — `(output: (B, L, dim_model), total_aux_loss: scalar)`

**Utility methods / 工具方法:**

| Method / 方法 | Description / 说明 |
|---|---|
| `print_trainable_params()` | Print all trainable parameters / 打印所有训练参数 |
| `show_cells()` | Display all TransformerBlock instances / 显示所有 TransformerBlock 实例 |
| `total_params()` | Return total number of trainable parameters / 返回可训练参数总数 |
| `__len__()` | Return number of layers / 返回层数 |
