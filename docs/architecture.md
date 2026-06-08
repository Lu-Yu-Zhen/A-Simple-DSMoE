# Architecture Overview / 架构概述

> Bilingual documentation — English sections are followed by Chinese translations.

---

## 1. System Architecture / 系统架构

```
+------------------------------------------------------------------+
|                         DSMoE Model                               |
|  +------------------------------------------------------------+  |
|  |                     Embedding (external)                    |  |
|  +------------------------------------------------------------+  |
|                               |                                    |
|                               v                                    |
|  +------------------------------------------------------------+  |
|  |                TransformerBlock × N (stacked)               |  |
|  |  +------------------------------------------------------+  |  |
|  |  |  +------------------+    +-------------------------+ |  |  |
|  |  |  | MultiHeadLatent  |    |      MoELayer           | |  |  |
|  |  |  |   Attention      |    |  +--------+----------+  | |  |  |
|  |  |  | (MLA)            |    |  | Shared | Routed   |  | |  |  |
|  |  |  |                  |    |  | Expert | Experts  |  | |  |  |
|  |  |  | W_dq → W_uq →   |    |  | (FFN)  | (FFN×N)  |  | |  |  |
|  |  |  | W_qr + RoPE     |    |  |        | top_k    |  | |  |  |
|  |  |  | W_dkv → W_uk/uv |    |  |        | gating   |  | |  |  |
|  |  |  | ↓ concat + softmax  |  |  +--------+----------+  | |  |  |
|  |  |  +------------------+    +-------------------------+ |  |  |
|  |  +------------------------------------------------------+  |  |
|  +------------------------------------------------------------+  |
|                               |                                    |
|                               v                                    |
|  +------------------------------------------------------------+  |
|  |                    Output (B, L, dim_model)                 |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Module Dependency / 模块依赖关系

```
DSMoE
  └── TransformerBlock × N
        ├── MultiHeadLatentAttention (MLA)
        │     ├── RMSNorm
        │     ├── Projections: W_dq, W_uq, W_qr, W_dkv, W_uk, W_uv, W_kr, W_o
        │     └── RoPE (Rotary Position Embedding)
        └── MoELayer
              ├── ExpertLayer (shared expert)
              ├── ExpertLayer × num_experts (routed experts)
              ├── Router gate (nn.Dense + Softmax)
              └── Auxiliary loss (load balancing)
```

**中文架构概述:**

DSMoE 主体由 N 个 `TransformerBlock` 堆叠而成。每个 Block 包含两个核心子模块：

1. **MultiHeadLatentAttention (MLA)** — 多头潜在注意力，使用低秩压缩减少 KV 缓存
2. **MoELayer** — 混合专家前馈层，包含一个共享专家 + 多个路由专家 + Top-K 门控

---

## 2. Data Flow / 数据流

### 2.1 MoELayer Forward / MoELayer 前向传播

```
Input: (B, L, D)
  │
  ├── flatten to (N, D), N = B × L
  │
  ├── Router Gate ──→ gate_prob: (N, num_experts)  (Softmax over experts)
  │     │
  │     └── Top-K ──→ topk_indices: (N, K), topk_values: (N, K)
  │           │
  │           └── One-Hot Mask ──→ mask: (N, num_experts)
  │
  ├── Shared Expert ──→ shared_out: (N, D)
  │
  ├── Dispatch Weights = mask × gate_prob, normalized
  │
  ├── For each expert i: output += weight[i] × expert[i](shared_out)
  │
  ├── Dropout + reshape back to (B, L, D)
  │
  └── Auxiliary Loss = sum(mean(mask) × mean(gate_prob)) × num_experts
```

### 2.2 Training Pipeline / 训练流程

```
Input tokens
  │
  ├── Token Embedding (external) ──→ (B, L, dim_model)
  │
  └── DSMoE (forward)
        │
        ├── Block 1: MLA → residual → MoE FFN → residual
        ├── Block 2: MLA → residual → MoE FFN → residual
        ├── ...
        └── Block N: MLA → residual → MoE FFN → residual
              │
              └── Output: (B, L, dim_model), total_aux_loss
                      │
                      └── Loss = CrossEntropyLoss(output, labels) + alpha × aux_loss
```

---

## 3. Key Design Decisions / 关键设计决策

### 3.1 Multi-Head Latent Attention (MLA)

**English:**

MLA is inspired by DeepSeek-V2. The key idea is to compress both Key and Value into a low-rank latent space via shared down-projection (`W_dkv`), significantly reducing KV cache memory during inference. Query also uses LoRA-style low-rank decomposition (`W_dq` → `W_uq`). Both query and key have a "nope" (no positional encoding) component and a "rope" (RoPE) component.

| Component | Description |
|---|---|
| Query (nope) | `x → W_dq → W_uq → (B, n_heads, L, qh_nope_head_dim)` |
| Query (rope) | `x → W_qr → RoPE → broadcast to n_heads` |
| Key (nope) | `x → W_dkv → W_uk → (B, n_heads, L, qh_nope_head_dim)` |
| Key (rope) | `x → W_kr → RoPE → broadcast to n_heads` |
| Value | `x → W_dkv → W_uv → (B, n_heads, L, v_head_dim)` |

**中文:**

MLA 受 DeepSeek-V2 启发，核心思想是通过共享下投影 (`W_dkv`) 将 Key 和 Value 压缩到低秩潜在空间，显著减少推理时的 KV 缓存。Query 也使用 LoRA 风格的低秩分解 (`W_dq` → `W_uq`)。Query 和 Key 都分为无位置编码（nope）和 RoPE 两部分。

### 3.2 Shared + Routed Expert Design / 共享 + 路由专家设计

**English:**

This implementation adopts DeepSeek-MoE's shared expert isolation pattern:

- **Shared Expert** (`shared_ffn_dim`): Activated for every token, captures common knowledge
- **Routed Experts** (`num_experts` × `router_ffn_dim`): Each token activates only the top-k most relevant experts
- **Load-Balancing Loss**: An auxiliary loss that encourages even token distribution across routed experts, preventing expert collapse

The shared expert generally has a larger FFN dimension than routed experts, as it handles universal patterns.

**中文:**

本实现采用了 DeepSeek-MoE 的共享专家隔离模式：

- **共享专家** (`shared_ffn_dim`)：每个 token 都激活，负责捕获通用知识
- **路由专家** (`num_experts` 个 × `router_ffn_dim`)：每个 token 只激活 Top-K 个最相关的专家
- **负载均衡损失**：辅助损失函数，促使 token 在各路由专家间均匀分布，防止专家坍塌

共享专家的 FFN 维度通常大于路由专家，因为它需要处理通用模式。

### 3.3 Pre-Norm with RMSNorm / 使用 RMSNorm 的 Pre-Norm 结构

**English:**

Each TransformerBlock uses a **pre-norm** architecture with **RMSNorm**, where normalization is applied before each sub-layer (MLA and MoE FFN) rather than after. This improves training stability, especially for deep models. RMSNorm is computationally cheaper than LayerNorm while delivering comparable performance.

**中文:**

每个 TransformerBlock 采用 **pre-norm** 结构搭配 **RMSNorm**，归一化在每个子层（MLA 和 MoE FFN）之前执行，而非之后。这提高了训练稳定性，尤其对于深层模型。RMSNorm 比 LayerNorm 计算成本更低，同时性能相当。

### 3.4 Residual Connections / 残差连接

**English:**

Standard residual connections wrap each sub-layer:

```
x = x + Dropout(MLA(RMSNorm(x)))
x = x + Dropout(MoE_FFN(RMSNorm(x)))
```

This helps gradient flow through the deep stack of TransformerBlocks and prevents vanishing gradients.

**中文:**

标准的残差连接包裹每个子层，帮助梯度在深层 TransformerBlock 堆叠中顺畅传播，防止梯度消失。

---

## 4. Computation Flow Summary / 计算流程总结

```
Input (B, L, dim_model)
  │
  ├── RMSNorm ──→ MLA ──→ Dropout ──→ + (residual)
  │
  ├── RMSNorm ──→ MoELayer ──→ Dropout ──→ + (residual)
  │
  │     ┌── MoELayer internal:
  │     │   ├── Shared expert (all tokens)
  │     │   ├── Router gate → Top-K → Dispatch
  │     │   ├── Routed experts (selected)
  │     │   ├── Weighted sum + Dropout
  │     │   └── Auxiliary loss (scalar)
  │     │
  │     └── Forward pass returns: output + aux_loss
  │
  ├── Repeat × num_blocks layers
  │
  └── Final output + cumulative aux_loss
```
