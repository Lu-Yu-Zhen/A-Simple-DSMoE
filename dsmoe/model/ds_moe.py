import copy

import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor

from model.expert import MoELayer

__all__ = ["RMSNorm", "MultiHeadLatentAttention", "TransformerBlock", "DSMoE"]


class RMSNorm(nn.Cell):
    """
    RMS 归一化层（Root Mean Square Layer Normalization）。

    与标准 LayerNorm 不同，RMSNorm 仅使用均方根进行归一化，去掉了均值中心化操作，
    计算效率更高，同时在 Transformer 模型中表现相当。

    公式: RMSNorm(x) = x / RMS(x) * gamma,  其中 RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        normalize_shape: 归一化的形状，可以是 int 或 tuple
        eps (float): 防止除零的小常数，默认 1e-6
    """

    def __init__(self, normalize_shape, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if isinstance(normalize_shape, int):
            normalize_shape = (normalize_shape, )
        self.normalize_shape = normalize_shape
        self.weight = Parameter(ops.ones(normalize_shape), name="rms_norm_weight")

    def construct(self, x: Tensor):
        # 等价公式: x * rsqrt(mean(x^2) + eps) * weight
        variance = ops.mean(ops.pow(x, 2), axis=-1, keep_dims=True)
        x_normed = x * ops.rsqrt(variance + self.eps)
        return x_normed * self.weight


class MultiHeadLatentAttention(nn.Cell):
    """
    Multi-head Latent Attention (MLA) 模块。

    该模块是 DeepSeek-V2 中提出的高效注意力机制，核心思想是通过低秩压缩（LoRA-style）
    来降低 KV 缓存的显存占用，同时保持甚至提升模型表达能力。

    架构特点：
    - Query 和 Key 各自拆分为 nope（无位置编码）和 rope（含旋转位置编码）两部分
    - 使用低秩投影对 KV 进行压缩，减少 KV 缓存大小
    - 计算注意力时再将低秩表示升维还原到原始头空间

    Args:
        dim_model (int): 模型隐藏层维度，默认 2048
        n_heads (int): 注意力头数，默认 8
        q_lora_rank (int): Query 低秩压缩维度，默认 1024
        kv_lora_rank (int): Key/Value 低秩压缩维度，默认 512
        qh_nope_head_dim (int): 每个注意力头中无位置编码部分的维度，默认 128
        qk_rope_head_dim (int): 每个注意力头中含 RoPE 部分的维度，默认 64
        v_head_dim (int): 每个注意力头中 Value 的维度，默认 128
        rope_theta (float): RoPE 的 base frequency，默认 10000.0
        max_position_embeddings (int): 最大序列长度，默认 32000
    """

    def __init__(
            self,
            dim_model: int = 1024,
            n_heads: int = 16,
            q_lora_rank: int = 128,
            kv_lora_rank: int = 128,
            qh_nope_head_dim: int = 128,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            rope_theta: float = 10000.0,
            max_position_embeddings: int = 32000,
    ):
        """
        初始化 MLA 模块，构建所有投影矩阵。

        投影矩阵说明：
        - W_dq / W_uq: Query 的低秩压缩及恢复（LoRA 风格）
        - W_qr: Query 的 RoPE 部分投影
        - W_dkv: KV 共享的低秩下投影
        - W_uk / W_uv: Key 和 Value 的升维上投影
        - W_kr: Key 的 RoPE 部分投影
        - W_o: 输出投影
        """
        super().__init__()

        self.W_dq = nn.Dense(dim_model, q_lora_rank)
        self.W_uq = nn.Dense(q_lora_rank, n_heads * qh_nope_head_dim)
        self.W_qr = nn.Dense(dim_model, qk_rope_head_dim)

        self.W_dkv = nn.Dense(dim_model, kv_lora_rank)
        self.W_uk = nn.Dense(kv_lora_rank, n_heads * qh_nope_head_dim)
        self.W_uv = nn.Dense(kv_lora_rank, n_heads * v_head_dim)
        self.W_kr = nn.Dense(dim_model, qk_rope_head_dim)

        self.W_o = nn.Dense(qh_nope_head_dim * n_heads, dim_model)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

        self.dim_model = dim_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qh_nope_head_dim = qh_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.softmax_scale = ops.sqrt(
            Tensor(qh_nope_head_dim + qk_rope_head_dim, ms.float32)
        )

    def _get_position_ids(self, seq_len: int) -> Tensor:
        """生成位置索引序列 [0, 1, ..., seq_len-1]"""
        return ops.arange(seq_len, dtype=ms.int64)

    def _apply_rotary_emb(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """
        对输入张量应用旋转位置编码（RoPE）。

        原理：
        - 将最后维拆成两半，分别视为复数表示的实部和虚部
        - 根据位置生成旋转角度，对实部和虚部进行旋转变换
        - 公式: RoPE(x) = [x1*cos - x2*sin, x1*sin + x2*cos]

        Args:
            x: 输入张量，形状 (B, L, D)
            position_ids: 位置索引，形状 (L,)

        Returns:
            施加 RoPE 后的张量，形状与输入相同
        """
        B, L, D = x.shape
        half_dim = D // 2

        inv_freq = 1.0 / (
            self.rope_theta ** (ops.arange(0, half_dim, dtype=ms.float32) * 2.0 / float(D))
        )

        freqs = position_ids.astype(ms.float32).reshape(1, L, 1) * inv_freq.reshape(1, 1, half_dim)

        cos = ops.cos(freqs)
        sin = ops.sin(freqs)

        x1 = x[:, :, :half_dim]
        x2 = x[:, :, half_dim:]

        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        return ops.concat([x1_rotated, x2_rotated], axis=-1)

    def construct(
            self,
            x: Tensor
    ) -> Tensor:
        """
        MLA 前向传播。

        Args:
            x: 输入张量，形状 (B, L, dim_model)

        Returns:
            注意力输出，形状 (B, L, dim_model)
        """
        B, L, D = x.shape

        x_dq = self.W_dq(x)                                          # (B, L, q_lora_rank)
        x_uq = self.W_uq(x_dq)                                       # (B, L, n_heads * qh_nope_head_dim)
        x_r_reshape = self.reshape(x_uq, (B, self.n_heads, L, self.qh_nope_head_dim))  # (B, n_heads, L, qh_nope_head_dim)

        x_qr = self.W_qr(x)                                          # (B, L, qk_rope_head_dim)
        x_qr = self._apply_rotary_emb(x_qr, self._get_position_ids(L))  # 施加 RoPE
        # 广播到多头: (B, 1, L, qk_rope_head_dim) -> (B, n_heads, L, qk_rope_head_dim)
        x_qr = ops.broadcast_to(
            x_qr.reshape(B, 1, L, -1), (B, self.n_heads, L, x_qr.shape[-1])
        )

        q = ops.concat([x_r_reshape, x_qr], axis=-1)                 # (B, n_heads, L, qh_nope_head_dim + qk_rope_head_dim)

        x_dkv = self.W_dkv(x)                                        # (B, L, kv_lora_rank)，KV 共享低秩表示

        x_uk = self.W_uk(x_dkv)                                      # (B, L, n_heads * qh_nope_head_dim)
        x_k_reshape = self.reshape(x_uk, (B, self.n_heads, L, self.qh_nope_head_dim))  # (B, n_heads, L, qh_nope_head_dim)

        x_uv = self.W_uv(x_dkv)                                      # (B, L, n_heads * v_head_dim)
        v = ops.reshape(x_uv, (B, self.n_heads, L, self.v_head_dim)) # (B, n_heads, L, v_head_dim)

        x_kr = self.W_kr(x)                                          # (B, L, qk_rope_head_dim)
        x_kr = self._apply_rotary_emb(x_kr, self._get_position_ids(L))  # 施加 RoPE
        # 广播到多头
        x_kr = ops.broadcast_to(
            x_kr.reshape(B, 1, L, -1), (B, self.n_heads, L, x_kr.shape[-1])
        )

        k = ops.concat([x_k_reshape, x_kr], axis=-1)                 # (B, n_heads, L, qh_nope_head_dim + qk_rope_head_dim)
        k = self.transpose(k, (0, 1, 3, 2))                          # (B, n_heads, D_k, L)，转置为 (seq, dim) 以便 matmul

        muti_attn = ops.matmul(q, k)                                # (B, n_heads, L, L)
        muti_attn = muti_attn / self.softmax_scale                   # 1 / sqrt(d_head)
        muti_attn = ops.softmax(muti_attn, axis=-1)                  # 对最后一维（序列维度）做 softmax
        muti_attn = ops.matmul(muti_attn, v)                        # (B, n_heads, L, v_head_dim)

        attn = ops.reshape(muti_attn, (B, L, -1))
        attn_output = self.W_o(attn)                                 # (B, L, dim_model)

        return attn_output


class TransformerBlock(nn.Cell):
    """
    DSMoE Transformer 块。

    每个 Transformer 块包含：
    - 多头潜在注意力（MLA）：用于序列级特征交互
    - MoE 前馈网络（FFN）：包含共享专家 + 路由专家，用于 token 级特征变换
    - 两层 RMSNorm 归一化 + 残差连接

    前向传播流程:
        x = x + Dropout(MLA(RMSNorm(x)))
        x = x + Dropout(MoE_FFN(RMSNorm(x)))

    Args:
        dim_model (int): 模型隐藏层维度，默认 1024
        n_heads (int): 注意力头数，默认 16
        shared_ffn_dim (int): 共享专家 FFN 隐藏层维度，默认 1024
        router_ffn_dim (int): 路由专家 FFN 隐藏层维度，默认 512
        num_experts (int): 路由专家数量，默认 16
        top_k (int): 每个 token 激活的专家数，默认 2
        q_lora_rank (int): Query 低秩压缩维度，默认 128
        kv_lora_rank (int): Key/Value 低秩压缩维度，默认 128
        qh_nope_head_dim (int): 每头无位置编码维度，默认 128
        qk_rope_head_dim (int): 每头 RoPE 维度，默认 64
        v_head_dim (int): 每头 Value 维度，默认 128
        rope_theta (float): RoPE 基频，默认 10000.0
        max_position_embeddings (int): 最大序列长度，默认 32000
        dropout (float): Dropout 比例，默认 0.1
    """

    def __init__(
            self,
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
            dropout: float = 0.1,
    ):
        super().__init__()

        self.muti_latent_attn = MultiHeadLatentAttention(
            dim_model=dim_model,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qh_nope_head_dim=qh_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.moe_layer = MoELayer(
            dim_model=dim_model,
            shared_ffn_dim=shared_ffn_dim,
            router_ffn_dim=router_ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

        self.rms_norm1 = RMSNorm(dim_model)
        self.rms_norm2 = RMSNorm(dim_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.dim_model = dim_model
        self.n_heads = n_heads
        self.shared_ffn_dim = shared_ffn_dim
        self.router_ffn_dim = router_ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qh_nope_head_dim = qh_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout

    def __repr__(self):
        return (f"TransformerBlock(dim_model={self.dim_model}, n_heads={self.n_heads}, shared_ffn_dim={self.shared_ffn_dim}, router_ffn_dim={self.router_ffn_dim}, num_experts={self.num_experts}, top_k={self.top_k}, q_lora_rank={self.q_lora_rank}, kv_lora_rank={self.kv_lora_rank}, "
                f"qh_nope_head_dim={self.qh_nope_head_dim}, qk_rope_head_dim={self.qk_rope_head_dim}, v_head_dim={self.v_head_dim}, rope_theta={self.rope_theta}, max_position_embeddings={self.max_position_embeddings}, dropout={self.dropout})")

    def construct(self, x: Tensor):
        """
        Transformer 块前向传播。

        流程:
        1. RMSNorm -> MLA -> Dropout -> 残差连接
        2. RMSNorm -> MoE FFN -> Dropout -> 残差连接

        Args:
            x: 输入张量，形状 (B, L, dim_model)

        Returns:
            Tuple[output, aux_loss]:
            - output: 块输出，形状 (B, L, dim_model)
            - aux_loss: MoE 路由的辅助负载均衡损失
        """
        input_data = x

        norm_input = self.rms_norm1(input_data)
        mla_result = self._mla_block(norm_input)
        input_data += mla_result
        norm_input_1 = self.rms_norm2(input_data)
        moe_result, aux_loss = self._moe_ffn_block(norm_input_1)
        input_data += moe_result

        return input_data, aux_loss

    def _mla_block(self, x: Tensor) -> Tensor:
        """
        MLA 注意力子块：经过多头潜在注意力后再做 Dropout。

        Args:
            x: 已归一化的输入张量，形状 (B, L, dim_model)

        Returns:
            Dropout 后的注意力输出，形状 (B, L, dim_model)
        """
        x = self.muti_latent_attn(x)
        return self.dropout1(x)

    def _moe_ffn_block(self, x: Tensor):
        """
        MoE 前馈子块：经过 MoE FFN 后再做 Dropout。

        Args:
            x: 已归一化的输入张量，形状 (B, L, dim_model)

        Returns:
            Tuple[ffn_output, aux_loss]:
            - ffn_output: Dropout 后的 FFN 输出，形状 (B, L, dim_model)
            - aux_loss: MoE 路由的辅助负载均衡损失
        """
        moe_output, aux_loss = self.moe_layer(x)
        return self.dropout2(moe_output), aux_loss


class DSMoE(nn.Cell):
    """
    DeepSeek MoE 模型主体。

    DSMoE 是 DeepSeek-V2 架构的核心模型类，由多个 TransformerBlock 堆叠而成。
    每个 Block 内部使用 MLA（多头潜在注意力）+ MoE（混合专家）FFN 结构。

    模型结构:
        Embedding -> [TransformerBlock × N] -> Output

    前向传播返回处理后的隐状态以及所有 Block 的辅助负载均衡损失总和。

    Args:
        vocab_size (int): 词表大小
        transformer_block (TransformerBlock): 一个配置好的 TransformerBlock 实例，
            用于 deepcopy 复制 N 份
        num_blocks (int): Transformer 层的堆叠数量，默认 12
    """

    def __init__(
            self,
            transformer_block: TransformerBlock,
            num_blocks: int = 12,
    ):
        super().__init__()

        transformer_block = TransformerBlock(
            dim_model=transformer_block.dim_model,
            n_heads=transformer_block.n_heads,
            shared_ffn_dim=transformer_block.shared_ffn_dim,
            router_ffn_dim=transformer_block.router_ffn_dim,
            num_experts=transformer_block.num_experts,
            top_k=transformer_block.top_k,
            q_lora_rank=transformer_block.q_lora_rank,
            kv_lora_rank=transformer_block.kv_lora_rank,
            qh_nope_head_dim=transformer_block.qh_nope_head_dim,
            qk_rope_head_dim=transformer_block.qk_rope_head_dim,
            v_head_dim=transformer_block.v_head_dim,
            rope_theta=transformer_block.rope_theta,
            max_position_embeddings=transformer_block.max_position_embeddings,
            dropout=transformer_block.dropout,
        )
        self.transformer_blocks = nn.CellList(
            [copy.deepcopy(transformer_block) for _ in range(num_blocks)]
        )

        self.transformer_block=transformer_block
        self.num_blocks = num_blocks

    def __len__(self):
        """
        返回模型中层数。

        Returns:
            int: 层数
        """
        params=[]
        for param in self.trainable_params():
            params.append(param)
        return len(params)

    def __repr__(self):
        return f"DSMoE(num_blocks={self.num_blocks}, transformer_block={self.transformer_block})"

    def construct(self, x: Tensor):
        processed_data = x
        total_aux_loss = 0.0

        for transformer_block in self.transformer_blocks:
            output, aux_loss = transformer_block(processed_data)
            processed_data = output
            total_aux_loss += aux_loss

        return processed_data, total_aux_loss

    def print_trainable_params(self):
        """
        打印模型中所有训练参数。

        Returns:
            None
        """
        for param in self.trainable_params():
            print(param)

    def show_cells(self):
        """
        显示模型中所有 TransformerBlock 实例。

        Returns:
            None
        """
        for name, cell in self.cells_and_names():
            print(f"{name}: {cell}")

    def total_params(self):
        """
        返回模型中所有训练参数的总数量。

        Returns:
            int: 可训练参数的数量
        """
        return sum([p.numel() for p in self.trainable_params()])
        
