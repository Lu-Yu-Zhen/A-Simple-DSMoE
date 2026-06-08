"""
MoE (Mixture of Experts) layer implementation.
Includes Expert, MoELayer, and auxiliary loss MoELoss for building MoE Transformer models.
"""
from typing import Optional, Union

import mindspore as ms
from mindspore import Tensor, nn, ops


__all__ = ["ExpertLayer", "MoELayer", "MoELoss"]


# ==================== Expert Definition ====================


class ExpertLayer(nn.Cell):
    """Single expert network: two FC layers + GELU activation."""

    def __init__(
        self,
        dim_model: int = 1024,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.ffn = ffn_dim
        self.fc1 = nn.Dense(dim_model, ffn_dim)
        self.fc2 = nn.Dense(ffn_dim, dim_model)
        self.act_fn = nn.GELU()

    def construct(self, x: Tensor):
        out = self.fc1(x)
        out = self.act_fn(out)
        out = self.fc2(out)
        return out


# ==================== RoutedExpert Layer Definition ====================


class MoELayer(nn.Cell):
    """
    RoutedExpert layer.
    Routes tokens to top-k experts via router gate,
    computes weighted sum of expert outputs and auxiliary load-balancing loss.
    """

    def __init__(
        self,
        dim_model: int = 1024,
        shared_ffn_dim: int = 1024,
        router_ffn_dim: int = 512,
        num_experts: int = 16,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.shared_ffn_dim = shared_ffn_dim
        self.router_ffn_dim = router_ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        self.dropout1 = nn.Dropout(p=dropout)

        self.shared_expert = ExpertLayer(dim_model, shared_ffn_dim)
        self.experts = nn.CellList([
            ExpertLayer(dim_model, router_ffn_dim) for _ in range(num_experts)
        ])

        self.router_gate = nn.Dense(dim_model, num_experts)

        self.softmax = nn.Softmax(axis=-1)
        self.topk = ops.TopK(sorted=True)
        self.onehot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.reduce_mean = ops.ReduceMean()
        self.cast = ops.Cast()

    def construct(self, x: Tensor):
        """Forward pass: returns MoE output and auxiliary loss."""
        if not isinstance(x, Tensor):
            raise TypeError(f"{x} must be a tensor")

        origin_shape = x.shape                # (B, L, D)
        x_flat = x.reshape(-1, origin_shape[-1])
        N = x_flat.shape[0]

        gate_prob = self._gate_forward(x_flat)
        topk_values, topk_indices = self._top_k_selection(gate_prob)
        mask = self._get_mask(topk_indices, N)
        dispatched_weights = self._normalize_dispatch_weights(mask, gate_prob)
        shared_output = self._shared_expert(x_flat)
        moe_output = self._expert_forward(shared_output, dispatched_weights, N, origin_shape)
        aux_loss = self._aux_loss_fn(mask, gate_prob) * self.num_experts

        return moe_output, aux_loss

    def _gate_forward(self, x_flat):
        """Router gate forward: compute per-token probability distribution over experts."""
        gate_logits = self.router_gate(x_flat)          # (N, num_experts)
        gate_prob = self.softmax(gate_logits)     # (N, num_experts)
        return gate_prob

    def _top_k_selection(self, gate_prob):
        """Select top-k experts and their weights."""
        topk_values, topk_indices = self.topk(gate_prob, self.top_k)
        return topk_values, topk_indices

    def _get_mask(self, topk_indices, N):
        """Generate one-hot dispatch mask marking which experts each token is assigned to."""
        indices_flat = topk_indices.reshape(-1)
        one_hot_flat = self.onehot(
            indices=indices_flat,
            depth=self.num_experts,
            on_value=Tensor(1.0, ms.float32),
            off_value=Tensor(0.0, ms.float32),
        )  # (N * top_k, num_experts)

        one_hot = one_hot_flat.reshape(N, self.top_k, self.num_experts)
        mask = self.reduce_sum(one_hot, axis=1)   # (N, 1, num_experts)
        mask = ops.reshape(mask, (N, self.num_experts))
        return mask

    def _normalize_dispatch_weights(self, mask, gate_prob):
        """Normalize dispatch weights so each token's expert weights sum to 1."""
        dispatch_weights = mask * gate_prob
        weights_sum = self.reduce_sum(dispatch_weights, -1)
        dispatch_weights = dispatch_weights / weights_sum
        return dispatch_weights

    def _shared_expert(self, x_flat):
        return self.shared_expert(x_flat)

    def _expert_forward(self, x_flat, dispatch_weights, N, origin_shape):
        """Compute weighted sum over all experts."""
        output = ops.zeros_like(x_flat)

        for i in range(self.num_experts):
            w = dispatch_weights[:, i]
            w = ops.reshape(w, (N, 1))

            expert_out = self.experts[i](x_flat)   # (N, D)
            output += w * expert_out               # (N, D)

        moe_output = ops.reshape(output, origin_shape)   # (B, L, D)
        return self.dropout1(moe_output)

    def _aux_loss_fn(self, mask, gate_prob):
        """Compute load-balancing auxiliary loss."""
        fraction_tokens = self.reduce_mean(mask, axis=0)
        avg_probs = self.reduce_mean(gate_prob, axis=0)
        aux_loss = self.reduce_sum(fraction_tokens * avg_probs, axis=0)
        aux_loss = ops.squeeze(aux_loss)
        return aux_loss


# ==================== Loss Function ====================


class MoELoss(nn.LossBase):
    """
    RoutedExpert total loss = main cross-entropy loss + alpha * auxiliary load-balancing loss.
    """

    def __init__(self, aux_loss, alpha: float = 0.1):
        super().__init__()
        self.aux_loss = aux_loss
        self.alpha = alpha
        self.main_loss_fn = nn.CrossEntropyLoss()

    def construct(self, logits, labels):
        main_loss = self.main_loss_fn(logits, labels)
        return main_loss + self.alpha * self.aux_loss