"""Quick start demo: build a small DSMoE and run a forward pass."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mindspore as ms
from mindspore import Tensor, ops

from dsmoe.model.ds_moe import DSMoE, TransformerBlock

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


def main():
    block = TransformerBlock(
        dim_model=256, n_heads=8, shared_ffn_dim=256,
        router_ffn_dim=128, num_experts=16, top_k=6,
    )
    model = DSMoE(block, num_blocks=6)

    x = ops.randn((1, 5, 256), dtype=ms.float32)
    output, aux_loss = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss}")


if __name__ == "__main__":
    main()