#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Decode (M = 1) MatMulNBits shapes served by the dp4a path.

With K <= 512 the short-K dp4a kernel spreads each row's K/32 chunks over that
many lanes of a shared wave, one wave covering 32 / (K/32) columns. It reads
the scale and zero point of the block_size group a chunk falls in, so block
sizes above 32 are covered too. The DecodeDp4a LUT rows measured for
Qwen3.8-Flash-Next are pinned at the end.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

BITS = 4


def _model(rng, K: int, N: int, block_size: int, asym: bool):
    n_blocks = (K + block_size - 1) // block_size
    q_weight = rng.integers(0, 256, [N, n_blocks, block_size // 2], dtype=np.uint8)
    scales = rng.uniform(-0.02, 0.02, [N, n_blocks]).astype(np.float16)
    inits = [
        numpy_helper.from_array(q_weight, name="W"),
        numpy_helper.from_array(scales, name="scales"),
    ]
    inputs = ["X", "W", "scales"]
    if asym:
        # Two 4-bit zero points per byte, each row padded to whole bytes.
        zp = rng.integers(0, 256, [N * ((n_blocks + 1) // 2)], dtype=np.uint8)
        inits.append(numpy_helper.from_array(zp, name="zp"))
        inputs.append("zp")
    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=BITS,
        block_size=block_size,
        accuracy_level=4,
    )
    return make_model_from_nodes(
        [node],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 1, K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 1, N])],
        initializers=inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _run(model_runner, K: int, N: int, block_size: int, asym: bool, seed: int):
    rng = np.random.default_rng(seed)
    model = _model(rng, K, N, block_size, asym)
    x = rng.uniform(-1, 1, [1, 1, K]).astype(np.float16)
    actual, expected = model_runner.run_sample(model, [x])
    compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)


class TestMatMulNBitsShortK:
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "zp"])
    @pytest.mark.parametrize(
        "K,N",
        [(320, 10240), (160, 8192), (512, 2048), (96, 333), (32, 100)],
        ids=["flash_next_k320", "k160", "k512", "k96_tail", "k32_one_lane"],
    )
    def test_block32(self, model_runner, K, N, asym):
        _run(model_runner, K, N, 32, asym, seed=13)

    # Several 32-wide chunks share one scale and zero point.
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "zp"])
    @pytest.mark.parametrize(
        "K,N,block_size",
        [(320, 2048, 64), (384, 1000, 64), (256, 2560, 128), (512, 777, 128)],
        ids=["k320_bs64", "k384_bs64", "k256_bs128", "k512_bs128"],
    )
    def test_wide_blocks(self, model_runner, K, N, block_size, asym):
        _run(model_runner, K, N, block_size, asym, seed=17)


@pytest.mark.parametrize(
    "K,N",
    [(10240, 324), (6144, 2560), (2560, 6144), (2560, 10240)],
    ids=["k10240_n324", "k6144", "n6144", "n10240"],
)
def test_decode_flash_next_lut_shapes(model_runner, K, N):
    """Decode shapes whose DecodeDp4a LUT row was measured for Qwen3.8-Flash-Next."""
    _run(model_runner, K, N, 32, asym=False, seed=31)
