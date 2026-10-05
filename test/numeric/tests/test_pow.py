#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Pow with a constant scalar exponent.

Integer exponents and 0.5 / -1 stay on the Mul / Sqrt / Reciprocal
decomposition. Other finite scalar exponents use hip.pow.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_pow_model(dtype, shape, exponent: float):
    tp = np_to_onnx_type(dtype)
    x = helper.make_tensor_value_info("X", tp, shape)
    y = helper.make_tensor_value_info("Y", tp, shape)
    exp = numpy_helper.from_array(
        np.array(exponent, dtype=dtype), name="exponent"
    )
    node = helper.make_node("Pow", ["X", "exponent"], ["Y"])
    return make_model_from_nodes([node], [x], [y], initializers=[exp])


class TestPow:
    @pytest.mark.parametrize(
        "dtype,exponent,atol",
        [
            (np.float32, 2.0, 1e-5),
            (np.float32, 2.2, 1e-5),
            (np.float32, 0.45454547, 1e-5),
            (np.float16, 2.2, 1e-2),
            (np.float16, 0.45454547, 1e-2),
        ],
    )
    def test_scalar_exponent(self, model_runner, dtype, exponent, atol):
        shape = [2, 16]
        model = _make_pow_model(dtype, shape, exponent)
        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 2.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)
