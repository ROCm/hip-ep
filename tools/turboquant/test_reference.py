#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independent packing, reconstruction, and attention invariants."""

import numpy as np
import pytest
from reference import (
    attention,
    encode,
    pack_indices,
    partitioned_attention,
    reconstruct,
    unpack_rows,
)


@pytest.fixture(scope="module")
def constants():
    q, _ = np.linalg.qr(np.random.default_rng(21).normal(size=(256, 256)))
    # Deliberately unequal K/V codebooks catch accidental table interchange.
    return (
        q.astype(np.float32),
        np.linspace(-0.19, 0.19, 16, dtype=np.float32),
        np.linspace(-0.23, 0.23, 16, dtype=np.float32),
    )


def test_all_nibbles_and_inline_norm():
    idx = np.tile(np.arange(16, dtype=np.uint8), 16)[None]
    raw = pack_indices(idx, np.array([1.5], np.float32))
    assert raw.shape == (1, 132)
    assert raw[0, :4].tolist() == [0, 0, 192, 63]
    assert raw[0, 4:12].tolist() == [16, 50, 84, 118, 152, 186, 220, 254]
    i, n = unpack_rows(raw)
    np.testing.assert_array_equal(i, idx)
    np.testing.assert_array_equal(n, [1.5])
    np.testing.assert_array_equal(raw.view(np.float16).view(np.uint8), raw)


def test_zero_tiny_and_large_finite(constants):
    pi, ck, _ = constants
    x = np.zeros((3, 256), np.float16)
    x[1, 19] = np.nextafter(np.float16(0), np.float16(1))
    x[2] = np.float16(65504)
    packed = encode(x, pi, ck)
    i, n = unpack_rows(packed)
    np.testing.assert_array_equal(packed[0], 0)
    assert n[1] > 0 and np.isfinite(n).all()
    assert np.isfinite(reconstruct(packed, pi, ck)).all()
    assert n[2] > 65504  # Norm must remain FP32, not FP16.


@pytest.mark.parametrize("length", [1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129])
def test_partition_invariance_and_rotation(constants, length):
    pi, ck, cv = constants
    rng = np.random.default_rng(length)
    q = rng.normal(size=(24, 256)).astype(np.float16)
    k = encode(rng.normal(size=(4, length, 256)).astype(np.float16), pi, ck)
    v = encode(rng.normal(size=(4, length, 256)).astype(np.float16), pi, cv)
    expected = attention(q, k, v, pi, ck, cv, 0.0625)
    for partitions in (1, 4, 16, length + 1):
        actual = partitioned_attention(q, k, v, pi, ck, cv, 0.0625, partitions)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_equal_scores_constant_values(constants):
    pi, ck, cv = constants
    q = np.zeros((24, 256), np.float16)
    keys = encode(np.zeros((4, 2, 256), np.float16), pi, ck)
    values = encode(np.ones((4, 2, 256), np.float16), pi, cv)
    expected = reconstruct(values, pi, cv)[:, 0].repeat(6, axis=0)
    for partitions in (1, 2, 4):
        np.testing.assert_allclose(
            partitioned_attention(q, keys, values, pi, ck, cv, 0.0625, partitions),
            expected,
            atol=1e-6,
        )


def test_lower_index_tie():
    cb = (np.arange(-8, 8, dtype=np.float32) + 0.5) / 16
    x = np.zeros((1, 256), np.float16)
    x[0, 0] = 1
    codes, _ = unpack_rows(encode(x, np.eye(256, dtype=np.float32), cb))
    assert codes[0, 1] == 7  # Zero is exactly halfway between entries 7 and 8.


@pytest.mark.parametrize(
    "case",
    [
        "negative_index",
        "large_index",
        "negative_norm",
        "nan_norm",
        "nan_input",
        "wrong_dtype",
        "wrong_rotation",
    ],
)
def test_invalid_contract(constants, case):
    pi, ck, _ = constants
    idx = np.zeros((1, 256), np.int32)
    n = np.ones(1, np.float32)
    x = np.ones((1, 256), np.float16)
    with pytest.raises(ValueError):
        if case == "negative_index":
            idx[0, 0] = -1
        elif case == "large_index":
            idx[0, 0] = 16
        elif case == "negative_norm":
            n[0] = -1
        elif case == "nan_norm":
            n[0] = np.nan
        elif case == "nan_input":
            x[0, 0] = np.nan
            encode(x, pi, ck)
        elif case == "wrong_dtype":
            encode(x.astype(np.float32), pi, ck)
        elif case == "wrong_rotation":
            encode(x, np.ones_like(pi), ck)
        pack_indices(idx, n)
