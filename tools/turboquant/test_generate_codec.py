#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Generated assets are valid, reproducible locally, and protect existing codecs."""

import hashlib
import struct

import numpy as np
import pytest
from generate_codec import generate
from write_codec import codec_bytes


def test_generated_codec_and_provenance(tmp_path):
    manifest = generate(tmp_path)
    data = (tmp_path / "codec.bin").read_bytes()
    with np.load(tmp_path / "constants.npz", allow_pickle=False) as arrays:
        pi, ck, cv = arrays["rotation"], arrays["cb_k"], arrays["cb_v"]
        np.testing.assert_allclose(pi @ pi.T, np.eye(256), atol=2e-6)
        assert np.all(np.diff(ck) > 0)
        np.testing.assert_allclose(ck, -ck[::-1], atol=1e-7)
        np.testing.assert_array_equal(ck, cv)
        assert data == codec_bytes(pi, ck, cv)
    assert struct.unpack("<8I", data[:32]) == (
        0x34515448,
        1,
        256,
        16,
        1,
        132,
        0,
        262272,
    )
    assert manifest["files"]["codec.bin"] == hashlib.sha256(data).hexdigest()
    assert (
        generate(tmp_path, expected_sha256=manifest["files"]["codec.bin"]) == manifest
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate(tmp_path / "wrong-hash", expected_sha256="0" * 64)
    assert not (tmp_path / "wrong-hash").exists()
    with pytest.raises(ValueError, match="Refusing to replace"):
        generate(tmp_path, seed=43)
    assert (tmp_path / "codec.bin").read_bytes() == data
