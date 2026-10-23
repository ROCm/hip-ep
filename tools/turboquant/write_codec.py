#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Write a versioned runtime codec from an exported rotation/cb_k/cb_v NPZ.

This preserves the supplied constants; it never regenerates a rotation or
changes codebooks. Export the actual standalone model's constants first.
"""

import argparse
from pathlib import Path
import struct
import numpy as np
from reference import validate_constants


def codec_bytes(rotation, cb_k, cb_v):
    rotation, cb_k = validate_constants(rotation, cb_k)
    _, cb_v = validate_constants(rotation, cb_v)
    payload = b"".join(
        np.ascontiguousarray(x, dtype="<f4").tobytes() for x in (rotation, cb_k, cb_v)
    )
    checksum = 14695981039346656037
    for byte in payload:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    header = struct.pack(
        "<8IQ", 0x34515448, 1, 256, 16, 1, 132, 0, len(payload), checksum
    )
    return header + payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("constants", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.constants, allow_pickle=False) as data:
        result = codec_bytes(data["rotation"], data["cb_k"], data["cb_v"])
    args.output.write_bytes(result)
