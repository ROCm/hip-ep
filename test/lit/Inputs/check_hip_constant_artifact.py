#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

import json
import pathlib
import sys


binary_path = pathlib.Path(sys.argv[1])
json_path = pathlib.Path(sys.argv[2])

expected_binary = bytes([7]) + bytes(63) + bytes([1, 2, 3, 4])
actual_binary = binary_path.read_bytes()
assert actual_binary == expected_binary, (
    f"unexpected constants binary: {actual_binary.hex()}"
)

expected_json = {
    "version": 1,
    "binary_file": "model.constants.bin",
    "num_constants": 2,
    "total_bytes": 68,
    "constants": [
        {
            "name": "hip_ext_constant_plugin_weight_0",
            "element_type": "i8",
            "shape": [],
            "offset": 0,
            "size": 1,
            "alignment": 64,
        },
        {
            "name": "hip_ext_constant_1",
            "element_type": "i8",
            "shape": [4],
            "offset": 64,
            "size": 4,
            "alignment": 64,
        },
    ],
}
actual_json = json.loads(json_path.read_text(encoding="utf-8"))
assert actual_json == expected_json, f"unexpected constants JSON: {actual_json}"
