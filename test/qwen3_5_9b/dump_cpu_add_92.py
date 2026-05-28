#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Dump CPU's add_92 value (and other rope-chain intermediates) for grid [2,8,8].
Used to compare against EP's runtime int64-add dumps."""

import os
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
MODEL = str(
    REPO_ROOT
    / "models"
    / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
    / "vision.onnx"
)
GRID = [2, 8, 8]

m = onnx.load(MODEL, load_external_data=False)
vi_map = {
    v.name: v
    for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)
}

# Add many rope-chain intermediates as outputs
TAPS = [
    "add_92",
    "add_62",
    "add_77",
    "add_107",
    "embedding",
    "embedding_2",
    "view_2",
    "view_4",
    "mul_117",
    "mul_106",
    "mul_94",
    "unsqueeze_4",
    "unsqueeze_1",
    "unsqueeze_18",
    "mul_144",
    "add_206",
    "add_191",
    "add_221",
    "repeat",
    "view_10",
    "_unsafe_view",
    "permute",
]

added = []
for tap in TAPS:
    if tap in vi_map and tap not in {o.name for o in m.graph.output}:
        m.graph.output.append(vi_map[tap])
        added.append(tap)

probe = str(
    REPO_ROOT
    / "models"
    / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
    / "vision_probe_dump.onnx"
)
onnx.save(m, probe, save_as_external_data=False)

rng = np.random.default_rng(0)
t, h, w = GRID
n = t * h * w
inputs = {
    "pixel_values": (rng.standard_normal((n, 1536)) * 0.1).astype(np.float16),
    "image_grid_thw": np.array([GRID], dtype=np.int64),
}

sess = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
results = sess.run(out_names, inputs)
out_map = dict(zip(out_names, results))

for tap in added:
    if tap not in out_map:
        continue
    arr = out_map[tap]
    print(f"[CPU] {tap}: shape={arr.shape} dtype={arr.dtype}", flush=True)
    flat = arr.flatten()
    n_print = min(16, flat.size)
    print(f"      vals[:{n_print}]={flat[:n_print].tolist()}", flush=True)
    if flat.size > n_print:
        print(f"      vals[-4:]={flat[-4:].tolist()}", flush=True)

os.remove(probe)
