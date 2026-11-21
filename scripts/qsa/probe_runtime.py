#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Prove strict hip-ep GPU execution and raw GPU bindings before model load."""
# isort: skip_file

import json

from runtime import INSTALL, WORK, binding, register, run, session
import numpy as np
import onnx
import torch
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    device = register()
    print("Registered", device.ep_name, device.ep_vendor, device.device, flush=True)
    model = h.make_model(
        h.make_graph(
            [h.make_node("Add", ["x", "one"], ["y"])],
            "GPU_binding_probe",
            [h.make_tensor_value_info("x", T.FLOAT16, [1, 16])],
            [h.make_tensor_value_info("y", T.FLOAT16, [1, 16])],
            [nh.from_array(np.ones((1, 16), np.float16), "one")],
        ),
        opset_imports=[h.make_opsetid("", 21)],
        ir_version=10,
    )
    path = WORK / "gpu-binding-probe.onnx"
    onnx.save(model, path)
    sess = session(path)
    x = torch.arange(16, device="cuda", dtype=torch.float16)[None]
    y = torch.empty_like(x)
    io = binding(sess, {"x": x}, {"y": y})
    run(sess, io)
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    receipt = {
        "status": "passed",
        "install": str(INSTALL),
        "providers": sess.get_providers(),
        "cpu_fallback_disabled": True,
        "device": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "onnxruntime": __import__("onnxruntime").__version__,
        "bound_result": y.tolist(),
    }
    (WORK / "gpu-binding-probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
