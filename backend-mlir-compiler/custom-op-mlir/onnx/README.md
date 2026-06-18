<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Gather CPU fallback ONNX

`gather_cpu_fb_axis0_fp16_i64.onnx` is a **minimal valid ONNX** graph (opset 13 `Gather`, `axis=0`) used only by the EP debug path `HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS=Gather`.

MorphiZen’s **MLIR** backend implements `model_proto_serialize_as_string` as **MLIR serialization**, not ONNX protobuf, so we **cannot** build this session from `morphizen_cxx::Model` + ORT `Session` in-process. The EP embeds this file via CMake + `scripts/embed_binary_c_array.py`, emitting **`gather_cpu_fb_onnx_data.inc`** into the build dir; **`cpu_fallback_bridge.cpp` `#include`s it** (single TU, avoids MSVC `LNK2019` if a separate generated `.cpp` is not linked). The embedded graph uses **int64** indices in ONNX; the EP **widens int32** host staging to int64 before `CreateTensor` when `indices_element_size_bytes==4`. **Output rank:** ONNX Gather output rank is `indices_rank + (data_rank - 1)`; if the MLIR memref uses the same linear size with a lower rank, the EP supplies the **canonical ONNX output shape** to ORT so IoBinding matches the embedded Gather. The EP runs the inner ORT session via **`Ort::Session::Run` with caller-owned output `Ort::Value`s** (not `IoBinding`): some ORT builds still mis-handle external-memory tensors in `IoBinding` for tiny graphs, producing bogus Gather index diagnostics. Inputs are still addressed as **`Input_0` / `Input_1` / `Output_0`**.

## Regenerate

From repo root (conda env with `onnx`):

```bash
python -c "
from onnx import TensorProto, helper
d = helper.make_tensor_value_info('Input_0', TensorProto.FLOAT16, ['d0','d1'])
i = helper.make_tensor_value_info('Input_1', TensorProto.INT64, ['i0','i1'])
o = helper.make_tensor_value_info('Output_0', TensorProto.FLOAT16, ['o0','o1','o2'])
n = helper.make_node('Gather', ['Input_0','Input_1'], ['Output_0'], axis=0)
g = helper.make_graph([n], 'g', [d, i], [o])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
import onnx
onnx.checker.check_model(m)
onnx.save(m, 'backend-mlir-compiler/custom-op-mlir/onnx/gather_cpu_fb_axis0_fp16_i64.onnx')
"
```

Other `(axis, data dtype, indices dtype)` combinations require additional `.onnx` assets and a small dispatch table in `cpu_fallback_bridge.cpp`.
