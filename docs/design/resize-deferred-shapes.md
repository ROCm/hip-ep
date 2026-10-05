<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Resize shapes still deferred

Dynamic NCHW `Resize` whose `scales` vector is a compile-time constant is
lowered. The three models below still fail conversion and are intentionally
out of scope until a follow-up.

| Model | Resize nodes | Why it is not covered |
|---|---|---|
| `\\cfssvm2-lif5\ai_models\migraphx_models\MLSR\swarr\generator_amdnet_tiny_lr_luma_66GOP_pq_float32_f16.onnx` | `Resize_35`, `Resize_54`, `Resize_73` | `sizes` is the shape of a different activation. The batch dim is not this input's dim, and on `Resize_35` / `Resize_54` the channel count changes (64→48, 32→16). The resampled axes are not a 1–3 axis NCHW suffix. |
| `\\cfssvm2-lif5\ai_models\migraphx_models\Topaz\Gigapixel\gfrf-v2-fp32-1024x1024.onnx` | `/Resize` | The one `Resize` after the NHWC→NCHW transpose (`perm = [0, 3, 1, 2]`). `sizes` was `Concat(Slice(Shape(Transpose)), [512, 512])`, rewritten to `hip.gather` of the pre-transpose shape with indices `[0, 3, 1, 2]`. |
| `\\cfssvm2-lif5\ai_models\migraphx_models\Topaz\PhotoAI\gfrf-v2-fp32-1024x1024.onnx` | `/Resize` | Same node and transpose as the Gigapixel graph. |
