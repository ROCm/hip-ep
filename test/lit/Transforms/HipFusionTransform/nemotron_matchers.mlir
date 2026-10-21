// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// These generated cases check fusion guards, shared users, and shape queries.
// They run without a GPU. Kernel and model checks run separately.
// RUN: %python %S/../../../../scripts/nemotron_perf/test_linear_attention_fusion.py hip-mlir-opt
// RUN: %python %S/../../../../scripts/nemotron_perf/test_linear_attention_heads.py hip-mlir-opt
// RUN: %python %S/../../../../scripts/nemotron_perf/test_swiglu_f32_fusion.py hip-mlir-opt
// RUN: %python %S/../../../../scripts/nemotron_perf/test_rounded_scaled_add_fusion.py hip-mlir-opt
