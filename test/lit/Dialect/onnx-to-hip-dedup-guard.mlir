// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: %python %S/../Inputs/check_onnx_to_hip_dedup.py \
// RUN:   --source-root %S/../../..
// RUN: %python %S/../Inputs/check_onnx_to_hip_dedup.py --self-test

// This file intentionally contains no IR. The RUN lines prevent recursive
// ONNX-to-HIP sources and the six reduction converters from regaining local
// shape/destination skeletons, including extra occurrences in reviewed files.
