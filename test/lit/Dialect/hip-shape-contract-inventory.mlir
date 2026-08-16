// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: %python %S/../Inputs/check_hip_shape_contracts.py \
// RUN:   --tblgen llvm-tblgen \
// RUN:   --project-include %S/../../../include \
// RUN:   --ops %S/../../../include/hip/Dialect/IR/HipOps.td \
// RUN:   --inventory %S/../../../docs/design/hip-shape-contract-inventory.md \
// RUN:   --reifiers %S/../../../lib/Dialect/IR/HipReifyResultShapesImpl.cpp \
// RUN:   --verifiers %S/../../../lib/Dialect/IR/HipDialect.cpp
// RUN: %python %S/../Inputs/check_hip_shape_contracts.py --self-test

// This file intentionally contains no IR. The RUN line verifies that every
// HIP destination-style op is classified as compute or control-flow DPS, every
// compute leaf inherits its exact reviewed shape-contract base, handwritten
// outs-lift fallback is guarded by its verifier, and malformed fixtures fail.
