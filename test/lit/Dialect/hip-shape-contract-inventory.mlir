// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: %python %S/../Inputs/check_hip_shape_contracts.py \
// RUN:   --tblgen llvm-tblgen \
// RUN:   --project-include %S/../../../include \
// RUN:   --ops %S/../../../include/hip/Dialect/IR/HipOps.td \
// RUN:   --inventory %S/../../../docs/design/hip-shape-contract-inventory.md
// RUN: %python %S/../Inputs/check_hip_shape_contracts.py --self-test

// This file intentionally contains no IR. The RUN line verifies that every
// HIP DPS op inherits its exact reviewed shape-contract base, that the design
// inventory matches the TableGen source of truth, and that malformed guard
// fixtures fail the audit.
