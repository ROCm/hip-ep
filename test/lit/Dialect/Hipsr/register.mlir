// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr
//
//===----------------------------------------------------------------------===//
// Checks that hipsr shows up in hip-mlir-opt's list of registered dialects.
// --show-dialects prints the dialect names and exits, so no input IR is needed.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --show-dialects | FileCheck %s
// CHECK: hipsr
