// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// The filter of a real ConvTranspose does not arrive as an inline
// DenseElementsAttr: the importer externalizes any initializer past a small
// byte threshold, so it arrives as a byte range in the model's external data
// file. Verify that such a carrier is read and decomposed, and that a
// process-local memory-address carrier is left alone instead of being
// dereferenced.
//
// Test cases:
// 1. file_source   - file-backed filter -> residues carry the file's values
// 2. memory_source - memory-address filter, op survives
// ============================================================================

// RUN: split-file %s %t
// The four f32 taps 1.0, 2.0, 3.0, 4.0, little-endian.
// RUN: printf '\000\000\200\077\000\000\000\100\000\000\100\100\000\000\200\100' > %t/weights.bin
// RUN: cd %t && hip-mlir-opt file-source.mlir --hip-decompose-conv-transpose --canonicalize | FileCheck %s --check-prefix=FILE
// RUN: hip-mlir-opt %t/memory-source.mlir --hip-decompose-conv-transpose --canonicalize | FileCheck %s --check-prefix=MEMORY

// A 2x2 filter at stride 2 splits into four single-tap residues, one per
// filter position, so each residue constant pins one value read from the file.
// FILE-LABEL: func.func @file_source
// FILE-NOT: hip.conv_transpose
// FILE-DAG: hip.constant {value = dense<1.000000e+00> : tensor<1x1x1x1xf32>}
// FILE-DAG: hip.constant {value = dense<2.000000e+00> : tensor<1x1x1x1xf32>}
// FILE-DAG: hip.constant {value = dense<3.000000e+00> : tensor<1x1x1x1xf32>}
// FILE-DAG: hip.constant {value = dense<4.000000e+00> : tensor<1x1x1x1xf32>}

// The memory address is process-local and owned by whoever recorded it, which
// this pass cannot establish -- resolving it would dereference a bare integer.
// MEMORY-LABEL: func.func @memory_source
// MEMORY: hip.conv_transpose

//--- file-source.mlir
module {
  func.func @file_source(%ctx: !hip.context, %x: tensor<1x1x2x2xf32>)
      -> tensor<1x1x4x4xf32> {
    %w = hip.constant {location = "weights.bin", offset = 0 : i64,
                       size = 16 : i64} : tensor<1x1x2x2xf32>
    %init = tensor.empty() : tensor<1x1x4x4xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x2x2xf32>,
                                               tensor<1x1x2x2xf32>)
        outs(%init : tensor<1x1x4x4xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x4x4xf32>
    return %y : tensor<1x1x4x4xf32>
  }
}

//--- memory-source.mlir
module {
  func.func @memory_source(%ctx: !hip.context, %x: tensor<1x1x2x2xf32>)
      -> tensor<1x1x4x4xf32> {
    %w = hip.constant {memory_address = 4096 : i64,
                       size = 16 : i64} : tensor<1x1x2x2xf32>
    %init = tensor.empty() : tensor<1x1x4x4xf32>
    %y = hip.conv_transpose(%ctx) ins(%x, %w : tensor<1x1x2x2xf32>,
                                               tensor<1x1x2x2xf32>)
        outs(%init : tensor<1x1x4x4xf32>)
        {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0],
         dilations = [1, 1], output_padding = [0, 0], group = 1 : i64}
        : tensor<1x1x4x4xf32>
    return %y : tensor<1x1x4x4xf32>
  }
}
