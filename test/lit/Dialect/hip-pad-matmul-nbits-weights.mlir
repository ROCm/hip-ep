// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-pad-matmul-nbits-weights retypes an aliasing-prone int4 weight carrier
// to [N, K/2 + 128] with row_bytes = K/2 and sets b_row_bytes on its users;
// hip-externalize-constants then stores it padded (full constants file) or
// records the row layout for the runtime loader (streaming).

// RUN: split-file %s %t
// RUN: mkdir -p %t/full %t/stream
// RUN: %python -c "import sys; open(sys.argv[1], 'wb').write(bytes((i * 7 + 1) % 251 for i in range(4096)))" %t/w.bin
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights %t/pad.mlir | FileCheck %s --check-prefix=PAD
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights %t/skip.mlir | FileCheck %s --check-prefix=SKIP
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/stream skip-constant-data=true' pad.mlir | FileCheck %s --check-prefix=STREAM
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/full' pad.mlir | FileCheck %s --check-prefix=FULL
// RUN: %python -c "import sys; s = open(sys.argv[1], 'rb').read(); d = open(sys.argv[2], 'rb').read(); assert len(d) == 2 * 2176, len(d); assert d[0:2048] == s[0:2048] and d[2176:4224] == s[2048:4096]; assert d[2048:2176] == bytes(128) and d[4224:4352] == bytes(128)" %t/w.bin %t/full/model.constants.bin

//--- pad.mlir
// K = 4096: K/2 = 2048 bytes is 16 cache lines, gcd(16, 64) = 16 >= 8.
// PAD-LABEL: func.func @pad
// PAD:         %[[W:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, row_bytes = 2048 : i64, size = 4096 : i64} : tensor<2x2176xui8>
// PAD:         hip.matmul_nbits(%{{.*}}) ins(%{{.*}}, %[[W]], %{{.*}} : tensor<4x4096xf16>, tensor<2x2176xui8>, tensor<2x32xf16>)
// PAD-SAME:      b_row_bytes = 2176 : i64
func.func @pad(%ctx: !hip.context, %a: tensor<4x4096xf16>,
               %s: tensor<2x32xf16>) -> tensor<4x2xf16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 4096 : i64}
      : tensor<2x32x64xui8>
  %e = tensor.empty() : tensor<4x2xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<4x4096xf16>, tensor<2x32x64xui8>, tensor<2x32xf16>)
      outs(%e : tensor<4x2xf16>)
      {K = 4096 : i64, N = 2 : i64, bits = 4 : i64, block_size = 128 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<4x2xf16>
  return %y : tensor<4x2xf16>
}

// STREAM:      module attributes
// STREAM-SAME: hipdnn.constant_row_bytes = array<i64: 2048>
// STREAM-SAME: hipdnn.constant_row_pitches = array<i64: 2176>
// STREAM-SAME: hipdnn.constant_sizes = array<i64: 4352>
// STREAM:      memref.global "private" @hip_ext_constant_0 : memref<2x2176xui8>

// FULL:        module attributes
// FULL-NOT:    hipdnn.constant_row_bytes
// FULL-SAME:   hipdnn.constant_sizes = array<i64: 4352>

//--- skip.mlir
// SKIP-LABEL: func.func @not_aliasing
// SKIP:         hip.constant {location = "w.bin", offset = 0 : i64, size = 2880 : i64} : tensor<2x45x32xui8>
// SKIP-NOT:     b_row_bytes
func.func @not_aliasing(%ctx: !hip.context, %a: tensor<4x2880xf16>,
                        %s: tensor<2x45xf16>) -> tensor<4x2xf16> {
  // K = 2880: K/2 = 1440 bytes is not a whole number of cache lines.
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 2880 : i64}
      : tensor<2x45x32xui8>
  %e = tensor.empty() : tensor<4x2xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<4x2880xf16>, tensor<2x45x32xui8>, tensor<2x45xf16>)
      outs(%e : tensor<4x2xf16>)
      {K = 2880 : i64, N = 2 : i64, bits = 4 : i64, block_size = 64 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<4x2xf16>
  return %y : tensor<4x2xf16>
}

// SKIP-LABEL: func.func @shared_weight
// SKIP:         hip.constant {location = "w.bin", offset = 0 : i64, size = 4096 : i64} : tensor<2x32x64xui8>
// SKIP-NOT:     b_row_bytes
func.func @shared_weight(%ctx: !hip.context, %a: tensor<4x4096xf16>,
                         %s: tensor<2x32xf16>)
    -> (tensor<4x2xf16>, tensor<2x32x64xui8>) {
  // A second, non-MatMulNBits user keeps the dense layout.
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 4096 : i64}
      : tensor<2x32x64xui8>
  %e = tensor.empty() : tensor<4x2xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<4x4096xf16>, tensor<2x32x64xui8>, tensor<2x32xf16>)
      outs(%e : tensor<4x2xf16>)
      {K = 4096 : i64, N = 2 : i64, bits = 4 : i64, block_size = 128 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<4x2xf16>
  return %y, %w : tensor<4x2xf16>, tensor<2x32x64xui8>
}
