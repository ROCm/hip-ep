// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-pad-matmul-nbits-weights retypes an int4 weight carrier to
// [N, K/2 + pad] with row_bytes = K/2 and sets b_row_bytes on its users; the
// pad is the weight's row-pads entry, or default-row-pad when the stride
// aliases. Without either (a device whose table sets none) nothing changes.
// hip-externalize-constants then stores it padded (full constants file) or
// records the row layout for the runtime loader (streaming).

// RUN: split-file %s %t
// RUN: mkdir -p %t/full %t/stream
// RUN: %python -c "import sys; open(sys.argv[1], 'wb').write(bytes((i * 7 + 1) % 251 for i in range(4096)))" %t/w.bin
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256' %t/pad.mlir | FileCheck %s --check-prefix=PAD
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights %t/pad.mlir | FileCheck %s --check-prefix=NOTABLE
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256' %t/skip.mlir | FileCheck %s --check-prefix=SKIP
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256 row-pads=2:4096:128:0=16' %t/pad.mlir | FileCheck %s --check-prefix=TABLE16
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256 row-pads=2:4096:128:0=0' %t/pad.mlir | FileCheck %s --check-prefix=TABLE0
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:2880:64:0=32' %t/skip.mlir | FileCheck %s --check-prefix=LISTED
// RUN: not hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=24' %t/pad.mlir 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:48:16:0=16' %t/skip.mlir | FileCheck %s --check-prefix=MISALIGNED
// RUN: not hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=24' %t/pad.mlir 2>&1 | FileCheck %s --check-prefix=BADDEFAULT
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/stream skip-constant-data=true' pad.mlir | FileCheck %s --check-prefix=STREAM
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='default-row-pad=256' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/full' pad.mlir | FileCheck %s --check-prefix=FULL
// RUN: %python -c "import sys; s = open(sys.argv[1], 'rb').read(); d = open(sys.argv[2], 'rb').read(); assert len(d) == 2 * 2304, len(d); assert d[0:2048] == s[0:2048] and d[2304:4352] == s[2048:4096]; assert d[2048:2304] == bytes(256) and d[4352:4608] == bytes(256)" %t/w.bin %t/full/model.constants.bin

//--- pad.mlir
// K = 4096: K/2 = 2048 bytes is 16 cache lines, gcd(16, 64) = 16 >= 8.
// PAD-LABEL: func.func @pad
// PAD:         %[[W:.*]] = hip.constant {location = "w.bin", offset = 0 : i64, row_bytes = 2048 : i64, size = 4096 : i64} : tensor<2x2304xui8>
// PAD:         hip.matmul_nbits(%{{.*}}) ins(%{{.*}}, %[[W]], %{{.*}} : tensor<4x4096xf16>, tensor<2x2304xui8>, tensor<2x32xf16>)
// PAD-SAME:      b_row_bytes = 2304 : i64
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

// No table: the weight is left to the runtime (dense, repacked per call).
// NOTABLE-LABEL: func.func @pad
// NOTABLE:         hip.constant {location = "w.bin", offset = 0 : i64, size = 4096 : i64} : tensor<2x32x64xui8>
// NOTABLE-NOT:     b_row_bytes

// A listed weight takes its table pad; a listed pad of 0 keeps it dense and
// marks the use with b_row_bytes = K/2 so the runtime does not repack it.
// TABLE16-LABEL: func.func @pad
// TABLE16:         hip.constant {{.*}}row_bytes = 2048 : i64{{.*}} : tensor<2x2064xui8>
// TABLE16:         b_row_bytes = 2064 : i64
// TABLE0-LABEL:  func.func @pad
// TABLE0:          hip.constant {location = "w.bin", offset = 0 : i64, size = 4096 : i64} : tensor<2x32x64xui8>
// TABLE0:          b_row_bytes = 2048 : i64
// BAD: malformed row-pads entry '2:4096:128:0=24'
// BADDEFAULT: default-row-pad 24 is not a non-negative multiple of 16

// STREAM:      module attributes
// STREAM-SAME: hipdnn.constant_row_bytes = array<i64: 2048>
// STREAM-SAME: hipdnn.constant_row_pitches = array<i64: 2304>
// STREAM-SAME: hipdnn.constant_sizes = array<i64: 4608>
// STREAM:      memref.global "private" @hip_ext_constant_0 : memref<2x2304xui8>

// FULL:        module attributes
// FULL-NOT:    hipdnn.constant_row_bytes
// FULL-SAME:   hipdnn.constant_sizes = array<i64: 4608>

//--- skip.mlir
// SKIP-LABEL: func.func @not_aliasing
// SKIP:         hip.constant {location = "w.bin", offset = 0 : i64, size = 2880 : i64} : tensor<2x45x32xui8>
// SKIP-NOT:     b_row_bytes
// A weight the default rule leaves alone is padded when the table lists it.
// LISTED-LABEL: func.func @not_aliasing
// LISTED:         hip.constant {{.*}}row_bytes = 1440 : i64{{.*}} : tensor<2x1472xui8>
// LISTED:         b_row_bytes = 1472 : i64
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

// A listed pad that would leave the pitch off the 16-byte B loads (K/2 = 24
// plus 16 is 40) keeps the weight dense.
// MISALIGNED-LABEL: func.func @misaligned_pitch
// MISALIGNED:         hip.constant {location = "w.bin", offset = 0 : i64, size = 48 : i64} : tensor<2x3x8xui8>
// MISALIGNED-NOT:     b_row_bytes
func.func @misaligned_pitch(%ctx: !hip.context, %a: tensor<4x48xf16>,
                            %s: tensor<2x3xf16>) -> tensor<4x2xf16> {
  %w = hip.constant {location = "w.bin", offset = 0 : i64, size = 48 : i64}
      : tensor<2x3x8xui8>
  %e = tensor.empty() : tensor<4x2xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<4x48xf16>, tensor<2x3x8xui8>, tensor<2x3xf16>)
      outs(%e : tensor<4x2xf16>)
      {K = 48 : i64, N = 2 : i64, bits = 4 : i64, block_size = 16 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<4x2xf16>
  return %y : tensor<4x2xf16>
}
