// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-pad-matmul-nbits-weights retypes an int4 weight carrier to
// [N, K/2 + pad] with row_bytes = K/2 and sets b_row_bytes on its users; the
// pad is the weight's row-pads entry. An unlisted weight is left alone, even
// when its stride aliases (the runtime repacks it per call), and so is every
// weight without a table.
// hip-externalize-constants then stores it padded (full constants file) or
// records the row layout for the runtime loader (streaming).

// RUN: split-file %s %t
// RUN: mkdir -p %t/full %t/stream
// RUN: %python -c "import sys; open(sys.argv[1], 'wb').write(bytes((i * 7 + 1) % 251 for i in range(4096)))" %t/w.bin
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=256' %t/pad.mlir | FileCheck %s --check-prefix=PAD
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights %t/pad.mlir | FileCheck %s --check-prefix=NOTABLE
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=4:4096:128:0=256,2:4096:128:1=256' %t/pad.mlir | FileCheck %s --check-prefix=NOTABLE
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=256' %t/skip.mlir | FileCheck %s --check-prefix=SKIP
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=16' %t/pad.mlir | FileCheck %s --check-prefix=TABLE16
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=0' %t/pad.mlir | FileCheck %s --check-prefix=TABLE0
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:2880:64:0=32' %t/skip.mlir | FileCheck %s --check-prefix=LISTED
// RUN: not hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=24' %t/pad.mlir 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:48:16:0=16' %t/skip.mlir | FileCheck %s --check-prefix=MISALIGNED
// RUN: not hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=16,2:4096:128:0=32' %t/pad.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=256' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/stream skip-constant-data=true' pad.mlir | FileCheck %s --check-prefix=STREAM
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=256' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/full' pad.mlir | FileCheck %s --check-prefix=FULL
// RUN: %python -c "import sys; s = open(sys.argv[1], 'rb').read(); d = open(sys.argv[2], 'rb').read(); assert len(d) == 2 * 2304, len(d); assert d[0:2048] == s[0:2048] and d[2304:4352] == s[2048:4096]; assert d[2048:2304] == bytes(256) and d[4352:4608] == bytes(256)" %t/w.bin %t/full/model.constants.bin
// Byte orders (16 rows of K = 4096): XOR-swizzled lines and 16-row blocks.
// RUN: mkdir -p %t/blkfull %t/blkstream %t/xorfull
// RUN: %python -c "import sys; open(sys.argv[1], 'wb').write(bytes((i * 7 + 1) % 251 for i in range(32768)))" %t/w16.bin
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=blocks' %t/layout.mlir | FileCheck %s --check-prefix=BLOCKS
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=xor' %t/layout.mlir | FileCheck %s --check-prefix=XOR
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=blocks,16:4096:128:1=blocks' %t/layout_dense.mlir | FileCheck %s --check-prefix=LAYDENSE
// RUN: hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=2:4096:128:0=blocks' %t/pad.mlir | FileCheck %s --check-prefix=NOTABLE
// RUN: not hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=tiles' %t/layout.mlir 2>&1 | FileCheck %s --check-prefix=BADLAYOUT
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=blocks' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/blkstream skip-constant-data=true' layout.mlir | FileCheck %s --check-prefix=BLKSTREAM
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=blocks' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/blkfull' layout.mlir
// RUN: cd %t && hip-mlir-opt --hip-pad-matmul-nbits-weights='row-pads=16:4096:128:0=xor' --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/xorfull' layout.mlir
// Every byte where the layout puts it: block (n / 16, j) at j * 512, row n % 16
// at 32 * (n % 16); with XOR lines, line l of row n at l ^ ((n / 4) % 16).
// RUN: %python -c "import sys; s = open(sys.argv[1], 'rb').read(); d = open(sys.argv[2], 'rb').read(); assert len(d) == 32768, len(d); assert all(d[(b // 32) * 512 + n * 32 + b % 32] == s[n * 2048 + b] for n in range(16) for b in range(2048))" %t/w16.bin %t/blkfull/model.constants.bin
// RUN: %python -c "import sys; s = open(sys.argv[1], 'rb').read(); d = open(sys.argv[2], 'rb').read(); assert len(d) == 32768, len(d); assert all(d[n * 2048 + (((b // 128) ^ ((n // 4) % 16)) * 128) + b % 128] == s[n * 2048 + b] for n in range(16) for b in range(2048))" %t/w16.bin %t/xorfull/model.constants.bin

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

// Not listed (no table, or a table that lists other weights only): the
// aliasing weight is left to the runtime (dense, repacked per call).
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
// DUPLICATE: duplicate row-pads entry '2:4096:128:0=32'

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
// A listed weight is padded whether or not its stride aliases.
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

//--- layout.mlir
// BLOCKS-LABEL: func.func @layout
// BLOCKS:         %[[W:.*]] = hip.constant {location = "w16.bin", offset = 0 : i64, row_bytes = 2048 : i64, size = 32768 : i64, weight_layout = 2 : i64} : tensor<16x2048xui8>
// BLOCKS:         hip.matmul_nbits(%{{.*}}) ins(%{{.*}}, %[[W]], %{{.*}} : tensor<1x4096xf16>, tensor<16x2048xui8>, tensor<16x32xf16>)
// BLOCKS-SAME:      b_layout = 2 : i64, b_row_bytes = 2048 : i64
// XOR-LABEL:    func.func @layout
// XOR:            hip.constant {{.*}}weight_layout = 1 : i64} : tensor<16x2048xui8>
// XOR:            b_layout = 1 : i64, b_row_bytes = 2048 : i64
// BADLAYOUT: malformed row-pads entry '16:4096:128:0=tiles'
// BLKSTREAM:      module attributes
// BLKSTREAM-SAME: hipdnn.constant_row_bytes = array<i64: 2048>
// BLKSTREAM-SAME: hipdnn.constant_row_pitches = array<i64: 2048>
// BLKSTREAM-SAME: hipdnn.constant_sizes = array<i64: 32768>
// BLKSTREAM-SAME: hipdnn.constant_weight_layouts = array<i32: 2>
func.func @layout(%ctx: !hip.context, %a: tensor<1x4096xf16>,
                  %s: tensor<16x32xf16>) -> tensor<1x16xf16> {
  %w = hip.constant {location = "w16.bin", offset = 0 : i64, size = 32768 : i64}
      : tensor<16x32x64xui8>
  %e = tensor.empty() : tensor<1x16xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<1x4096xf16>, tensor<16x32x64xui8>, tensor<16x32xf16>)
      outs(%e : tensor<1x16xf16>)
      {K = 4096 : i64, N = 16 : i64, bits = 4 : i64, block_size = 128 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x16xf16>
  return %y : tensor<1x16xf16>
}

//--- layout_dense.mlir
// A biased weight and one with fp16 zero points keep the dense rows.
// LAYDENSE-LABEL: func.func @layout_bias
// LAYDENSE-NOT:     weight_layout
// LAYDENSE-NOT:     b_layout
// LAYDENSE-LABEL: func.func @layout_fp16_zp
// LAYDENSE-NOT:     weight_layout
// LAYDENSE-NOT:     b_layout
func.func @layout_bias(%ctx: !hip.context, %a: tensor<1x4096xf16>,
                       %s: tensor<16x32xf16>, %bias: tensor<16xf16>)
    -> tensor<1x16xf16> {
  %w = hip.constant {location = "w16.bin", offset = 0 : i64, size = 32768 : i64}
      : tensor<16x32x64xui8>
  %e = tensor.empty() : tensor<1x16xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<1x4096xf16>, tensor<16x32x64xui8>, tensor<16x32xf16>)
      bias(%bias : tensor<16xf16>)
      outs(%e : tensor<1x16xf16>)
      {K = 4096 : i64, N = 16 : i64, bits = 4 : i64, block_size = 128 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x16xf16>
  return %y : tensor<1x16xf16>
}
func.func @layout_fp16_zp(%ctx: !hip.context, %a: tensor<1x4096xf16>,
                          %s: tensor<16x32xf16>, %zp: tensor<16x32xf16>)
    -> tensor<1x16xf16> {
  %w = hip.constant {location = "w16.bin", offset = 0 : i64, size = 32768 : i64}
      : tensor<16x32x64xui8>
  %e = tensor.empty() : tensor<1x16xf16>
  %y = hip.matmul_nbits(%ctx) ins(%a, %w, %s :
      tensor<1x4096xf16>, tensor<16x32x64xui8>, tensor<16x32xf16>)
      zero_points(%zp : tensor<16x32xf16>)
      outs(%e : tensor<1x16xf16>)
      {K = 4096 : i64, N = 16 : i64, bits = 4 : i64, block_size = 128 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x16xf16>
  return %y : tensor<1x16xf16>
}
