// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --hip-resolve-extern-constants %t/gather.mlir | FileCheck %s
// RUN: not hip-mlir-opt --hip-resolve-extern-constants %t/other-use.mlir 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: hip-mlir-opt --hip-resolve-extern-constants %t/qmoe.mlir | FileCheck %s --check-prefix=QMOE

// CHECK-LABEL: func.func @gather
// CHECK:       %[[T:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident} : memref<4x8xf16>
// CHECK:       hip.gather(%{{.*}}) ins(%[[T]], %{{.*}} : memref<4x8xf16>, memref<2xi64>)
// CHECK:       hip.get_constant(%{{.*}}, %{{.*}}) : memref<4x8xf16>
// CHECK:       %[[Q:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident} : memref<4x4xui8>
// CHECK:       hip.gather_block_quantized(%{{.*}}) ins(%[[Q]], %{{.*}}, %{{.*}} : memref<4x4xui8>

// ERR: error: disk-resident constant has no GPU copy; only hip.gather / hip.gather_block_quantized data and hip.qmoe expert operands may use it

// QMOE-LABEL: func.func @qmoe
// QMOE-COUNT-6: hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident}
// QMOE:         hip.qmoe
// QMOE-SAME:    hip.expert_offload_node = array<i64: 0, 2>
// QMOE:         hip.qmoe
// QMOE-NOT:     hip.expert_offload_node
// QMOE:         hip.qmoe
// QMOE-SAME:    hip.expert_offload_node = array<i64: 1, 2>

//--- gather.mlir
module {
  memref.global "private" @ple : memref<4x8xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }
  memref.global "private" @dense : memref<4x8xf16> alignment = 64 {
    hip.external_data = {index = 1 : i64, offset = 0 : i64, size = 64 : i64}
  }
  memref.global "private" @ple_q4 : memref<4x4xui8> alignment = 64 {
    hip.external_data = {disk_resident, index = 2 : i64, offset = 0 : i64, size = 16 : i64}
  }
  func.func @gather(%ctx: !hip.context, %idx: memref<2xi64>,
                    %out: memref<2x8xf16>, %scales: memref<4x1xf16>)
      -> memref<4x8xf16> {
    %t = memref.get_global @ple : memref<4x8xf16>
    hip.gather(%ctx) ins(%t, %idx : memref<4x8xf16>, memref<2xi64>)
        outs(%out : memref<2x8xf16>) {axis = 0 : i64}
    %d = memref.get_global @dense : memref<4x8xf16>
    %q = memref.get_global @ple_q4 : memref<4x4xui8>
    hip.gather_block_quantized(%ctx)
        ins(%q, %idx, %scales : memref<4x4xui8>, memref<2xi64>, memref<4x1xf16>)
        outs(%out : memref<2x8xf16>)
        {bits = 4 : i64, block_size = 8 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
    return %d : memref<4x8xf16>
  }
}

//--- other-use.mlir
module {
  memref.global "private" @ple : memref<4x8xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }
  func.func @other_use(%ctx: !hip.context) -> memref<4x8xf16> {
    %t = memref.get_global @ple : memref<4x8xf16>
    return %t : memref<4x8xf16>
  }
}

//--- qmoe.mlir
module {
  memref.global "private" @w1 : memref<4x64x16xui8> alignment = 64 {
    hip.external_data = {disk_resident, index = 0 : i64, offset = 0 : i64, size = 4096 : i64}
  }
  memref.global "private" @s1 : memref<4x64x2xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 1 : i64, offset = 0 : i64, size = 1024 : i64}
  }
  memref.global "private" @b1 : memref<4x64xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 2 : i64, offset = 0 : i64, size = 512 : i64}
  }
  memref.global "private" @w2 : memref<4x64x8xui8> alignment = 64 {
    hip.external_data = {disk_resident, index = 3 : i64, offset = 0 : i64, size = 2048 : i64}
  }
  memref.global "private" @s2 : memref<4x64x1xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 4 : i64, offset = 0 : i64, size = 512 : i64}
  }
  memref.global "private" @b2 : memref<4x64xf16> alignment = 64 {
    hip.external_data = {disk_resident, index = 5 : i64, offset = 0 : i64, size = 512 : i64}
  }
  func.func @qmoe(%ctx: !hip.context, %x: memref<1x2x64xf16>,
                  %r: memref<2x4xf16>, %rw1: memref<4x64x16xui8>,
                  %rs1: memref<4x64x2xf16>, %rw2: memref<4x64x8xui8>,
                  %rs2: memref<4x64x1xf16>, %out: memref<1x2x64xf16>) {
    %w1 = memref.get_global @w1 : memref<4x64x16xui8>
    %s1 = memref.get_global @s1 : memref<4x64x2xf16>
    %b1 = memref.get_global @b1 : memref<4x64xf16>
    %w2 = memref.get_global @w2 : memref<4x64x8xui8>
    %s2 = memref.get_global @s2 : memref<4x64x1xf16>
    %b2 = memref.get_global @b2 : memref<4x64xf16>
    hip.qmoe(%ctx) ins(%x, %r, %w1, %s1, %w2, %s2 : memref<1x2x64xf16>, memref<2x4xf16>, memref<4x64x16xui8>, memref<4x64x2xf16>, memref<4x64x8xui8>, memref<4x64x1xf16>)
        fc1_bias(%b1 : memref<4x64xf16>) fc2_bias(%b2 : memref<4x64xf16>)
        outs(%out : memref<1x2x64xf16>)
        {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64}
    hip.qmoe(%ctx) ins(%x, %r, %rw1, %rs1, %rw2, %rs2 : memref<1x2x64xf16>, memref<2x4xf16>, memref<4x64x16xui8>, memref<4x64x2xf16>, memref<4x64x8xui8>, memref<4x64x1xf16>)
        outs(%out : memref<1x2x64xf16>)
        {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64}
    hip.qmoe(%ctx) ins(%x, %r, %w1, %s1, %w2, %s2 : memref<1x2x64xf16>, memref<2x4xf16>, memref<4x64x16xui8>, memref<4x64x2xf16>, memref<4x64x8xui8>, memref<4x64x1xf16>)
        fc1_bias(%b1 : memref<4x64xf16>) fc2_bias(%b2 : memref<4x64xf16>)
        outs(%out : memref<1x2x64xf16>)
        {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64}
    return
  }
}
