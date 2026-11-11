// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-materialize-host-scalars on constant globals with
// a signed or unsigned integer element type.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-materialize-host-scalars %s 2>&1 | FileCheck %s

// --- An unsigned constant global (a QLinear ui8 zero point) is staged into
//     host scratch. arith.constant takes only signless integers, so the value
//     is built as i8 and cast to ui8 before the host store. ---
memref.global "private" constant @__constant_xui8 : memref<ui8> = dense<200>
// CHECK-LABEL: func.func @unsigned_constant_global_hip_consumer
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK:         %[[SCRATCH:.*]] = hip.get_host_scratch(%[[CTX]],
// CHECK-NOT:     memref.get_global @__constant_xui8
// CHECK:         %[[V:.*]] = memref.view %[[SCRATCH]]{{.*}} : memref<?xi8> to memref<ui8>
// CHECK:         %[[RAW:.*]] = arith.constant -56 : i8
// CHECK:         %[[U:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i8 to ui8
// CHECK:         memref.store %[[U]], %[[V]][] : memref<ui8>
// CHECK:         hip.mul(%[[CTX]]) ins(%[[V]], %{{.*}} : memref<ui8>, memref<ui8>)
func.func @unsigned_constant_global_hip_consumer(%ctx: !hip.context,
                                                 %x: memref<ui8>,
                                                 %out: memref<ui8>) {
  %c = memref.get_global @__constant_xui8 : memref<ui8>
  hip.mul(%ctx) ins(%c, %x : memref<ui8>, memref<ui8>) outs(%out : memref<ui8>)
  return
}
