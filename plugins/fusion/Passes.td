//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Pass definition for the out-of-tree fusion plugin.
//===----------------------------------------------------------------------===//

#ifndef HIP_PLUGIN_FUSION_PASSES
#define HIP_PLUGIN_FUSION_PASSES

include "mlir/Pass/PassBase.td"

def HipFuseMulAddPass : Pass<"hip-fuse-mul-add", "mlir::func::FuncOp"> {
  let summary = "Fuse hip.add(hip.mul(x, b), a) into hip.fused_mul_add (plugin)";
  let description = [{
    Plugin-contributed tensor-domain fusion pass. Matches
    `hip.add(hip.mul(x, b), a)` (either operand ordering) where the
    intermediate `hip.mul` result is single-use, and rewrites it to the
    plugin op `hip.fused_mul_add` computing `(b * x) + a`.

    Demonstrates that a dynamically-loaded plugin can register both a real
    `hip.*` op and a named pass with no rebuild of `hip-mlir-opt`.
  }];
  let dependentDialects = ["mlir::hip::HipDialect"];
}

#endif // HIP_PLUGIN_FUSION_PASSES
