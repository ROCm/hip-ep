// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Out-of-tree plugin PoC — the §4 / R-1 negative path.
//
// Loads the *_nobuf* plugin variant, which is identical to the positive plugin
// EXCEPT it omits the bufferization external-model attach (compiled with
// -DHIPPOC_SKIP_BUFFERIZE_ATTACH).
//
// The op still registers and the pass still fires — the IR transforms
// successfully — but --one-shot-bufferize then fails deep inside legalization,
// exactly the "easy-to-miss, hard-to-diagnose" footgun documented in §4 of the
// DLL plugin interface requirements doc. This test makes that failure mode an
// executable, regression-guarded demonstration: a plugin that forgets the
// external model produces an op that loads, transforms, and then detonates at
// bufferize.
//===----------------------------------------------------------------------===//

// REQUIRES: hip_plugins
//
// NOTE: driven via --pass-pipeline (not the --hip-fuse-mul-add shorthand) — see
// the positive test hip-fuse-mul-add-plugin.mlir for why the shorthand flag is
// avoided when a dialect plugin and a pass plugin are loaded together.
// RUN: not hip-mlir-opt \
// RUN:   --load-dialect-plugin=%hip_fusion_plugin_nobuf \
// RUN:   --load-pass-plugin=%hip_fusion_plugin_nobuf \
// RUN:   --pass-pipeline="builtin.module(func.func(hip-fuse-mul-add),one-shot-bufferize{bufferize-function-boundaries})" \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: {{failed to bufferize op|was not bufferized|failed to legalize}}
func.func @fuse_mul_add_nobuf(
    %ctx: !hip.context,
    %x: tensor<4xf32>,
    %b: tensor<4xf32>,
    %a: tensor<4xf32>) -> tensor<4xf32> {
  %init_mul = tensor.empty() : tensor<4xf32>
  %m = hip.mul(%ctx)
      ins(%x, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%init_mul : tensor<4xf32>)
      -> tensor<4xf32>
  %init_add = tensor.empty() : tensor<4xf32>
  %y = hip.add(%ctx)
      ins(%m, %a : tensor<4xf32>, tensor<4xf32>)
      outs(%init_add : tensor<4xf32>)
      -> tensor<4xf32>
  return %y : tensor<4xf32>
}
