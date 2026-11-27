// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-to-llvm --split-input-file --verify-diagnostics %s

func.func @mismatched_b(%ctx: !hipsr.context,
                        %a: memref<2x2xbf16, #hipsr.mem<device>>,
                        %b: memref<2x2xf16, #hipsr.mem<device>>,
                        %out: memref<2x2xbf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{failed to legalize operation 'hipsr.matmul'}}
  hipsr.matmul(%ctx)
      ins(%a, %b : memref<2x2xbf16, #hipsr.mem<device>>, memref<2x2xf16, #hipsr.mem<device>>)
      outs(%out : memref<2x2xbf16, #hipsr.mem<device>>)
  return
}

// -----

func.func @mismatched_output(%ctx: !hipsr.context,
                             %a: memref<2x2xf32, #hipsr.mem<device>>,
                             %b: memref<2x2xf32, #hipsr.mem<device>>,
                             %out: memref<2x2xf16, #hipsr.mem<device>>) {
  // expected-error @+1 {{failed to legalize operation 'hipsr.matmul'}}
  hipsr.matmul(%ctx)
      ins(%a, %b : memref<2x2xf32, #hipsr.mem<device>>, memref<2x2xf32, #hipsr.mem<device>>)
      outs(%out : memref<2x2xf16, #hipsr.mem<device>>)
  return
}

// -----

func.func @unsupported_f64(%ctx: !hipsr.context,
                           %a: memref<2x2xf64, #hipsr.mem<device>>,
                           %b: memref<2x2xf64, #hipsr.mem<device>>,
                           %out: memref<2x2xf64, #hipsr.mem<device>>) {
  // expected-error @+1 {{failed to legalize operation 'hipsr.matmul'}}
  hipsr.matmul(%ctx)
      ins(%a, %b : memref<2x2xf64, #hipsr.mem<device>>, memref<2x2xf64, #hipsr.mem<device>>)
      outs(%out : memref<2x2xf64, #hipsr.mem<device>>)
  return
}
