// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_stft
// CHECK-NOT: onnx.STFT

// Mirrors Kokoro's iSTFTNet STFT call: f32 (1, 320) signal, frame_step=4,
// frame_length=20, onesided=1, Hann window.  Result shape:
//   n_frames = (320 - 20) / 4 + 1 = 76
//   n_freqs  = 20 / 2 + 1         = 11
module {
  func.func @main_graph(%signal: tensor<1x320xf32>) -> tensor<1x76x11x2xf32> {
    %step   = "onnx.Constant"() {value = dense<4>  : tensor<i64>} : () -> tensor<i64>
    %wlen   = "onnx.Constant"() {value = dense<20> : tensor<i64>} : () -> tensor<i64>
    %window = "onnx.Constant"() {
      value = dense<[0.0, 0.0244717, 0.0954915, 0.2061074, 0.3454915, 0.5,
                     0.6545085, 0.7938926, 0.9045085, 0.9755283, 1.0,
                     0.9755283, 0.9045085, 0.7938926, 0.6545085, 0.5,
                     0.3454915, 0.2061074, 0.0954915, 0.0244717]>
        : tensor<20xf32>} : () -> tensor<20xf32>
    %0 = "onnx.STFT"(%signal, %step, %window, %wlen) {onesided = 1 : si64}
        : (tensor<1x320xf32>, tensor<i64>, tensor<20xf32>, tensor<i64>)
            -> tensor<1x76x11x2xf32>
    "onnx.Return"(%0) : (tensor<1x76x11x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
