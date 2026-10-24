// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the ONNX Resize conversion produces a hip.resize DPS op with all
// string attributes resolved to integer enums at compile time.  Also
// covers:
//   * variadic ONNX operands (X, NoValue roi, scales, NoValue sizes)
//   * rejection / pass-through of attribute defaults
//   * channels-last rank 4, packed as prefix N and window (H, W, C)
//   * single Variadic input form (only X — no extra operands at all)

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: 2D bilinear upsample with default half_pixel coord transform.
  // Variadic operand list: (X, roi=NoValue, scales).
  func.func @test_resize_linear_half_pixel(%arg0: tensor<1x3x16x16xf16>,
                                            %scales: tensor<4xf32>)
      -> tensor<1x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_linear_half_pixel
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x16x16xf16>
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<1x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x3x32x32xf16>
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<1x3x16x16xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x3x32x32xf16>)
    // mode=1 (linear), coord_transform=0 (half_pixel), nearest_mode=0
    // CHECK-SAME: coord_transform = 0
    // CHECK-SAME: mode = 1
    // CHECK-SAME: nearest_mode = 0

    return %y : tensor<1x3x32x32xf16>
  }

  // Test 2: 2D nearest with align_corners.
  func.func @test_resize_nearest_align_corners(%arg0: tensor<1x4x8x8xf32>,
                                                %scales: tensor<4xf32>)
      -> tensor<1x4x16x16xf32> {
    // CHECK-LABEL: func.func @test_resize_nearest_align_corners
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "nearest",
         coordinate_transformation_mode = "align_corners",
         nearest_mode = "round_prefer_floor"}
        : (tensor<1x4x8x8xf32>, none, tensor<4xf32>)
        -> tensor<1x4x16x16xf32>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 2
    // CHECK-SAME: mode = 0
    // CHECK-SAME: nearest_mode = 0
    return %y : tensor<1x4x16x16xf32>
  }

  // Test 3: 3D linear (volumetric upsample) with asymmetric coord.
  func.func @test_resize_3d_linear_asymmetric(%arg0: tensor<1x2x4x4x4xf32>,
                                                %scales: tensor<5xf32>)
      -> tensor<1x2x8x8x8xf32> {
    // CHECK-LABEL: func.func @test_resize_3d_linear_asymmetric
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "asymmetric"}
        : (tensor<1x2x4x4x4xf32>, none, tensor<5xf32>)
        -> tensor<1x2x8x8x8xf32>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 1
    return %y : tensor<1x2x8x8x8xf32>
  }

  // Test 4: dynamic batch (N) — verifies tensor.dim flowing into init.
  // Spatial dims must remain static.
  func.func @test_resize_dynamic_n(%arg0: tensor<?x3x16x16xf16>,
                                    %scales: tensor<4xf32>)
      -> tensor<?x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_dynamic_n
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x3x16x16xf16>
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<?x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<?x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DN:.*]] = tensor.dim %[[X]], %[[C0]] : tensor<?x3x16x16xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DN]]) : tensor<?x3x32x32xf16>
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<?x3x16x16xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<?x3x32x32xf16>)
    return %y : tensor<?x3x32x32xf16>
  }

  // Test 5: channels-last. Axes 1 and 2 change; batch and channel stay.
  func.func @test_resize_nhwc(%arg0: tensor<1x16x16x3xf32>,
                              %scales: tensor<4xf32>)
      -> tensor<1x32x32x3xf32> {
    // CHECK-LABEL: func.func @test_resize_nhwc
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x16x16x3xf32>, none, tensor<4xf32>)
        -> tensor<1x32x32x3xf32>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 0
    // CHECK-SAME: mode = 1
    return %y : tensor<1x32x32x3xf32>
  }

  // Test 6: empty roi supplied as a statically empty (0-element) tensor rather
  // than a NoValue -- what some ONNX exporters emit for an omitted roi. It must
  // be treated as absent so the Resize still converts (regression: previously
  // the "roi present" guard rejected it and onnx.Resize survived unlowered).
  func.func @test_resize_empty_roi_tensor(%arg0: tensor<1x3x16x16xf16>,
                                          %scales: tensor<4xf32>)
      -> tensor<1x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_empty_roi_tensor
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x16x16xf16>
    %roi = "onnx.Constant"() {value = dense<> : tensor<0xf32>}
        : () -> tensor<0xf32>
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x16x16xf16>, tensor<0xf32>, tensor<4xf32>)
        -> tensor<1x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<1x3x16x16xf16>)
    // CHECK-SAME: mode = 1
    return %y : tensor<1x3x32x32xf16>
  }

  // Test 7: dynamic H/W with constant scales [1, 1, 2, 2]. N and C are
  // copies; H and W are floor(dim * 2).
  func.func @test_resize_dynamic_spatial_nchw(%arg0: tensor<?x3x?x?xf16>)
      -> tensor<?x3x?x?xf16> {
    // CHECK-LABEL: func.func @test_resize_dynamic_spatial_nchw
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "nearest", coordinate_transformation_mode = "half_pixel"}
        : (tensor<?x3x?x?xf16>, none, tensor<4xf32>) -> tensor<?x3x?x?xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: arith.sitofp
    // CHECK: arith.mulf
    // CHECK: arith.fptosi
    // CHECK: tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x3x?x?xf16>
    // CHECK: hip.resize
    // CHECK: return
    return %y : tensor<?x3x?x?xf16>
  }

  // Test 8: a scales argument with dynamic spatial dims cannot be folded.
  func.func @test_resize_dynamic_scales_rejected(%arg0: tensor<?x3x?x?xf16>,
                                                  %scales: tensor<4xf32>)
      -> tensor<?x3x?x?xf16> {
    // CHECK-LABEL: func.func @test_resize_dynamic_scales_rejected
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "nearest", coordinate_transformation_mode = "half_pixel"}
        : (tensor<?x3x?x?xf16>, none, tensor<4xf32>) -> tensor<?x3x?x?xf16>
    // CHECK: onnx.Resize
    // CHECK-NOT: hip.resize
    // CHECK: return
    return %y : tensor<?x3x?x?xf16>
  }

  // Test 9: constant sizes. Extents are written into the type, so the
  // type-only plan applies and nothing is read back.
  func.func @test_resize_constant_sizes(%arg0: tensor<1x3x16x16xf32>)
      -> tensor<1x3x32x32xf32> {
    // CHECK-LABEL: func.func @test_resize_constant_sizes
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.NoValue"() {value} : () -> none
    %sizes = "onnx.Constant"() {value = dense<[1, 3, 32, 32]> : tensor<4xi64>}
        : () -> tensor<4xi64>
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x16x16xf32>, none, none, tensor<4xi64>)
        -> tensor<1x3x32x32xf32>
    // CHECK-NOT: hip.readback_scalar
    // CHECK-NOT: onnx.Resize
    // CHECK: tensor.empty() : tensor<1x3x32x32xf32>
    // CHECK: hip.resize
    // CHECK-NOT: prefix_count
    // CHECK: return
    return %y : tensor<1x3x32x32xf32>
  }

  // Test 10: runtime sizes. Each element is a host index before tensor.empty.
  // Static input extents beside dynamic outputs need the saved launch.
  func.func @test_resize_runtime_sizes(%arg0: tensor<?x3x1024x1024xf32>,
                                       %sizes: tensor<4xi64>)
      -> tensor<?x?x?x?xf32> {
    // CHECK-LABEL: func.func @test_resize_runtime_sizes
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "linear", coordinate_transformation_mode = "half_pixel",
         nearest_mode = "floor"}
        : (tensor<?x3x1024x1024xf32>, none, none, tensor<4xi64>)
        -> tensor<?x?x?x?xf32>
    // CHECK-NOT: onnx.Resize
    // CHECK-COUNT-4: hip.readback_scalar
    // CHECK: tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-SAME: tensor<?x?x?x?xf32>
    // CHECK: hip.resize
    // CHECK-SAME: prefix_count = 1
    // CHECK-SAME: spatial_rank = 3
    // CHECK: return
    return %y : tensor<?x?x?x?xf32>
  }

  // Test 11: a runtime sizes vector that resizes the leading axes does not
  // fit the copied prefix. The op stays onnx.Resize.
  func.func @test_resize_runtime_sizes_prefix_rejected(
      %arg0: tensor<1x3x8x8xf32>, %sizes: tensor<4xi64>)
      -> tensor<?x?x?x?xf32> {
    // CHECK-LABEL: func.func @test_resize_runtime_sizes_prefix_rejected
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "nearest", coordinate_transformation_mode = "asymmetric"}
        : (tensor<1x3x8x8xf32>, none, none, tensor<4xi64>)
        -> tensor<?x?x?x?xf32>
    // CHECK: onnx.Resize
    // CHECK-NOT: hip.resize
    // CHECK: return
    return %y : tensor<?x?x?x?xf32>
  }

  // Test 12: quantized nearest upsample. Empty roi and scales are the
  // 0-element tensors an exporter emits for omitted inputs; sizes is the
  // output shape. nearest_mode round_prefer_ceil is 1.
  func.func @test_resize_nearest_ui8(%arg0: tensor<1x256x13x13xui8>)
      -> tensor<1x256x26x26xui8> {
    // CHECK-LABEL: func.func @test_resize_nearest_ui8
    %roi = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    %scales = "onnx.Constant"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
    %sizes = "onnx.Constant"() {value = dense<[1, 256, 26, 26]> : tensor<4xi64>}
        : () -> tensor<4xi64>
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "nearest",
         coordinate_transformation_mode = "half_pixel",
         nearest_mode = "round_prefer_ceil"}
        : (tensor<1x256x13x13xui8>, tensor<0xf32>, tensor<0xf32>, tensor<4xi64>)
        -> tensor<1x256x26x26xui8>
    // CHECK-NOT: onnx.Resize
    // CHECK: tensor.empty() : tensor<1x256x26x26xui8>
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 0
    // CHECK-SAME: mode = 0
    // CHECK-SAME: nearest_mode = 1
    return %y : tensor<1x256x26x26xui8>
  }

  // Test 13: signless int8 nearest uses the same index-copy path.
  func.func @test_resize_nearest_i8(%arg0: tensor<1x128x26x26xi8>)
      -> tensor<1x128x52x52xi8> {
    // CHECK-LABEL: func.func @test_resize_nearest_i8
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.NoValue"() {value} : () -> none
    %sizes = "onnx.Constant"() {value = dense<[1, 128, 52, 52]> : tensor<4xi64>}
        : () -> tensor<4xi64>
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "nearest", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x128x26x26xi8>, none, none, tensor<4xi64>)
        -> tensor<1x128x52x52xi8>
    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: mode = 0
    return %y : tensor<1x128x52x52xi8>
  }

  // Test 14: linear on uint8 would interpolate stored codes. Leave it.
  func.func @test_resize_linear_ui8_rejected(%arg0: tensor<1x3x8x8xui8>)
      -> tensor<1x3x16x16xui8> {
    // CHECK-LABEL: func.func @test_resize_linear_ui8_rejected
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x8x8xui8>, none, tensor<4xf32>) -> tensor<1x3x16x16xui8>
    // CHECK: onnx.Resize
    // CHECK-NOT: hip.resize
    return %y : tensor<1x3x16x16xui8>
  }

  // Test 15: empty roi and scales passed as arguments, not folded constants.
  // sizes supplies the output shape; the converter reads it from the result type.
  func.func @test_resize_empty_roi(%arg0: tensor<1x512x6x6xf16>,
                                    %roi: tensor<0xf32>,
                                    %scales: tensor<0xf32>,
                                    %sizes: tensor<4xi64>)
      -> tensor<1x512x12x12xf16> {
    // CHECK-LABEL: func.func @test_resize_empty_roi
    %y = "onnx.Resize"(%arg0, %roi, %scales, %sizes)
        {mode = "linear", coordinate_transformation_mode = "asymmetric",
         nearest_mode = "floor"}
        : (tensor<1x512x6x6xf16>, tensor<0xf32>, tensor<0xf32>, tensor<4xi64>)
        -> tensor<1x512x12x12xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 1
    return %y : tensor<1x512x12x12xf16>
  }

  // Test 16: pytorch_half_pixel is its own coord id (3), not an alias of
  // half_pixel. An output axis of length 1 samples input coordinate 0.
  func.func @test_resize_pytorch_half_pixel(%arg0: tensor<1x3x16x16xf16>,
                                             %scales: tensor<4xf32>)
      -> tensor<1x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_pytorch_half_pixel
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear",
         coordinate_transformation_mode = "pytorch_half_pixel"}
        : (tensor<1x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<1x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 3
    // CHECK-SAME: mode = 1
    return %y : tensor<1x3x32x32xf16>
  }

  // Test 17: floor(3 * 1.5) = 4, so out/in is 4/3, not 1.5. The ONNX
  // scales are kept on the op for the coordinate transform.
  func.func @test_resize_scales_kept(%arg0: tensor<1x1x3xf32>)
      -> tensor<1x1x4xf32> {
    // CHECK-LABEL: func.func @test_resize_scales_kept
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 1.5]> : tensor<3xf32>}
        : () -> tensor<3xf32>
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x1x3xf32>, none, tensor<3xf32>) -> tensor<1x1x4xf32>
    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: scales = [1.000000e+00, 1.000000e+00, 1.500000e+00]
    return %y : tensor<1x1x4xf32>
  }

  // Test 18: scales that match out/in, and align_corners, need no scales.
  func.func @test_resize_scales_dropped(%arg0: tensor<1x1x3xf32>,
                                        %arg1: tensor<1x1x3xf32>)
      -> (tensor<1x1x6xf32>, tensor<1x1x4xf32>) {
    // CHECK-LABEL: func.func @test_resize_scales_dropped
    %roi = "onnx.NoValue"() {value} : () -> none
    %s2 = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0]> : tensor<3xf32>}
        : () -> tensor<3xf32>
    %s15 = "onnx.Constant"() {value = dense<[1.0, 1.0, 1.5]> : tensor<3xf32>}
        : () -> tensor<3xf32>
    %y0 = "onnx.Resize"(%arg0, %roi, %s2)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x1x3xf32>, none, tensor<3xf32>) -> tensor<1x1x6xf32>
    %y1 = "onnx.Resize"(%arg1, %roi, %s15)
        {mode = "linear", coordinate_transformation_mode = "align_corners"}
        : (tensor<1x1x3xf32>, none, tensor<3xf32>) -> tensor<1x1x4xf32>
    // CHECK: hip.resize
    // CHECK-NOT: scales
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 2
    // CHECK-NOT: scales
    // CHECK: return
    return %y0, %y1 : tensor<1x1x6xf32>, tensor<1x1x4xf32>
  }
}
