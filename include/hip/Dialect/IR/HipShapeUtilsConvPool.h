/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_CONV_POOL_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_CONV_POOL_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

/// Compute the ONNX forward-convolution result shape for rank-3 NCL or rank-4
/// NCHW operands. The result is `{input[0], weights[0], spatial...}` and each
/// spatial extent follows the signed-floor window formula.
///
/// An empty `kernelShape` means the ONNX attribute was omitted. In that case
/// the kernel is derived from the static spatial dimensions of `weightShape`;
/// dynamic weight spatial dimensions require an explicit kernel.
FailureOr<SmallVector<int64_t>>
inferConvShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
               ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
               ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
               int64_t group, function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvShape`. Validates through the static helper
/// before materializing any dimension arithmetic. Dynamic spatial extents are
/// safely narrowed from signed i128; `runtimeValid`, when requested, receives
/// the combined range check consumed by `hip.conv`.
FailureOr<SmallVector<OpFoldResult>>
reifyConvResultShape(OpBuilder &b, Location loc, Value input, Value weights,
                     ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
                     int64_t group,
                     function_ref<InFlightDiagnostic()> emitError,
                     Value *runtimeValid = nullptr);

/// Compute an ONNX MaxPool/AveragePool/LpPool result shape for spatial rank
/// 1..3. N/C pass through from the input; spatial extents use signed floor
/// division. When `ceilMode` is 1, signed ceil division is followed by ONNX's
/// trailing-window correction: decrement a positive output extent when its
/// final window starts at or beyond `input + pad_begin`.
FailureOr<SmallVector<int64_t>>
inferPoolShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> kernelShape,
               ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
               ArrayRef<int64_t> dilations, int64_t ceilMode,
               function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferPoolShape`. The returned shape is shared by the
/// values result and optional MaxPool indices result. Dynamic spatial
/// arithmetic is materialized in signed i128, range-checked, safely narrowed
/// to index, and combined into `runtimeValid` when that optional output is
/// requested.
FailureOr<SmallVector<OpFoldResult>>
reifyPoolResultShape(OpBuilder &b, Location loc, Value input,
                     ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
                     int64_t ceilMode,
                     function_ref<InFlightDiagnostic()> emitError,
                     Value *runtimeValid = nullptr);

/// Compute the supported rank-4 NCHW ONNX ConvTranspose result shape:
///   N = input[0]
///   M = weights[1] * group
///   spatial[i] = stride[i] * (input[i+2] - 1) + output_padding[i]
///                + ((kernel[i] - 1) * dilation[i] + 1)
///                - pads[i] - pads[i+2]
///
/// Dynamic input/weight dimensions remain dynamic. All attribute vectors are
/// validated before a result is produced.
FailureOr<SmallVector<int64_t>> inferConvTransposeShape(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> outputPadding, int64_t group,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvTransposeShape`. Validates through the static
/// helper before materializing any dimension arithmetic.
FailureOr<SmallVector<OpFoldResult>>
reifyConvTransposeResultShape(OpBuilder &b, Location loc, Value input,
                              Value weights, ArrayRef<int64_t> kernelShape,
                              ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
                              ArrayRef<int64_t> dilations,
                              ArrayRef<int64_t> outputPadding, int64_t group,
                              function_ref<InFlightDiagnostic()> emitError);

/// GlobalPool shape: preserve N/C and replace every spatial extent with 1.
FailureOr<SmallVector<int64_t>>
inferGlobalPoolShape(ArrayRef<int64_t> inputShape,
                     function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferGlobalPoolShape`.
FailureOr<SmallVector<OpFoldResult>>
reifyGlobalPoolResultShape(OpBuilder &b, Location loc, Value input,
                           function_ref<InFlightDiagnostic()> emitError);

/// Compute the supported spatial Resize result shape. The HIP op deliberately
/// does not carry ONNX `sizes` or `scales`: N/C therefore come from `input`,
/// while every spatial extent must already be static in `outputTemplate`.
///
/// A dynamic input N/C extent requires the corresponding template extent to
/// remain dynamic; a static template cannot promise equality to an unknown
/// runtime input extent. A static input N/C extent may refine a dynamic
/// template extent.
FailureOr<SmallVector<int64_t>>
inferResizeShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputTemplate,
                 function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferResizeShape`. Validation completes before any
/// `tensor.dim` is emitted. Dynamic N/C extents come from `input`; spatial
/// extents are constants from `outputTemplate`.
FailureOr<SmallVector<OpFoldResult>>
reifyResizeShape(OpBuilder &b, Location loc, Value input,
                 ArrayRef<int64_t> outputTemplate,
                 function_ref<InFlightDiagnostic()> emitError);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_CONV_POOL_H
