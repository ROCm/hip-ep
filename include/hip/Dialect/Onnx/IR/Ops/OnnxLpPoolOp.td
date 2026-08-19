//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_LpPoolOp
//===----------------------------------------------------------------------===//

def Onnx_LpPoolOp : Onnx_Op<"LpPool", [Pure]> {
  let summary = "ONNX LpPool, schema version 22";

  let description = [{
     LpPool consumes an input tensor X and applies Lp pooling across
     the tensor according to kernel sizes, stride sizes, and pad lengths.
     Lp pooling consisting of computing the Lp norm on all values of a subset
     of the input tensor according to the kernel size and downsampling the
     data into the output tensor Y for further processing. The output spatial shape will be following:
     ```
     output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
     ```
     or
     ```
     output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
     ```
     if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

     `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
     ```
     VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
     SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
     ```
     And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
     ```
     pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
     ```
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
    DefaultValuedAttr<SI64Attr, "0">:$ceil_mode,
    OptionalAttr<I64ArrayAttr>:$dilations,
    I64ArrayAttr:$kernel_shape,
    DefaultValuedAttr<SI64Attr, "2">:$p,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
