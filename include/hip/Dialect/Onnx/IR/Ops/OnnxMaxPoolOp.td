//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MaxPoolOp
//===----------------------------------------------------------------------===//

def Onnx_MaxPoolOp : Onnx_Op<"MaxPool", [Pure]> {
  let summary = "ONNX MaxPool, schema version 22";

  let description = [{
     MaxPool consumes an input tensor X and applies max pooling across
     the tensor according to kernel sizes, stride sizes, and pad lengths.
     max pooling consisting of computing the max on all values of a
     subset of the input tensor according to the kernel size and downsampling the
     data into the output tensor Y for further processing. The output spatial shape is calculated differently
     depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
     With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
     ```
     output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
     ```
     or
     ```
     output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
     ```
     if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

     `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
     ```
     VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
     SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
     ```
     or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
     ```
     VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
     SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
     ```
     And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
     ```
     pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
     ```
     The output of each pooling window is maximum number of elements exclude pad.

  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I8]>, TensorOf<[UI8]>
    ]>:$X,
    DefaultValuedStrAttr<StrAttr, "NOTSET">:$auto_pad,
    DefaultValuedAttr<SI64Attr, "0">:$ceil_mode,
    OptionalAttr<I64ArrayAttr>:$dilations,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    DefaultValuedAttr<SI64Attr, "0">:$storage_order,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I8]>, TensorOf<[UI8]>
    ]>:$Y,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$Indices
  );
}
