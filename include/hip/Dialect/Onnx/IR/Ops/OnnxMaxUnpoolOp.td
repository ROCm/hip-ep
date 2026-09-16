//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_MaxUnpoolOp
//===----------------------------------------------------------------------===//

def Onnx_MaxUnpoolOp : Onnx_Op<"MaxUnpool", [Pure]> {
  let summary = "ONNX MaxUnpool, schema version 22";

  let description = [{
    MaxUnpool essentially computes the partial inverse of the MaxPool op.
     The input information to this op is typically the output information from a MaxPool op. The first
     input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
     from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
     to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
     The third (optional) input is a tensor that specifies the output size of the unpooling operation.

    MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
     values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
     the result of an unpooling operation should give back the original input to the unpooling op.

    MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
     The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
     known/predictable size.

    In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
     which define the exact unpooling op. The attributes typically have the same values as the corresponding
     pooling op that the unpooling op is trying to invert.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    TensorOf<[I64]>:$I,
    AnyTypeOf<[TensorOf<[I64]>, NoneType]>:$output_shape,
    I64ArrayAttr:$kernel_shape,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$output
  );
}
