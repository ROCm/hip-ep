//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_Col2ImOp
//===----------------------------------------------------------------------===//

def Onnx_Col2ImOp : Onnx_Op<"Col2Im", [Pure]> {
  let summary = "ONNX Col2Im, schema version 18";

  let description = [{
    The operator rearranges column blocks back into a multidimensional image

    Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
    but it only supports *batched* multi-dimensional image tensors.
    Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

    NOTE:
      Although specifying image_shape looks redundant because it could be calculated from
      convolution formulas, it is required as input for more advanced scenarios as explained
      at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$input,
    TensorOf<[I64]>:$image_shape,
    TensorOf<[I64]>:$block_shape,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<I64ArrayAttr>:$pads,
    OptionalAttr<I64ArrayAttr>:$strides
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>
    ]>:$output
  );
}
