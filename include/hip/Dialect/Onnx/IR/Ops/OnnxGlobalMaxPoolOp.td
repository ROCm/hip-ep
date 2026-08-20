//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GlobalMaxPoolOp
//===----------------------------------------------------------------------===//

def Onnx_GlobalMaxPoolOp : Onnx_Op<"GlobalMaxPool", [Pure]> {
  let summary = "ONNX GlobalMaxPool, schema version 22";

  let description = [{
     GlobalMaxPool consumes an input tensor X and applies max pooling across
     the values in the same channel. This is equivalent to MaxPool with kernel size
     equal to the spatial dimension of input tensor.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
