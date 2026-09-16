//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GlobalAveragePoolOp
//===----------------------------------------------------------------------===//

def Onnx_GlobalAveragePoolOp : Onnx_Op<"GlobalAveragePool", [Pure]> {
  let summary = "ONNX GlobalAveragePool, schema version 22";

  let description = [{
     GlobalAveragePool consumes an input tensor X and applies average pooling across
     the values in the same channel. This is equivalent to AveragePool with kernel size
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
