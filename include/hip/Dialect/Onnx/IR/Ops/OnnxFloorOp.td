//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_FloorOp
//===----------------------------------------------------------------------===//

def Onnx_FloorOp : Onnx_Op<"Floor", [Pure]> {
  let summary = "ONNX Floor, schema version 13";

  let description = [{
    Floor takes one input data (Tensor<T>) and produces one output data
    (Tensor<T>) where the floor is, y = floor(x), is applied to
    the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$Y
  );
}
