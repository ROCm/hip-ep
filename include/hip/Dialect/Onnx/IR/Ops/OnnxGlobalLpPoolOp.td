//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_GlobalLpPoolOp
//===----------------------------------------------------------------------===//

def Onnx_GlobalLpPoolOp : Onnx_Op<"GlobalLpPool", [Pure]> {
  let summary = "ONNX GlobalLpPool, schema version 22";

  let description = [{
     GlobalLpPool consumes an input tensor X and applies lp pool pooling across
     the values in the same channel. This is equivalent to LpPool with kernel size
     equal to the spatial dimension of input tensor.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$X,
    DefaultValuedAttr<SI64Attr, "2">:$p
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>
    ]>:$Y
  );
}
