//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_OrOp
//===----------------------------------------------------------------------===//

def Onnx_OrOp : Onnx_Op<"Or", [Pure]> {
  let summary = "ONNX Or, schema version 7";

  let description = [{
    Returns the tensor resulted from performing the `or` logical operation
    elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

    This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
  }];

  let arguments = (ins
    TensorOf<[I1]>:$A,
    TensorOf<[I1]>:$B
  );

  let results = (outs
    TensorOf<[I1]>:$C
  );
}
