//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_NotOp
//===----------------------------------------------------------------------===//

def Onnx_NotOp : Onnx_Op<"Not", [Pure]> {
  let summary = "ONNX Not, schema version 1";

  let description = [{
    Returns the negation of the input tensor element-wise.
  }];

  let arguments = (ins
    TensorOf<[I1]>:$X
  );

  let results = (outs
    TensorOf<[I1]>:$Y
  );
}
