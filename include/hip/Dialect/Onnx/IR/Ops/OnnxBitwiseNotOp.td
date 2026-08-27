//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_BitwiseNotOp
//===----------------------------------------------------------------------===//

def Onnx_BitwiseNotOp : Onnx_Op<"BitwiseNot", [Pure]> {
  let summary = "ONNX BitwiseNot, schema version 18";

  let description = [{
    Returns the bitwise not of the input tensor element-wise.
  }];

  let arguments = (ins
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>
    ]>:$X
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>
    ]>:$Y
  );
}
