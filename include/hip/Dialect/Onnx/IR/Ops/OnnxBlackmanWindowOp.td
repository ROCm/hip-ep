//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_BlackmanWindowOp
//===----------------------------------------------------------------------===//

def Onnx_BlackmanWindowOp : Onnx_Op<"BlackmanWindow", [Pure]> {
  let summary = "ONNX BlackmanWindow, schema version 17";

  let description = [{
    Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.
  }];

  let arguments = (ins
    AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I64]>]>:$size,
    DefaultValuedAttr<SI64Attr, "1">:$output_datatype,
    DefaultValuedAttr<SI64Attr, "1">:$periodic
  );

  let results = (outs
    AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>
    ]>:$output
  );
}
