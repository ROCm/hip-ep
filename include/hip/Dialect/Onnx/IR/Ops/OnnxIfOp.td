//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_IfOp
//===----------------------------------------------------------------------===//

def Onnx_IfOp : Onnx_Op<"If", [RecursiveMemoryEffects]> {
  let summary = "ONNX If, schema version 25";

  let description = [{
    If conditional
  }];

  let arguments = (ins
    TensorOf<[I1]>:$cond
  );

  let results = (outs
    Variadic<AnyTypeOf<[
      TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>,
      TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>,
      TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>,
      TensorOf<[I1]>, TensorOf<[Complex<F32>]>, TensorOf<[Complex<F64>]>,
      TensorOf<[F8E4M3FN]>, TensorOf<[F8E4M3FNUZ]>, TensorOf<[F8E5M2]>,
      TensorOf<[F8E5M2FNUZ]>, TensorOf<[UI<4>]>, TensorOf<[I<4>]>,
      TensorOf<[F4E2M1FN]>, TensorOf<[F8E8M0FNU]>, TensorOf<[UI<2>]>,
      TensorOf<[I<2>]>
    ]>>:$outputs
  );

  let regions = (region AnyRegion:$then_branch, AnyRegion:$else_branch);
}
