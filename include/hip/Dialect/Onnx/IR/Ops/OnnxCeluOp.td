//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Onnx_CeluOp
//===----------------------------------------------------------------------===//

def Onnx_CeluOp : Onnx_Op<"Celu", [Pure]> {
  let summary = "ONNX Celu, schema version 12";

  let description = [{
    Continuously Differentiable Exponential Linear Units:
    Perform the linear unit element-wise on the input tensor X
    using formula:

    ```
    max(0,x) + min(0,alpha*(exp(x/alpha)-1))
    ```
  }];

  let arguments = (ins
    TensorOf<[F32]>:$X,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha
  );

  let results = (outs
    TensorOf<[F32]>:$Y
  );
}
