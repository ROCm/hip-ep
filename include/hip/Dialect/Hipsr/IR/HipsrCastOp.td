//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.cast op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

#ifndef HIPSR_CAST_OP
#define HIPSR_CAST_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_CastOp
//===----------------------------------------------------------------------===//

def Hipsr_CastOp : Hipsr_DpsOp<"cast"> {
  let summary = "Elementwise type cast, destination-passing style";
  let description = [{
    Casts `input` to the element type of the result (ONNX Cast). The output
    has the same shape as the input; only the element type changes.
    Destination-passing style: the result buffer is passed as the `init`
    operand.

    Regular op: carries a shape region (region 0) that copies the input shape
    (dim(input, i) for each i).

    Example:
    ```mlir
    %0 = hipsr.cast ins(%input : tensor<?x8xf32>) outs(%init : tensor<?x8xf16>)
                    -> tensor<?x8xf16> shape_region {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
      hipsr.shape_yield %d0, %c8
    }
    ```
  }];

  let arguments = (ins
    Hipsr_TensorOrDeviceMemRef:$input,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict `->` type($result_tensors) `shape_region` $shape_region
  }];
}

#endif // HIPSR_CAST_OP
