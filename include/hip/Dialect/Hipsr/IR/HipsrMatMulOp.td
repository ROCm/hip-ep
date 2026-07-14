//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.matmul op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

#ifndef HIPSR_MATMUL_OP
#define HIPSR_MATMUL_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"

//===----------------------------------------------------------------------===//
// Hipsr_MatMulOp
//===----------------------------------------------------------------------===//

def Hipsr_MatMulOp : Hipsr_DpsOp<"matmul"> {
  let summary = "Matrix multiply (row-major), destination-passing style";
  let description = [{
    Computes `result = lhs * rhs` for 2-D operands: `lhs : [M, K]`,
    `rhs : [K, N]`, `result : [M, N]`. Destination-passing style: the result
    buffer is passed as the `init` operand.

    Regular op: carries a shape region (region 0) computing `[M, N]` from the
    input shapes (M = dim(lhs, 0), N = dim(rhs, 1)).

    Example:
    ```mlir
    %0 = hipsr.matmul ins(%lhs, %rhs : tensor<?x256xf16>, tensor<256x512xf16>)
                      outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16> shape_region {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %m = tensor.dim %lhs, %c0 : tensor<?x256xf16>
      %n = tensor.dim %rhs, %c1 : tensor<256x512xf16>
      hipsr.shape_yield %m, %n
    }
    ```
  }];

  let arguments = (ins
    Hipsr_TensorOrDeviceMemRef:$lhs,
    Hipsr_TensorOrDeviceMemRef:$rhs,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict `->` type($result_tensors) `shape_region` $shape_region
  }];
}

#endif // HIPSR_MATMUL_OP
