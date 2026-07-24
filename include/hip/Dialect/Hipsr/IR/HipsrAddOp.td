//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_ADD_OP
#define HIPSR_ADD_OP

include "hip/Dialect/Hipsr/IR/HipsrOps.td"

def Hipsr_AddOp : Hipsr_DpsOp<"add"> {
  let summary = "Elementwise add, destination-passing style";
  let description = [{
    Elementwise addition with ONNX/NumPy broadcasting: `init = lhs + rhs`.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$lhs,
    Hipsr_TensorOrDeviceMemRef:$rhs,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)? (`shape_region` $shape_region^)?
  }];
}

#endif // HIPSR_ADD_OP
