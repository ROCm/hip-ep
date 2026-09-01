//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_MinOp : Hipsr_DpsOp<"min"> {
  let summary = "Elementwise min, destination-passing style";
  let description = [{
    Elementwise minimum with ONNX/NumPy broadcasting:
    `init = minimum(lhs, rhs)`.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$lhs,
    Hipsr_DeviceTensorOrMemRef:$rhs,
    Hipsr_DeviceTensorOrMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}
