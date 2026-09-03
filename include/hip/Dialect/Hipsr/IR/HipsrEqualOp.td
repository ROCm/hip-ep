//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_EqualOp : Hipsr_DpsOp<"equal",
    [AllElementTypesMatch<["lhs", "rhs"]>]> {
  let summary = "Elementwise equality, destination-passing style";
  let description = [{
    Elementwise equality with ONNX broadcasting: `init = lhs == rhs`.

    Shapes are right-aligned, and each aligned pair must be equal or contain a
    one. `init` holds `i1` rather than the operand element type, because a
    comparison yields a bool whatever it compared. The runtime stores that bool
    one byte per element.
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

  let hasVerifier = 1;
}
