//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.cast op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

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
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$input,
    Hipsr_DeviceTensorOrMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];
}
