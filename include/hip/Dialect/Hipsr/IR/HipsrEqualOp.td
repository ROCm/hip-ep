//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_EqualOp : Hipsr_DpsOp<"equal"> {
  let summary = "Elementwise equality, destination-passing style";
  let description = [{
    Elementwise equality with ONNX/NumPy broadcasting: `init = lhs == rhs`.

    Dimensions are right-aligned and each aligned pair must be equal or contain
    a one. The result is a mask, so its element type is a boolean rather than
    the operands' element type.
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
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
