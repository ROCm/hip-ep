//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_MulOp : Hipsr_DpsOp<"mul"> {
  let summary = "Elementwise mul, destination-passing style";
  let description = [{
    Elementwise multiplication with ONNX/NumPy broadcasting: `init = lhs * rhs`.
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
}
