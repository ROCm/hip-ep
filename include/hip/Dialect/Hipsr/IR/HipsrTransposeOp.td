//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_TransposeOp : Hipsr_DpsOp<"transpose"> {
  let summary = "Permute a tensor's axes, destination-passing style";
  let description = [{
    Permutes `input`'s axes into `init`: output axis `i` is input axis
    `perm[i]`.

    `perm` is always explicit here, unlike ONNX Transpose where omitting it
    means the reverse permutation.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$input,
    Hipsr_TensorOrDeviceMemRef:$init,
    DenseI64ArrayAttr:$perm
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
