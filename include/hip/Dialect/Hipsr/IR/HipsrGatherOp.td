//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_GatherOp : Hipsr_DpsOp<"gather"> {
  let summary = "Gather slices along an axis, destination-passing style";
  let description = [{
    Selects entries of `data` along `axis` at the positions in `indices`, so
    the output shape is `data`'s with `axis` replaced by `indices`' shape.

    `axis` is non-negative here, unlike ONNX Gather where it may count back
    from the end.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$data,
    Hipsr_TensorOrDeviceMemRef:$indices,
    Hipsr_TensorOrDeviceMemRef:$init,
    I64Attr:$axis
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $data `,` $indices `:` type($data) `,` type($indices) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
