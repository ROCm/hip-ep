//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_GatherOp : Hipsr_DpsOp<"gather",
    [AllElementTypesMatch<["data", "init"]>]> {
  let summary = "Gather slices along an axis, destination-passing style";
  let description = [{
    Selects entries of `data` along `axis` at the positions in `indices`, so
    the output shape is `data`'s with `axis` replaced by `indices`' shape.
    Rank-0 indices therefore drop the axis.

    `axis` is non-negative here, unlike ONNX Gather where it may count back
    from the end.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$data,
    Hipsr_DeviceTensorOrMemRef:$indices,
    Hipsr_DeviceTensorOrMemRef:$init,
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
