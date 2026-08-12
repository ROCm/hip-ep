//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_ScatterNDOp : Hipsr_DpsOp<"scatter_nd"> {
  let summary = "Overwrite indexed slices of a tensor, destination-passing "
                "style";
  let description = [{
    Copies `data` and overwrites the slice each row of `indices` addresses with
    the matching slice of `updates`, so the output has `data`'s shape.

    The trailing extent of `indices` says how many leading `data` axes a row
    addresses, which leaves `updates` shaped as the leading `indices` extents
    followed by the `data` extents no row reaches.

    Overwriting is the only mode, unlike ONNX ScatterND which also reduces a
    duplicate index into its destination.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$data,
    Hipsr_TensorOrDeviceMemRef:$indices,
    Hipsr_TensorOrDeviceMemRef:$updates,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $data `,` $indices `,` $updates `:` type($data) `,`
        type($indices) `,` type($updates) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
