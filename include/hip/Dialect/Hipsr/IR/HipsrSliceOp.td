//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_SliceOp : Hipsr_DpsOp<"slice"> {
  let summary = "Extract a strided window along a set of axes, "
                "destination-passing style";
  let description = [{
    Takes `data[starts[k] : ends[k] : steps[k]]` along each axis in `axes` and
    keeps every other axis whole, so output axis `axes[k]` holds
    `ceil((ends[k] - starts[k]) / steps[k])` entries.

    The window is already resolved against the data extent, unlike ONNX Slice
    where `starts` and `ends` may be negative or out of bounds: each entry is
    non-negative, ordered, and within the axis, which is why a sliced axis must
    be statically sized. `steps` are positive; reversing an axis is a separate
    operation.

    `axes` is explicit and normalized here, and `steps` is always present,
    unlike ONNX Slice where both are optional.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$data,
    Hipsr_TensorOrDeviceMemRef:$init,
    DenseI64ArrayAttr:$starts,
    DenseI64ArrayAttr:$ends,
    DenseI64ArrayAttr:$axes,
    DenseI64ArrayAttr:$steps
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $data `:` type($data) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
