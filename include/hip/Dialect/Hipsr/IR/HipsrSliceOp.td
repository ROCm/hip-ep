//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

def Hipsr_SliceOp : Hipsr_DpsOp<"slice", [
    AttrSizedOperandSegments,
    AllElementTypesMatch<["data", "init"]>,
    AllRanksMatch<["data", "init"]>]> {
  let summary = "Extract a strided window along a set of axes, "
                "destination-passing style";
  let description = [{
    Takes `data[starts[k] : ends[k] : steps[k]]` along axis `axes[k]` and keeps
    every other axis whole, so the output has the data's rank and no axis wider
    than the data's.

    Every window slot has exactly one of two forms: an attribute holding
    entries the compiler knows, or a host operand holding entries the graph
    computes. None is left out, so ONNX Slice's defaults for `axes` and `steps`
    are written out where a graph omits them. Each slot holds one entry per
    sliced axis, and how many axes narrow is always known here: only the
    entries themselves may wait for run time.

    Entries follow ONNX Slice: an axis or a bound may be negative and a bound
    may fall outside the axis, and each is resolved against the data's own
    shape. An operand lives on the host, where both the output shape and the
    runtime call read it.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_DeviceTensorOrMemRef:$data,
    Optional<Hipsr_HostIndexVector>:$starts,
    Optional<Hipsr_HostIndexVector>:$ends,
    Optional<Hipsr_HostIndexVector>:$axes,
    Optional<Hipsr_HostIndexVector>:$steps,
    Hipsr_DeviceTensorOrMemRef:$init,
    OptionalAttr<DenseI64ArrayAttr>:$starts_attr,
    OptionalAttr<DenseI64ArrayAttr>:$ends_attr,
    OptionalAttr<DenseI64ArrayAttr>:$axes_attr,
    OptionalAttr<DenseI64ArrayAttr>:$steps_attr
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $data `:` type($data) `)`
    (`starts` `(` $starts^ `:` type($starts) `)`)?
    (`ends` `(` $ends^ `:` type($ends) `)`)?
    (`axes` `(` $axes^ `:` type($axes) `)`)?
    (`steps` `(` $steps^ `:` type($steps) `)`)?
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasVerifier = 1;
}
