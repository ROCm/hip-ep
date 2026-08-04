//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The hipsr.expand op. Each compute op lives in its own file and builds on the
// Hipsr_DpsOp base class from HipsrOps.td.
//

//===----------------------------------------------------------------------===//
// Hipsr_ExpandOp
//===----------------------------------------------------------------------===//

def Hipsr_ExpandOp : Hipsr_DpsOp<
    "expand",
    [DeclareOpInterfaceMethods<StartBarrierInterface, ["isStartBarrier"]>,
     DeclareOpInterfaceMethods<ShapeRegionInterface,
                               ["populateShapeRegion"]>]> {
  let summary = "Broadcast a tensor to a requested shape";
  let description = [{
    Broadcasts `input` and the requested shape using ONNX multidirectional
    broadcasting. Dimensions are right-aligned. Each aligned pair must be equal
    or contain a one. The output rank is the greater of the input rank and the
    requested shape length.

    Exactly one shape representation is present: `shape_attr` carries
    compile-time extents, while the host-resident `shape` operand carries
    runtime extents. Destination-passing style passes the output allocation as
    `init`.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$input,
    Optional<Hipsr_TensorOrHostMemRef>:$shape,
    Hipsr_TensorOrDeviceMemRef:$init,
    OptionalAttr<DenseI64ArrayAttr>:$shape_attr
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input (`,` $shape^)? `:` type($input)
        (`,` type($shape)^)? `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)?
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}
