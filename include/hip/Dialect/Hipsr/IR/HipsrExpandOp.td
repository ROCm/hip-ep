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
    [DeclareOpInterfaceMethods<StartBarrierInterface, ["isStartBarrier"]>]> {
  let summary = "Broadcast a tensor to a requested shape";
  let description = [{
    Broadcasts `input` and the requested `shape` using ONNX multidirectional
    broadcasting. Dimensions are right-aligned. Each aligned pair must be equal
    or contain a one. The output rank is the greater of the input rank and the
    requested shape length.

    The requested shape is host-resident because its values determine the
    output allocation. Destination-passing style passes that allocation as
    `init`.
  }];

  let arguments = (ins
    Hipsr_ContextType:$ctx,
    Hipsr_TensorOrDeviceMemRef:$input,
    Hipsr_TensorOrHostMemRef:$shape,
    Hipsr_TensorOrDeviceMemRef:$init
  );

  let assemblyFormat = [{
    `(` $ctx `)`
    `ins` `(` $input `,` $shape `:` type($input) `,` type($shape) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict (`:` type($result_tensors)^)? (`shape_region` $shape_region^)?
  }];

  let hasVerifier = 1;
}
