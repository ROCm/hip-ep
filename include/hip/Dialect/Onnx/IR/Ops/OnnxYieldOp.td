//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.Yield op: the terminator of the subgraph regions onnx.If, onnx.Loop
// and onnx.Scan carry. No ONNX schema describes it, since a subgraph's outputs
// are named in the GraphProto rather than by an operation.
//

//===----------------------------------------------------------------------===//
// Onnx_YieldOp
//===----------------------------------------------------------------------===//

def Onnx_YieldOp : Onnx_Op<"Yield", [Pure, ReturnLike, Terminator]> {
  let summary = "Terminates a subgraph region";
  let description = [{
    Yields a subgraph's outputs to the operation that owns the region, such as
    an `onnx.Loop` or `onnx.If`. Unlike `onnx.Return` it names no parent: the
    operations that carry subgraph regions are not all modeled, so the parent
    may be an unknown operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}
