//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.Return op. No ONNX schema describes it: it carries graph structure
// rather than an ONNX computation. The subgraph terminator is onnx.Yield, in
// OnnxYieldOp.td.
//

//===----------------------------------------------------------------------===//
// Onnx_ReturnOp
//===----------------------------------------------------------------------===//

def Onnx_ReturnOp : Onnx_Op<"Return",
    [Pure, ReturnLike, Terminator, HasParent<"func::FuncOp">]> {
  let summary = "Terminates the function holding an ONNX graph";
  let description = [{
    Returns the graph's outputs. Operands are unconstrained because this
    operation outlives the type conversion that rewrites the values it returns;
    the enclosing function's signature is what pins their types.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}
