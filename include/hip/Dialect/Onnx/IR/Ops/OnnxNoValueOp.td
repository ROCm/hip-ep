//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.NoValue op. No ONNX schema describes it: it carries graph structure
// rather than an ONNX computation.
//

//===----------------------------------------------------------------------===//
// Onnx_NoValueOp
//===----------------------------------------------------------------------===//

def Onnx_NoValueOp : Onnx_Op<"NoValue", [Pure]> {
  let summary = "Stands in for an operand the exporter omitted";
  let description = [{
    Materializes the absence of an optional operand, so an operation carries
    every operand position its schema declares whichever ones the exporter
    omitted. An omitted optional result needs no stand-in: it is simply typed
    `none`.

    Example:
    ```mlir
    %none = "onnx.NoValue"() <{value}> : () -> none
    ```
  }];

  let arguments = (ins UnitAttr:$value);
  let results = (outs NoneType:$none_val);
}
