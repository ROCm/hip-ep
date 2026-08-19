//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The onnx.EntryPoint op. No ONNX schema describes it: it records where the
// model starts rather than an ONNX computation.
//

//===----------------------------------------------------------------------===//
// Onnx_EntryPointOp
//===----------------------------------------------------------------------===//

def Onnx_EntryPointOp : Onnx_Op<"EntryPoint"> {
  let summary = "Names the function that is the model's entry point";
  let description = [{
    Records which function a consumer should call to run the model. Not `Pure`:
    the operation has no results, so a pure operation carrying this information
    would be dead on arrival.
  }];

  let arguments = (ins SymbolRefAttr:$func);
}
