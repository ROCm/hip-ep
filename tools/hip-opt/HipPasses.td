/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===-- HipPasses.td - HIP pass definitions ----------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_PASSES
#define HIP_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOnnxToHipPass : Pass<"convert-onnx-to-hip", "mlir::ModuleOp"> {
  let summary = "Convert ONNX dialect to HIP dialect (tensor DPS)";
  let description = [{
    Converts ONNX dialect IR (from onnx-mlir) into HIP dialect IR using
    destination-passing style (DPS) with tensor types.

    Transformations:
      1. Weight extraction: onnx.Constant -> new func arguments
      2. Op mapping: onnx.{MatMul,Transpose,Mul,Softmax} -> hip.* (tensor DPS)
      3. Lifecycle: insert hip.create_handle / hip.destroy_handle
      4. Cleanup: erase onnx.EntryPoint
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def ConvertHipToLLVMPass : Pass<"convert-hip-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HIP dialect to LLVM dialect";
  let description = [{
    Lowers HIP dialect operations to LLVM dialect by converting each HIP op
    into calls to the HIP runtime C API (hipCreateHandle, hipblasltMatmul,
    etc.). MemRef operands are unpacked into bare pointers for the C calls.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // HIP_PASSES
