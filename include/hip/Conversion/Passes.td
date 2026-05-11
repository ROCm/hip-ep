//===-- Passes.td - HIP conversion pass definitions --------*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Umbrella TableGen file for the HIP conversion passes (ONNX -> HIP, HIP ->
// LLVM, and the ONNX -> HipDNN graph outliner). Mirrors the umbrella pattern
// used by upstream `mlir/Conversion/Passes.td` and `torch-mlir/Conversion/
// Passes.td`: a single `.td` per conversion subsystem keeps registration,
// option declarations, and dependent-dialect lists in one place.
//
// Out of scope: the handle-bound `CompileHipDNNGraphsPass` is hand-rolled.
// Its constructor takes a runtime `hipdnnHandle_t` and a `CompiledGraphMap`
// out-parameter, neither of which can be expressed as a TableGen `Option`.
// See `include/hip/Conversion/OnnxToHipDNN/Passes.h` for that factory.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_PASSES
#define HIP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertOnnxToHipPass : Pass<"convert-onnx-to-hip", "mlir::ModuleOp"> {
  let summary = "Convert ONNX dialect to HIP dialect (tensor DPS)";
  let description = [{
    Converts ONNX dialect IR (from onnx-mlir) into HIP dialect IR using
    destination-passing style (DPS) with tensor types.

    Transformations:
      1. Constants: onnx.Constant -> arith.constant (small/inline) or
         memref.global extern + memref.get_global + bufferization.to_tensor
         (large/externalized to sidecar .constants.bin + .constants.json)
      2. Op mapping: onnx.{MatMul,Transpose,Mul,Softmax} -> hip.* (tensor DPS)
      3. Cleanup: erase onnx.EntryPoint

    Requires hip-add-context-arg to have run first so that !hip.context is
    available as function argument 0.

    Constant externalization (opt-in via --externalize-min-num-elements):
      Large constants above the threshold are written to a binary sidecar file
      and replaced with extern memref.global ops carrying a hip.external_data
      attribute (offset + size).  The filename is recorded as a
      hip.constants_file attribute on the module.  This follows the onnx-mlir
      pattern (krnl.global + omMMapBinaryFile) adapted to standard MLIR ops.
      Splat constants are always kept inline regardless of size.
  }];
  let options = [
    Option<"externalizeOutputDir", "externalize-output-dir", "std::string",
           /*default=*/"\"\"",
           "Directory for sidecar .constants.bin/.json files (empty = cwd)">,
    Option<"externalizeMinNumElements", "externalize-min-num-elements",
           "int64_t", /*default=*/"0",
           "Minimum number of tensor elements to externalize (0 = disabled)">
  ];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::bufferization::BufferizationDialect"
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

def OutlineOnnxToHipDNNPass : Pass<"outline-onnx-to-hipdnn", "mlir::ModuleOp"> {
  let summary = "Outline supported ONNX ops into hip.hipdnn_graph_outline "
                "regions";
  let description = [{
    Walks every ONNX op in the module.  For each op that is supported by
    hipDNN graph compilation AND has fully static tensor shapes, wraps it
    in a hip.hipdnn_graph_outline region.  Unsupported or dynamic-shape ops
    are left unchanged for the standard ConvertOnnxToHip path to handle.

    This pass is the handle-free first half of the hipDNN graph compilation
    flow.  The second half (CompileHipDNNGraphsPass) needs a live
    hipdnnHandle_t and is therefore declared in OnnxToHipDNN/Passes.h
    rather than here.
  }];
  let dependentDialects = [
    "mlir::hip::HipDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // HIP_CONVERSION_PASSES
