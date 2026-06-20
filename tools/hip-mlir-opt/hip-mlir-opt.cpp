/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CrashHandler.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/InitAllPasses.h"

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-mlir-opt");
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<hip::compiler::detail::OnnxStubDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  // Shared with the EP path (hip::compiler::registerAllDialects); defined once
  // in HipBufferize.h so the HIP op bufferization models never drift between
  // the tool and the EP.
  mlir::hip::registerHipBufferizableOpInterfaceModels(registry);

  mlir::hip::registerHipPasses();
  mlir::hip::registerHipPipelines();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOutlineOnnxToHipDNNPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOnnxLoopOutlinePass();
  });
  mlir::bufferization::registerBufferizationPasses();
  mlir::bufferization::registerBufferizationPipelines();
  mlir::registerConvertBufferizationToMemRefPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  // Registered so that LIT tests and end-to-end pipelines can fold
  // `tensor.dim` / `memref.dim` of HIP op results through the reify
  // implementation. Used in `hip-matmul-reify-shapes.mlir`.
  mlir::memref::registerResolveShapedTypeResultDimsPass();
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return mlir::createCSEPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createCanonicalizerPass();
  });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
