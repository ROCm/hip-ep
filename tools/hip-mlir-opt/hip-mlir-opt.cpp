/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CrashHandler.h"
#include "hip/Compiler/PluginRegistry.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/IR/HipBufferize.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "mlir/Dialect/Shape/IR/Shape.h"
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
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::hipsr::HipsrDialect>();
  mlir::hipsr::registerConvertHipsrToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
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

  // Registers every nameable HIP / pipeline / standard-MLIR pass the tool and
  // the EP share. Defined once (InitAllPasses.h) so the two never drift; see
  // that function for the set and docs/pipeline_pass_menu.md for the catalogue.
  hip::compiler::registerAllPasses();

  // Tool-only extras: the standalone LLVM-lowering conversion passes. The
  // production pipeline reaches LLVM through `convert-hip-to-llvm` (which
  // populates these patterns internally), so the EP path does not register
  // them as separate names; they exist here purely so LIT tests can exercise
  // each conversion in isolation.
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertToLLVMPass();

  // Run every statically-linked plugin's registration before MlirOptMain
  // parses the command line, so `--<plugin-pass>` is recognised by the CL
  // parser and `--hipdnn-pipeline` sees plugin slot requests when it builds
  // its pass manager. No-op when no plugins were selected at configure time
  // (HIPDNN_EP_COMPILER_PLUGINS empty). See cmake/HipEpPlugins.cmake.
  hip::compiler::dispatchPluginRegistrationsOnce();

  // Companion to dispatchPluginRegistrationsOnce() above: that registers plugin
  // passes, this registers plugin dialects (custom ops + their bufferization
  // and HIP->LLVM-lowering interface models) into the tool's registry,
  // mirroring hip::compiler::loadAllDialects so the op set never drifts between
  // the tool and the EP. Without it convert-hip-to-llvm finds no lowering for
  // plugin ops and they survive unlowered. No-op when no plugins are selected.
  for (auto registerFn : hip::compiler::pluginDialectRegistrations())
    registerFn(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
