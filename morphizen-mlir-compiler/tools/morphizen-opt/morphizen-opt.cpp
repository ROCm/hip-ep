/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "morphizen-mlir-compiler/Dialect/Hip/IR/HipDialect.h"
#include "morphizen-mlir-compiler/InitAllPasses.h"

// Include ONNX dialect from onnx-mlir
#include "src/Dialect/ONNX/ONNXDialect.hpp"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::ONNXDialect>();

  // Register Morphizen passes and pipelines
  // Note: We don't call mlir::registerAllPasses() because it requires linking
  // many additional MLIR libraries. The morphizen pipeline includes the
  // necessary MLIR passes (canonicalizer, bufferization, etc.) that we actually
  // use.
  morphizen::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Morphizen MLIR Pass Runner\n", registry));
}
