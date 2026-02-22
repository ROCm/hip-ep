/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "HipDialect.h"
#include "HipPasses.h"

// Include ONNX dialect from onnx-mlir
#include "src/Dialect/ONNX/ONNXDialect.hpp"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::hip::HipDialect>();
  registry.insert<mlir::ONNXDialect>(); // Register ONNX dialect for ONNX→HIP
                                        // lowering

  // Register HIP passes
  mlir::hip::registerHipPasses();

  // Register MLIR standard buffer deallocation passes
  mlir::bufferization::registerBufferizationPasses();

  // Register standard transforms (including canonicalize)
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-opt: custom compiler driver\n", registry));
}
