/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "morphizen-mlir-compiler/InitAllPasses.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  morphizen::registerAllDialects(registry);

  // Register Morphizen passes and pipelines
  // Note: We don't call mlir::registerAllPasses() because it requires linking
  // many additional MLIR libraries. We selectively register only the standard
  // passes we need (canonicalizer) plus our custom passes.
  morphizen::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Morphizen MLIR Pass Runner\n", registry));
}
