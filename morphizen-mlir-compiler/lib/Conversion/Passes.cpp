/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-mlir-compiler/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"

namespace morphizen {

void registerConversionPasses() {
  // Register ONNX to HIP conversion pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertOnnxToHipPass();
  });

  // Register HIP to LLVM conversion pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertHipToLLVMPass();
  });
}

} // namespace morphizen
