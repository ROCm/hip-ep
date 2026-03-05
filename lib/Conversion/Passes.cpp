/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"

namespace hip::compiler {

void registerConversionPasses() {
  // Register context-arg insertion pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createHipAddContextArgPass();
  });

  // Register ONNX to HIP conversion pass (no-arg: falls back to DiskFileSystem
  // in current dir when model has constants — troubleshooting only)
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertOnnxToHipPass();
  });

  // Register HIP to LLVM conversion pass
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertHipToLLVMPass();
  });
}

} // namespace hip::compiler
