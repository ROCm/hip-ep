/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"

namespace hip::compiler {

void registerConversionPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertOnnxToHipPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOutlineOnnxToHipDNNPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createConvertHipToLLVMPass();
  });

  mlir::hipsr::registerHipsrConversionPasses();
}

} // namespace hip::compiler
