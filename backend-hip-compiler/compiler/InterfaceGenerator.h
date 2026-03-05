/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_INTERFACE_GENERATOR_H
#define HIPDNN_INTERFACE_GENERATOR_H

#include "HipCompiler.h"
#include "llvm/IR/Module.h"

namespace hipdnn::compiler {

/// Adds inference_init, inference_compute, inference_cleanup wrapper functions
/// to an LLVM IR module, bridging the span_t/tensor_t C ABI to the flattened
/// memref descriptor calling convention of the compiled compute function.
class InterfaceGenerator {
public:
  static void addInterfaceFunctions(llvm::Module &module,
                                    const ModelMetadata &metadata);
};

} // namespace hipdnn::compiler

#endif // HIPDNN_INTERFACE_GENERATOR_H
