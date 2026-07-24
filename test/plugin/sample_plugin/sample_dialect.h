/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Registration entry point for the sample plugin's minimal vendor dialect. In
// its own TU so sample_plugin.cpp needs no MLIR dialect/conversion headers.

#ifndef HIP_EP_TEST_SAMPLE_DIALECT_H
#define HIP_EP_TEST_SAMPLE_DIALECT_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

/// Insert the `hip_ep_sample` dialect and attach its
/// ConvertToLLVMPatternInterface. Passed to
/// HipEpPluginRegistry::addDialectRegistration (a plain function converts to
/// the required pointer). See docs/design/plugin-interface.md.
void registerHipEpSampleDialect(mlir::DialectRegistry &registry);

#endif // HIP_EP_TEST_SAMPLE_DIALECT_H
