/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_TEST_LIB_DIALECT_HIP_TEST_HIP_PASSES_H
#define HIP_TEST_LIB_DIALECT_HIP_TEST_HIP_PASSES_H

namespace mlir {
class DialectRegistry;

namespace hip::test {

/// Register HIP test-only operations, interfaces, and passes.
void registerHipTestPasses(DialectRegistry &registry);

} // namespace hip::test
} // namespace mlir

#endif // HIP_TEST_LIB_DIALECT_HIP_TEST_HIP_PASSES_H
