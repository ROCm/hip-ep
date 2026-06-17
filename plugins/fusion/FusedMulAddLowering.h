/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PLUGIN_FUSION_LOWERING_H
#define HIP_PLUGIN_FUSION_LOWERING_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hip {

// Runtime symbol emitted by FusedMulAddOpLowering.
// Signature: int wrap_fused_mul_add(void* state,
//                                   void* x, void* mul_operand, void* add_operand,
//                                   void* output, int64_t num_elements,
//                                   int64_t data_type)
inline constexpr const char *kWrapFusedMulAdd = "wrap_fused_mul_add";

/// Populate HipToLLVM lowering patterns for the plugin op hip.fused_mul_add.
/// Called from PluginMain.cpp when the lowering extension is registered.
void populateFusedMulAddLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace hip
} // namespace mlir

#endif // HIP_PLUGIN_FUSION_LOWERING_H
