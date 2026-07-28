/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPSR_TO_LLVM_H
#define HIP_CONVERSION_HIPSR_TO_LLVM_H

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace hipsr {

void populateHipsrAddLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateHipsrCastLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateHipsrConstantLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
void populateHipsrExpandLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populateHipsrGetPoolLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);
void populateHipsrMatMulLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSR_TO_LLVM_H
