/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipsrToLLVM.cpp - hipsr -> LLVM via ConvertToLLVMPatternInterface -===//
//
// HipsrDialect implements ConvertToLLVMPatternInterface, so the standard LLVM
// conversion driver discovers and applies hipsr lowering with no dedicated
// pass. This repo's convert-hip-to-llvm walks the module and, for every
// dialect that genuinely implements the interface, calls its populate (see
// HipToLLVM.cpp). The driver already sets up the LLVMTypeConverter (incl.
// !hip.context -> !llvm.ptr) and bundles the func/memref/arith/cf lowering, so
// the interface only adds the hipsr-specific type conversion, target legality,
// and patterns.
//
//===----------------------------------------------------------------------===//

#include "HipsrToLLVMUtils.h"

#include "hip/Conversion/HipsrToLLVM/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace hipsr {
namespace {

// Aggregates every hipsr -> LLVM pattern populator. As each op is ported, add
// its populateHipsr<Op>LoweringPatterns here (pattern defined in its own
// <Op>Lowering.cpp, declared in HipsrToLLVMUtils.h) and its addIllegalOp in the
// interface below. Mirrors how convert-hip-to-llvm's runOnOperation aggregates
// the per-category populate*LoweringPatterns.
void populateHipsrToLLVMPatterns(const LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  populateHipsrConstantLoweringPatterns(typeConverter, patterns);
}

struct HipsrConvertToLLVMInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    // #hipsr.mem<kind> -> integer address space. MemorySpaceKind's numeric
    // values already match the AMDGPU address spaces (host=0, device=1,
    // pinned=2, managed=3), so map the enum directly. (!hip.context ->
    // !llvm.ptr is registered by the driver.)
    typeConverter.addTypeAttributeConversion(
        [](BaseMemRefType,
           MemorySpaceAttr space) -> TypeConverter::AttributeConversionResult {
          return IntegerAttr::get(IntegerType::get(space.getContext(), 64),
                                  static_cast<int64_t>(space.getKind()));
        });
    // Only hipsr.constant has a lowering today; mark just it illegal (not the
    // whole dialect) so as-yet-unported hipsr ops do not fail legalization.
    target.addIllegalOp<ConstantOp>();
    populateHipsrToLLVMPatterns(typeConverter, patterns);
  }
};

} // namespace

void registerConvertHipsrToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, HipsrDialect *dialect) {
    dialect->addInterfaces<HipsrConvertToLLVMInterface>();
  });
}

} // namespace hipsr
} // namespace mlir
