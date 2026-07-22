/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipsrToLLVM.cpp - Convert the hipsr dialect to LLVM ---------------===//
//
// Structure mirrors ConvertHipToLLVMPass (HipToLLVM/HipToLLVM.cpp): build an
// LLVMTypeConverter, register the !hip.context and #hipsr.mem<device>
// conversions, bundle the hipsr constant pattern with the standard
// func/memref/arith/cf lowering, and run a single partial conversion. Unlike
// the hip pass there is no @main_graph ABI rewrite here.
//
//===----------------------------------------------------------------------===//

#include "HipsrToLLVMUtils.h"

#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/IR/HipDialect.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTHIPSRTOLLVMPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

struct ConvertHipsrToLLVMPass
    : public impl::ConvertHipsrToLLVMPassBase<ConvertHipsrToLLVMPass> {
  void runOnOperation() override;
};

void ConvertHipsrToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  LowerToLLVMOptions options(ctx);
  LLVMTypeConverter typeConverter(ctx, options);

  // !hip.context -> !llvm.ptr (opaque pointer to the runtime context). hipsr
  // constants have no operand; the pass finds the ctx via the enclosing func.
  typeConverter.addConversion([ctx](hip::ContextType) -> Type {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });

  // #hipsr.mem<kind> -> integer address space. MemorySpaceKind's numeric
  // values already match the AMDGPU address spaces (host=0, device=1,
  // pinned=2, managed=3), so map the enum directly.
  typeConverter.addTypeAttributeConversion(
      [](BaseMemRefType,
         MemorySpaceAttr space) -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(IntegerType::get(space.getContext(), 64),
                                static_cast<int64_t>(space.getKind()));
      });

  // Index for each externalized constant = module walk order (0, 1, 2, ...),
  // the second @hipdnn_ep_constant_get argument. The same walk collects each
  // constant's size/offset (stamped append-only by hipsr-externalize-constants)
  // into hipdnn.constant_sizes / hipdnn.constant_offsets. Because a constant's
  // index and its metadata slot come from the same iteration over the same op,
  // the ConstantInfo[index] that GenerateInterface builds from these attrs is
  // aligned with the runtime @hipdnn_ep_constant_get(ctx, index) call by
  // construction -- no cross-pass walk-order assumption. Only sizes/offsets are
  // emitted (no per-source descriptors), so every ConstantInfo.source stays
  // NONE and the runtime bulk-loads the constants file.
  llvm::DenseMap<Operation *, int64_t> indexMap;
  llvm::SmallVector<int64_t> constantSizes, constantOffsets;
  int64_t nextIndex = 0;
  module.walk([&](ConstantOp c) {
    if (c.isExternalized()) {
      indexMap[c.getOperation()] = nextIndex++;
      constantOffsets.push_back(c.getOffsetAttr().getInt());
      constantSizes.push_back(c.getSizeAttr().getInt());
    }
  });
  if (!constantSizes.empty()) {
    module->setAttr("hipdnn.constant_sizes",
                    DenseI64ArrayAttr::get(ctx, constantSizes));
    module->setAttr("hipdnn.constant_offsets",
                    DenseI64ArrayAttr::get(ctx, constantOffsets));
  }

  // ctx for each externalized constant = the !hip.context block argument of its
  // enclosing function (do not assume arg 0). A missing ctx is diagnosed in the
  // pattern.
  llvm::DenseMap<Operation *, Value> ctxMap;
  module.walk([&](func::FuncOp fn) {
    Value ctxArg;
    for (BlockArgument arg : fn.getArguments()) {
      if (isa<hip::ContextType>(arg.getType())) {
        ctxArg = arg;
        break;
      }
    }
    fn.walk([&](ConstantOp c) {
      if (c.isExternalized()) {
        ctxMap[c.getOperation()] = ctxArg;
      }
    });
  });

  RewritePatternSet patterns(ctx);
  populateHipsrConstantLoweringPatterns(typeConverter, patterns, indexMap,
                                        ctxMap);

  // Bundle func/memref/arith/cf lowering with the hipsr lowering to minimize
  // unrealized casts at the memref/LLVM boundary (same rationale as
  // ConvertHipToLLVMPass).
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  LLVMConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalOp<ConstantOp>();
  target.addLegalOp<ModuleOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

} // namespace hipsr
} // namespace mlir
