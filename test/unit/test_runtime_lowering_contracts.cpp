/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "RuntimeAccess.h"
#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/RuntimeAccessOpInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include <cstdio>

using namespace mlir;
using namespace mlir::hip;

namespace {
struct NoMatchPattern : RewritePattern {
  using Pattern::MatchAnyOpTypeTag;
  using Pattern::MatchInterfaceOpTypeTag;
  using Pattern::MatchTraitOpTypeTag;
  template <typename... Args>
  explicit NoMatchPattern(Args &&...args)
      : RewritePattern(std::forward<Args>(args)...) {}
  LogicalResult matchAndRewrite(Operation *, PatternRewriter &) const override {
    return failure();
  }
};

struct OverrideLowering : ConvertToLLVMPatternInterface {
  explicit OverrideLowering(Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &, LLVMTypeConverter &,
      RewritePatternSet &patterns) const override {
    patterns.add<NoMatchPattern>(CastOp::getOperationName(), 1, getContext());
  }
};

struct OverrideDialect : Dialect {
  explicit OverrideDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<OverrideDialect>()) {
    allowUnknownOperations();
    addInterfaces<OverrideLowering>();
  }
  static StringRef getDialectNamespace() { return "override"; }
};
} // namespace

int main() {
  MLIRContext ctx;
  ctx.loadDialect<HipDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect>();
  OperationName name(CastOp::getOperationName(), &ctx);
  int failures = 0;
  auto check = [&](bool ok, const char *label) {
    std::printf("%s %s\n", ok ? "[ OK ]" : "[FAIL]", label);
    failures += !ok;
  };
  check(!name.hasInterface<RuntimeAccessOpInterface>(),
        "the dialect alone does not promise backend behavior");
  DialectRegistry registry;
  registerRuntimeAccessOpInterfaceExternalModels(registry);
  ctx.appendDialectRegistry(registry);
  check(name.hasInterface<RuntimeAccessOpInterface>(),
        "the backend registers its access interface without LLVM patterns");

  RewritePatternSet alternatives(&ctx);
  alternatives.add<NoMatchPattern>("override.other", 1, &ctx);
  check(!RuntimeAccessOverrides(alternatives).contains(name),
        "an unrelated alternative retains the native contract");
  alternatives.add<NoMatchPattern>(name.getStringRef(), 1, &ctx);
  check(RuntimeAccessOverrides(alternatives).contains(name),
        "an alternative for the same op rejects the native contract");
  for (int kind = 0; kind != 3; ++kind) {
    alternatives.clear();
    if (kind == 0)
      alternatives.add<NoMatchPattern>(NoMatchPattern::MatchAnyOpTypeTag(), 1,
                                       &ctx);
    else if (kind == 1)
      alternatives.add<NoMatchPattern>(NoMatchPattern::MatchTraitOpTypeTag(),
                                       TypeID::get<OpTrait::ZeroRegions>(), 1,
                                       &ctx);
    else
      alternatives.add<NoMatchPattern>(
          NoMatchPattern::MatchInterfaceOpTypeTag(),
          MemoryEffectOpInterface::getInterfaceID(), 1, &ctx);
    check(RuntimeAccessOverrides(alternatives).contains(name),
          "an unknown-root alternative rejects native contracts");
  }
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  alternatives.clear();
  alternatives.add(PDLPatternModule(
      OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&ctx)))));
  check(RuntimeAccessOverrides(alternatives).contains(name),
        "PDL alternatives reject native contracts");
#endif

  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @mapped(%ctx: !hip.context, %out: memref<1xf32>) {
      %c0 = arith.constant 0 : index
      %size = arith.constant 64 : index
      %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
      %a = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
      hip.cast(%ctx) ins(%a : memref<1xi32>) outs(%out : memref<1xf32>)
          {to = 1 : i64}
      return
    }
  )mlir",
                                            &ctx);
  if (!module)
    return 1;
  module->walk([&](CastOp op) {
    auto access = cast<RuntimeAccessOpInterface>(op.getOperation());
    check(access.supportsHostMappedInputs() && !access.getEarlyHostRead(),
          "Cast retains its ordered mapped-input contract");
  });

  // A plugin loaded after placement cannot silently invalidate its contract.
  ctx.getOrLoadDialect<OverrideDialect>();
  module->getBody()->push_back(Operation::create(
      OperationState(UnknownLoc::get(&ctx), "override.anchor")));
  bool diagnosed = false;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagnosed |=
        diag.str().find("no host-mapped input contract") != std::string::npos;
    return success();
  });
  PassManager pm(&ctx);
  pm.addPass(createConvertHipToLLVMPass());
  check(failed(pm.run(*module)) && diagnosed,
        "lowering diagnoses an override of a mapped-input contract");
  bool retained = false;
  module->walk([&](CastOp) { retained = true; });
  check(retained, "contract failure occurs before IR conversion");
  return failures != 0;
}
