/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "RuntimeLoweringPatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cstdio>

using namespace mlir;
using namespace mlir::hip;

namespace {
Value firstOperand(Operation *op) { return op->getOperand(0); }

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
} // namespace

int main() {
  MLIRContext ctx;
  OperationName name("test.compute", &ctx);
  int failures = 0;
  auto check = [&](bool ok, const char *label) {
    std::printf("%s %s\n", ok ? "[ OK ]" : "[FAIL]", label);
    failures += !ok;
  };
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrderedMapped<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.add<NoMatchPattern>("test.other", 1, &ctx);
    auto contracts = patterns.finalizeAccessContracts();
    check(contracts.size() == 1 && contracts.contains(name) &&
              !contracts.lookup(name).earlyHostRead &&
              contracts.lookup(name).hostMappedInputs,
          "an unrelated lowering retains ordering and mapped-input support");
    check(patterns.getNativePatterns().size() == 2,
          "finalization retains all patterns");
  }
  for (bool unknownFirst : {false, true}) {
    RuntimeLoweringPatterns patterns(&ctx);
    if (unknownFirst)
      patterns.add<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.addOrderedMapped<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    if (!unknownFirst)
      patterns.add<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    const RewritePattern *unclassified =
        patterns.getNativePatterns().front().get();
    check(patterns.finalizeAccessContracts().empty(),
          "an unclassified alternative rejects the contract in either order");
    check(patterns.getNativePatterns().size() == 2 &&
              patterns.getNativePatterns()[unknownFirst ? 0 : 1].get() ==
                  unclassified,
          "a rejected contract preserves lowering order");
  }
  for (bool mappedAlternative : {false, true}) {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrderedMapped<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    if (mappedAlternative)
      patterns.addOrderedMapped<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    else
      patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    auto contracts = patterns.finalizeAccessContracts();
    check(contracts.contains(name) &&
              contracts.lookup(name).hostMappedInputs == mappedAlternative,
          "mapped-input support requires all alternatives; ordering survives");
  }
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    check(patterns.finalizeAccessContracts().contains(name),
          "alternatives with the same contract are accepted");
  }
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.addWithEarlyRead<NoMatchPattern>(firstOperand, name.getStringRef(),
                                              1, &ctx);
    check(patterns.finalizeAccessContracts().empty(),
          "alternatives with different contracts are rejected");
  }
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addWithEarlyRead<NoMatchPattern>(firstOperand, name.getStringRef(),
                                              1, &ctx);
    auto contracts = patterns.finalizeAccessContracts();
    check(contracts.contains(name) &&
              contracts.lookup(name).earlyHostRead == firstOperand,
          "an early-read contract retains its operand query");
  }
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.add<NoMatchPattern>(NoMatchPattern::MatchAnyOpTypeTag(), 1, &ctx);
    check(patterns.finalizeAccessContracts().empty(),
          "an any-op alternative disables contracts");
    check(patterns.getNativePatterns().size() == 2,
          "an any-op alternative does not discard declared lowerings");
  }
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.add<NoMatchPattern>(NoMatchPattern::MatchTraitOpTypeTag(),
                                 TypeID::get<OpTrait::ZeroRegions>(), 1, &ctx);
    check(patterns.finalizeAccessContracts().empty(),
          "a trait alternative disables contracts");
  }
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.add(PDLPatternModule(
        OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&ctx)))));
    check(patterns.finalizeAccessContracts().empty(),
          "PDL patterns disable contracts");
    check(patterns.getNativePatterns().size() == 1,
          "PDL patterns do not discard declared lowerings");
  }
#endif
  {
    RuntimeLoweringPatterns patterns(&ctx);
    patterns.addOrdered<NoMatchPattern>(name.getStringRef(), 1, &ctx);
    patterns.add<NoMatchPattern>(NoMatchPattern::MatchInterfaceOpTypeTag(),
                                 MemoryEffectOpInterface::getInterfaceID(), 1,
                                 &ctx);
    check(patterns.finalizeAccessContracts().empty(),
          "an interface alternative disables contracts");
  }
  return failures != 0;
}
