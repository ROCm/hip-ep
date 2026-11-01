/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPTOLLVM_RUNTIMELOWERINGPATTERNS_H
#define HIP_CONVERSION_HIPTOLLVM_RUNTIMELOWERINGPATTERNS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::hip {

// Payload accesses use the context stream. Host reads use an ordered copy
// and wait, except for the optional operand read before that wait.
// This contract does not make the operation a completion barrier.
struct RuntimeAccessContract {
  using EarlyHostRead = Value (*)(Operation *);
  EarlyHostRead earlyHostRead = nullptr;
  // Read-only inputs can use runtime-owned, host-mapped storage.
  bool hostMappedInputs = false;
};

using RuntimeAccessContracts = DenseMap<OperationName, RuntimeAccessContract>;

// Register access contracts with the lowerings that implement them. These
// contracts apply to this backend, not to the source dialect in general.
class RuntimeLoweringPatterns : public RewritePatternSet {
public:
  using RewritePatternSet::RewritePatternSet;

  template <typename... Patterns, typename... Args>
  void addOrdered(Args &&...args) {
    (addWithContract<Patterns>({}, std::forward<Args>(args)...), ...);
  }

  // The ordered lowering also accepts host-mapped input storage.
  template <typename Pattern, typename... Args>
  void addOrderedMapped(Args &&...args) {
    addWithContract<Pattern>({nullptr, true}, std::forward<Args>(args)...);
  }

  template <typename Pattern, typename... Args>
  void addWithEarlyRead(RuntimeAccessContract::EarlyHostRead read,
                        Args &&...args) {
    addWithContract<Pattern>({read}, std::forward<Args>(args)...);
  }

  // Call once, after all lowerings are registered, including plugin lowerings.
  // Keep only guarantees shared by every alternative for an operation.
  RuntimeAccessContracts finalizeAccessContracts() {
    RuntimeAccessContracts result;
    bool unknownRoot = false;
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
    unknownRoot = bool(getPDLPatterns().getModule());
#endif
    DenseSet<OperationName> rejected;
    // Unclassified patterns can override a declared lowering.
    for (const auto &pattern : getNativePatterns()) {
      if (auto root = pattern->getRootKind())
        rejected.insert(*root);
      else
        unknownRoot = true;
    }
    for (auto &entry : declared) {
      auto root = entry.pattern->getRootKind();
      if (!root) {
        unknownRoot = true;
        break;
      }
      auto [it, inserted] = result.try_emplace(*root, entry.contract);
      if (!inserted && it->second.earlyHostRead != entry.contract.earlyHostRead)
        rejected.insert(*root);
      it->second.hostMappedInputs &= entry.contract.hostMappedInputs;
    }
    // Preserve insertion order: equal-benefit patterns use that order.
    auto native = std::move(getNativePatterns());
    getNativePatterns().clear();
    size_t next = 0;
    for (auto &entry : declared) {
      while (next < entry.nativeCount && next < native.size())
        add(std::move(native[next++]));
      add(std::move(entry.pattern));
    }
    while (next < native.size())
      add(std::move(native[next++]));
    declared.clear();
    if (unknownRoot)
      return RuntimeAccessContracts();
    for (OperationName name : rejected)
      result.erase(name);
    return result;
  }

private:
  template <typename Pattern, typename... Args>
  void addWithContract(RuntimeAccessContract contract, Args &&...args) {
    declared.push_back(
        {RewritePattern::create<Pattern>(std::forward<Args>(args)...), contract,
         getNativePatterns().size()});
  }

  struct DeclaredPattern {
    std::unique_ptr<RewritePattern> pattern;
    RuntimeAccessContract contract;
    size_t nativeCount;
  };
  SmallVector<DeclaredPattern> declared;
};

} // namespace mlir::hip

#endif
