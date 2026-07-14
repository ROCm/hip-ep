/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_TRAITS_H
#define HIPSR_TRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hipsr {
namespace OpTrait {
namespace impl {
/// Verifier for the `IsolatedFromAboveButAllowOperands` trait: every value used
/// in `op`'s regions must be defined inside those regions (block arguments
/// included) or be one of `op`'s operands. `op` must carry the trait.
/// See HipsrTraits.cpp.
::mlir::LogicalResult
verifyIsolatedFromAboveButAllowOperands(::mlir::Operation *op);
} // namespace impl

/// `IsolatedFromAboveButAllowOperands` op trait; forwards to the verifier
/// above. Uses verifyRegionTrait (like IsIsolatedFromAbove) so it runs after
/// the nested region ops are verified.
template <typename ConcreteType>
class IsolatedFromAboveButAllowOperands
    : public ::mlir::OpTrait::TraitBase<ConcreteType,
                                        IsolatedFromAboveButAllowOperands> {
public:
  static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
    return impl::verifyIsolatedFromAboveButAllowOperands(op);
  }
};

/// `SingleBlockExplicitTerminator<Op>` op trait: like
/// `mlir::OpTrait::SingleBlockImplicitTerminator<Op>`, but the terminator is
/// never auto-inserted -- each non-empty region must already end with `Op`.
/// hipsr shape regions use it because the terminator (`hipsr.shape_yield`)
/// carries the computed shape, so a missing one is an error, not a default.
/// The one-block-per-region check comes from `SingleBlock`, which the ODS
/// `TraitList` adds alongside this trait. The nested `Impl` template matches
/// upstream so ODS can name the trait as `SingleBlockExplicitTerminator<Op>`.
template <typename TerminatorOpType> struct SingleBlockExplicitTerminator {
  template <typename ConcreteType>
  class Impl : public ::mlir::OpTrait::TraitBase<
                   ConcreteType,
                   SingleBlockExplicitTerminator<TerminatorOpType>::Impl> {
  public:
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      for (::mlir::Region &region : op->getRegions()) {
        // An empty region has nothing to terminate.
        if (region.empty())
          continue;
        ::mlir::Block &block = region.front();
        if (block.empty() || !::mlir::isa<TerminatorOpType>(block.back()))
          return op->emitOpError("region must end with an explicit '")
                 << TerminatorOpType::getOperationName()
                 << "' terminator (it is not auto-inserted)";
      }
      return ::mlir::success();
    }
  };
};

} // namespace OpTrait
} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TRAITS_H
