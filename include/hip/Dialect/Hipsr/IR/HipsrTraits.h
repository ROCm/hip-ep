/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_TRAITS_H
#define HIPSR_TRAITS_H

#include "mlir/IR/OpDefinition.h"
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

} // namespace OpTrait
} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TRAITS_H
