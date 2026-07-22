//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TRAITS
#define HIPSR_TRAITS

include "mlir/IR/OpBase.td"

// Like IsolatedFromAbove, but a region may also read the op's own operands.
// Scoped to pool-domain-style grouping ops (e.g. hipsr.pool_domain) whose
// region references those operands directly. Shape-region ops do NOT use this
// -- they are standard IsolatedFromAbove and thread inputs in as entry-block
// args (see ShapeRegionInterface). Verifier in HipsrTraits.h.
def IsolatedFromAboveButAllowOperands : NativeOpTrait<
    "IsolatedFromAboveButAllowOperands"> {
  let cppNamespace = "::mlir::hipsr::OpTrait";
}

#endif // HIPSR_TRAITS
