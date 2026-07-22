//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TRAITS
#define HIPSR_TRAITS

include "mlir/IR/OpBase.td"

// Like IsolatedFromAbove, but a region may also use the op's own operands (not
// just values defined inside the region). Used by shape regions that read the
// op's inputs to compute output shapes. See HipsrTraits.h for the verifier.
def IsolatedFromAboveButAllowOperands : NativeOpTrait<
    "IsolatedFromAboveButAllowOperands"> {
  let cppNamespace = "::mlir::hipsr::OpTrait";
}

#endif // HIPSR_TRAITS
