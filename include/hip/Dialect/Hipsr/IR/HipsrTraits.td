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

// Marks an op that starts a new pool domain. The op is the first op of the
// new domain because its output shape needs the op's input data (e.g. Expand).
// Marker only, no verifier. See HipsrTraits.h.
def StartBarrier : NativeOpTrait<"StartBarrier"> {
  let cppNamespace = "::mlir::hipsr::OpTrait";
}

// Marks an op that ends the current pool domain. The op stays in the current
// domain; the next domain starts at the first op that uses this op's result.
// Its output shape needs the op's output data (e.g. NonZero). Marker only, no
// verifier. See HipsrTraits.h.
def EndBarrier : NativeOpTrait<"EndBarrier"> {
  let cppNamespace = "::mlir::hipsr::OpTrait";
}

#endif // HIPSR_TRAITS
