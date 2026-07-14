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

// Like SingleBlockImplicitTerminator<op>, but the terminator must be written
// explicitly -- it is never auto-inserted. Used for shape regions, where the
// terminator (hipsr.shape_yield) carries the computed shape, so a missing one
// is a real error, not something to fill in. See HipsrTraits.h for the check.
class SingleBlockExplicitTerminatorImpl<string op>
    : ParamNativeOpTrait<"SingleBlockExplicitTerminator", op, [SingleBlock]>,
      StructuralOpTrait {
  let cppNamespace = "::mlir::hipsr::OpTrait";
}

class SingleBlockExplicitTerminator<string op>
    : TraitList<[SingleBlock, SingleBlockExplicitTerminatorImpl<op>]>;

#endif // HIPSR_TRAITS
