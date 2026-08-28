//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TYPES
#define HIPSR_TYPES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "hip/Dialect/Hipsr/IR/HipsrDialect.td"

// Type constraints for ops that need a memref in a named memory space, one per
// space in enum order (host, device, pinned, managed), plus one for an op that
// takes any of them. A memref with no space, or a space that is not a
// `#hipsr.mem<...>` attribute, matches none of them (see the predicates in
// HipsrTypes.cpp).
def Hipsr_HostMemRef    : Type<CPred<"::mlir::hipsr::isHostMemRef($_self)">,    "host memref">;
def Hipsr_DeviceMemRef  : Type<CPred<"::mlir::hipsr::isDeviceMemRef($_self)">,  "device memref">;
def Hipsr_PinnedMemRef  : Type<CPred<"::mlir::hipsr::isPinnedMemRef($_self)">,  "pinned memref">;
def Hipsr_ManagedMemRef : Type<CPred<"::mlir::hipsr::isManagedMemRef($_self)">, "managed memref">;
def Hipsr_AnySpaceMemRef : Type<CPred<"::mlir::hipsr::isAnySpaceMemRef($_self)">, "memref in a hipsr memory space">;

// Matches a ranked tensor (which has no memory space) or a memref in the named
// space. For ops that run both before and after bufferization.
def Hipsr_TensorOrHostMemRef    : AnyTypeOf<[AnyRankedTensor, Hipsr_HostMemRef],    "ranked tensor or host memref">;
def Hipsr_TensorOrDeviceMemRef  : AnyTypeOf<[AnyRankedTensor, Hipsr_DeviceMemRef],  "ranked tensor or device memref">;
def Hipsr_TensorOrPinnedMemRef  : AnyTypeOf<[AnyRankedTensor, Hipsr_PinnedMemRef],  "ranked tensor or pinned memref">;
def Hipsr_TensorOrManagedMemRef : AnyTypeOf<[AnyRankedTensor, Hipsr_ManagedMemRef], "ranked tensor or managed memref">;
def Hipsr_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef], "ranked tensor or memref">;
def Hipsr_TensorOrAnySpaceMemRef : AnyTypeOf<[AnyRankedTensor, Hipsr_AnySpaceMemRef], "ranked tensor or hipsr memref">;

// Base class for hipsr types.
class Hipsr_Type<string name, string typeMnemonic> : TypeDef<Hipsr_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Hipsr_ContextType : Hipsr_Type<"Context", "context"> {
  let summary = "Opaque hipsr execution context";
  let description = [{
    Opaque per-session context. hipsr ops take it as their first operand.
    Lowers to `!llvm.ptr`.
  }];
}

#endif // HIPSR_TYPES
