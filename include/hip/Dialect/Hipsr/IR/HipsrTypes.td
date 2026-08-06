//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TYPES
#define HIPSR_TYPES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "hip/Dialect/Hipsr/IR/HipsrDialect.td"

// Type constraints for ops that need a memref in a specific memory space.
//
// One per memory space, in enum order (host, device, pinned, managed). A memref
// with no space, or a space that is not a `#hipsr.mem<...>` attribute, does not
// match (see the predicates in HipsrTypes.cpp).
def Hipsr_HostMemRef    : Type<CPred<"::mlir::hipsr::isHostMemRef($_self)">,    "host memref">;
def Hipsr_DeviceMemRef  : Type<CPred<"::mlir::hipsr::isDeviceMemRef($_self)">,  "device memref">;
def Hipsr_PinnedMemRef  : Type<CPred<"::mlir::hipsr::isPinnedMemRef($_self)">,  "pinned memref">;
def Hipsr_ManagedMemRef : Type<CPred<"::mlir::hipsr::isManagedMemRef($_self)">, "managed memref">;

// Matches a ranked tensor (which has no memory space) or a memref in the named
// space. For ops that run both before and after bufferization.
def Hipsr_TensorOrHostMemRef    : AnyTypeOf<[AnyRankedTensor, Hipsr_HostMemRef],    "ranked tensor or host memref">;
def Hipsr_TensorOrDeviceMemRef  : AnyTypeOf<[AnyRankedTensor, Hipsr_DeviceMemRef],  "ranked tensor or device memref">;
def Hipsr_TensorOrPinnedMemRef  : AnyTypeOf<[AnyRankedTensor, Hipsr_PinnedMemRef],  "ranked tensor or pinned memref">;
def Hipsr_TensorOrManagedMemRef : AnyTypeOf<[AnyRankedTensor, Hipsr_ManagedMemRef], "ranked tensor or managed memref">;

// Deliberately space agnostic, unlike the four constraints above: a ranked
// tensor or a ranked memref in ANY space, including none. Only for metadata ops
// that hold a value without touching it, and therefore have to verify at every
// point of the pipeline -- including right after `-one-shot-bufferize`, whose
// `memref.alloc`s carry no `#hipsr.mem` space yet. Compute ops must keep naming
// their space.
def Hipsr_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef], "ranked tensor or memref">;

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
