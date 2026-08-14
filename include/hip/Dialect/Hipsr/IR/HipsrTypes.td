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

def Hipsr_AnyHostRankedTensor   : Type<CPred<"::mlir::hipsr::isHostRankedTensor($_self)">,   "host ranked tensor">;
def Hipsr_AnyDeviceRankedTensor : Type<CPred<"::mlir::hipsr::isDeviceRankedTensor($_self)">, "device ranked tensor">;

def Hipsr_HostTensorOrMemRef    : AnyTypeOf<[Hipsr_AnyHostRankedTensor, Hipsr_HostMemRef],    "ranked host tensor or host memref">;
def Hipsr_DeviceTensorOrMemRef  : AnyTypeOf<[Hipsr_AnyDeviceRankedTensor, Hipsr_DeviceMemRef],  "ranked device tensor or device memref">;
def Hipsr_TensorOrPinnedMemRef  : AnyTypeOf<[AnyRankedTensor, Hipsr_PinnedMemRef],  "ranked tensor or pinned memref">;
def Hipsr_TensorOrManagedMemRef : AnyTypeOf<[AnyRankedTensor, Hipsr_ManagedMemRef], "ranked tensor or managed memref">;
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
