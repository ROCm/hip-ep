//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

#ifndef HIPSR_TYPES
#define HIPSR_TYPES

include "mlir/IR/OpBase.td"

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

#endif // HIPSR_TYPES
