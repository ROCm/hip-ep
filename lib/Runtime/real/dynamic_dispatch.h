/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_EP_DYNAMIC_DISPATCH_H
#define HIPDNN_EP_DYNAMIC_DISPATCH_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// DynamicDispatch Backend Implementation
//===----------------------------------------------------------------------===//
//
// Implementation details for wrap_dd_matmul and wrap_dd_conv2d.
// Function declarations are in hipdnn_ep_runtime.h to avoid duplication.
//
// This header is for internal use by dynamic_dispatch.cpp only.
//===----------------------------------------------------------------------===//

// Forward declare RuntimeState (full definition in runtime_state_internal.h)
struct RuntimeState;

#ifdef __cplusplus
}
#endif

#endif // HIPDNN_EP_DYNAMIC_DISPATCH_H
