/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_RESIZE_ABI_H
#define HIP_RESIZE_ABI_H

//===----------------------------------------------------------------------===//
// The `mode` argument of wrap_resize / wrap_resize_scaled
//===----------------------------------------------------------------------===//
//
// hip.resize's lowering writes, and the runtime reads, one integer for the
// interpolation and its filter flags: the mode in the low byte, the flags
// above it (at the bits GGML's scale flags use). Both sides include this
// header, as they do hip/datatype_abi.h. The runtime hands the value to
// hip_resize, whose HIP_RESIZE_* (hip_custom_kernels.h) it checks against.
//===----------------------------------------------------------------------===//

#define HIPDNN_EP_RESIZE_MODE_NEAREST 0
#define HIPDNN_EP_RESIZE_MODE_LINEAR 1
#define HIPDNN_EP_RESIZE_MODE_CUBIC 2
#define HIPDNN_EP_RESIZE_MODE_MASK 0xFF
// ONNX antialias = 1 (linear only).
#define HIPDNN_EP_RESIZE_FLAG_ANTIALIAS (1 << 9)
// ONNX exclude_outside = 1.
#define HIPDNN_EP_RESIZE_FLAG_EXCLUDE_OUTSIDE (1 << 10)

#endif // HIP_RESIZE_ABI_H
