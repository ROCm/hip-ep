/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Native runtime sources compiled by MSVC, unlike the runtime bitcode, still
// include RuntimeState, which pulls in hipBLASLt's headers, whose host code
// calls the clang/GCC builtin __builtin_clz (hipblaslt_e5m3.h). Force-included
// (/FI) ahead of those headers.

#ifndef HIPDNN_EP_MSVC_BUILTINS_H
#define HIPDNN_EP_MSVC_BUILTINS_H

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int __builtin_clz(unsigned int x) {
  unsigned long index;
  _BitScanReverse(&index, x);
  return 31 - static_cast<int>(index);
}
#endif

#endif // HIPDNN_EP_MSVC_BUILTINS_H
