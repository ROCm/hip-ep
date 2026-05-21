// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// MSVC 14.51 (Visual Studio 17.13 / 2026 18.x) <cmath> added overloads of
// `isless` / `islessequal` / `islessgreater` / `isgreater` / `isgreaterequal` /
// `isunordered` / `isfinite` / `isinf` / `isnan` / `isnormal` that call
// `__builtin_*` directly. The MS STL guards them on `#ifdef __clang__` so they
// only fire under clang -- and in HIP compile mode clang treats their plain
// `inline` / `constexpr` declarations as implicitly `__host__ __device__`,
// colliding with the plain `__device__` overloads in clang-hip's bundled
// `__clang_hip_cmath.h` and `__clang_cuda_math_forward_declares.h`:
//
//   error: __device__ function 'isless' cannot overload
//          __host__ __device__ function 'isless'
//
// Status (MSVC 14.51.36231, May 2026): the primary fix is now the clang flag
// `-fno-cuda-host-device-constexpr` added to hipcc invocations in
// `hip_utils.cmake`, which the diagnostic itself recommends.  That flag tells
// clang not to implicitly mark unannotated `constexpr` functions as
// `__host__ __device__`, making MSVC's `<cmath>` overloads host-only and
// removing the collision entirely.
//
// This header remains in the build as a belt-and-suspenders backup.  It used
// to be the primary fix back when MSVC's pre-14.51.36231 `<cmath>` honoured
// pre-defined empty `_CLANG_BUILTIN1` / `_CLANG_BUILTIN2` macros; in
// 14.51.36231 the STL re-defines those macros unconditionally inside cmath
// (`#define _CLANG_BUILTIN2(NAME) ...` with no `#ifndef` guard), so empty
// pre-definitions no longer survive long enough to suppress the overload
// emission.  Force-included from `_hip_compile_sources` in
// `3rd-party/custom_kernels/cmake/hip_utils.cmake` (Windows only).

#ifndef HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
#define HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H

#if defined(_MSC_VER) && defined(__clang__) && defined(__HIP__)

// Layer 1: predefine the MS STL's `_CLANG_BUILTIN*` macros to empty.
#define _CLANG_BUILTIN1(NAME)
#define _CLANG_BUILTIN2(NAME)
#define _CLANG_BUILTIN2_TEMPLATED(NAME)

// Layer 2: rename the colliding function names before <cmath> pulls them in.
// Comparison ops (`_CLANG_BUILTIN2` family).
#define isgreater __msvc_hip_isgreater
#define isgreaterequal __msvc_hip_isgreaterequal
#define isless __msvc_hip_isless
#define islessequal __msvc_hip_islessequal
#define islessgreater __msvc_hip_islessgreater
#define isunordered __msvc_hip_isunordered
// Predicates (`_CLANG_BUILTIN1` family) — included defensively; the CI
// failure surfaced only the BUILTIN2 conflicts but the same MS STL change
// could expose BUILTIN1 conflicts on a future patch level.
#define isfinite __msvc_hip_isfinite
#define isinf __msvc_hip_isinf
#define isnan __msvc_hip_isnan
#define isnormal __msvc_hip_isnormal

// Force-include <cmath> NOW so the MS STL emits the renamed declarations
// while our `#define`s are still in scope. The header guard prevents the
// auto-included `__clang_hip_runtime_wrapper.h` chain from reintroducing
// the conflicting names later in the TU.
#include <cmath>

// Restore the original names so subsequent code (kernel device functions,
// downstream includes like `<hip/hip_runtime.h>`) sees `isgreater` resolve
// to clang-hip's `__device__` overloads.
#undef isgreater
#undef isgreaterequal
#undef isless
#undef islessequal
#undef islessgreater
#undef isunordered
#undef isfinite
#undef isinf
#undef isnan
#undef isnormal

#endif // _MSC_VER && __clang__ && __HIP__

#endif // HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
