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
// Force-included from `_hip_compile_sources` in `3rd-party/custom_kernels/
// cmake/hip_utils.cmake` (Windows only). Safe to include redundantly — every
// transform below is idempotent.
//
// Two layered defences (the second is load-bearing on MSVC 14.51.36231+):
//
//   1. Predefine `_CLANG_BUILTIN1/2/2_TEMPLATED` to empty.
//      Works on MSVC versions whose `<yvals_core.h>` guards these with
//      `#ifndef _CLANG_BUILTIN2 ... #endif`. Cheap and harmless.
//
//   2. Rename the colliding function names AND force-include `<cmath>` here,
//      so the MS STL's expansions emit `inline bool __msvc_hip_isgreater(...)`
//      etc. instead of `isgreater(...)`. The header guard on `<cmath>`
//      prevents re-inclusion later in the TU, so clang's auto-included
//      `__clang_hip_runtime_wrapper.h -> __clang_hip_cmath.h -> <cmath>`
//      path sees the already-loaded header and skips it. Clang-hip's
//      `__device__ bool isgreater(...)` declarations in
//      `__clang_cuda_math_forward_declares.h` / `__clang_hip_cmath.h` then
//      have no host overload to collide with.
//      Load-bearing on MSVC 14.51.36231 (VS 18 Enterprise), where the MS
//      STL `#define`s `_CLANG_BUILTIN2` unconditionally — predefine alone
//      is insufficient on that release.
//
//      We `#undef` the renames after `<cmath>` so any subsequent code that
//      references `isgreater(x, y)` resolves to clang-hip's `__device__`
//      overloads. The custom kernels in `3rd-party/custom_kernels/hip/`
//      have no host-side call sites for these names — verified by grep.
//      If a future kernel adds one, declare a host-side shim around
//      `__msvc_hip_<name>(...)` next to the call (or move the helper into
//      a separate `.cpp` TU that doesn't go through this workaround).

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
