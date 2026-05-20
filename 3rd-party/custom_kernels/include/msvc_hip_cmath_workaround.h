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
// Until the clang-hip headers are updated to detect MSVC 14.51's `<cmath>`,
// this header neutralises the MS STL macros that emit the offending overloads.
// MSVC's existing pre-14.51 `<cmath>` declarations of these symbols are kept,
// so host-side code compiles unchanged; the device-side overloads from
// `__clang_hip_cmath.h` stay in effect.
//
// Force-included from `_hip_compile_sources` in `3rd-party/custom_kernels/
// cmake/hip_utils.cmake` (Windows only). Safe to include redundantly: the
// macros are defined to expand to empty, so any later attempt to (re)define
// them by the MS STL is overridden.

#ifndef HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
#define HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H

#if defined(_MSC_VER) && defined(__clang__) && defined(__HIP__)

// The MS STL's `_CLANG_BUILTIN1` / `_CLANG_BUILTIN2` / `_CLANG_BUILTIN2_TEMPLATED`
// expansion in `<cmath>` is what produces the conflicting `inline` overloads.
// Pre-define them as empty so the expansion produces no declarations.
#define _CLANG_BUILTIN1(NAME)
#define _CLANG_BUILTIN2(NAME)
#define _CLANG_BUILTIN2_TEMPLATED(NAME)

#endif // _MSC_VER && __clang__ && __HIP__

#endif // HIPDNN_EP_MSVC_HIP_CMATH_WORKAROUND_H
