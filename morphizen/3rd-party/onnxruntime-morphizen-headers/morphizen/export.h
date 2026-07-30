// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#if defined(_WIN32)
#if MORPHIZEN_USE_DLL
#if MORPHIZEN_EXPORT_DLL == 1
#define MORPHIZEN_DLL_SPEC __declspec(dllexport)
#else
#define MORPHIZEN_DLL_SPEC __declspec(dllimport)
#endif
#else
#define MORPHIZEN_DLL_SPEC
#endif
#else
// Empty on non-Windows: protobuf v34's generated *.pb.h emits
//   class MORPHIZEN_DLL_SPEC [[gnu::warn_unused]] Name
// and GCC rejects a GNU __attribute__ placed before the C++11 [[...]]
// attribute (clang tolerates it). Default ELF visibility already exports these
// symbols, so an empty decoration is correct and portable across GCC and clang.
#define MORPHIZEN_DLL_SPEC
#endif

#if defined(_WIN32)
#if MORPHIZEN_USE_DLL == 1
#define MORPHIZEN_PASS_ENTRY __declspec(dllexport)
#else
#define MORPHIZEN_PASS_ENTRY
#endif
#else
#define MORPHIZEN_PASS_ENTRY __attribute__((visibility("default")))
#endif
