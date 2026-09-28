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
#define MORPHIZEN_DLL_SPEC __attribute__((visibility("default")))
#endif

// Decoration protoc stamps on the generated *.pb.h classes (passed as
// --cpp_out=dllexport_decl=...). It is deliberately empty, not
// MORPHIZEN_DLL_SPEC: protobuf v34 emits
//   class MORPHIZEN_PROTO_DLL_SPEC [[gnu::warn_unused]] Name
// and GCC rejects a GNU __attribute__ placed before a C++11 [[...]] attribute
// (only clang tolerates it). Generated protobuf symbols stay internal to the
// EP DSO; the version script / .def export only CreateEpFactories and
// ReleaseEpFactory. Keep in sync with the MORPHIZEN_PROTO_DLL_SPEC compile
// definition in morphizen-core/cmake/proto.cmake and
// morphizen-pattern/CMakeLists.txt.
#define MORPHIZEN_PROTO_DLL_SPEC

#if defined(_WIN32)
#if MORPHIZEN_USE_DLL == 1
#define MORPHIZEN_PASS_ENTRY __declspec(dllexport)
#else
#define MORPHIZEN_PASS_ENTRY
#endif
#else
#define MORPHIZEN_PASS_ENTRY __attribute__((visibility("default")))
#endif
