/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
/// @file Minimal ORT API header for use in source files that need Ort:: types
/// Include this header only in .cpp files or private .hpp files that
/// require ORT C++ API types, not in public headers to avoid dependency spread

#ifndef ORT_API_MANUAL_INIT
#  define ORT_API_MANUAL_INIT 1
#endif
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
#undef ORT_API_MANUAL_INIT
