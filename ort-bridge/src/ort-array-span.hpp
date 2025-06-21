/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include <gsl/gsl>
#include <memory>
#define ORT_API_MANUAL_INIT
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/session/onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

namespace morphizen {

// Forward declaration
struct ApiPtrs;

/**
 * @brief RAII wrapper for OrtArrayOfConstObjects with automatic lifetime
 * management
 *
 * This class derives from gsl::span<T> and provides a safe interface for
 * accessing OrtArrayOfConstObjects while ensuring automatic cleanup when the
 * object goes out of scope.
 *
 * Key features:
 * - RAII: Automatically releases the underlying OrtArrayOfConstObjects
 * - Move-only semantics: Prevents accidental copies that could lead to
 * double-free
 * - Inherits from gsl::span<T>: Provides full span interface and implicit
 * conversion
 * - Type-safe: Template parameter ensures proper typing of array elements
 *
 * Usage example:
 * @code
 * auto nodes_span = graph.nodes_managed();  // Returns OrtArraySpan<OrtValue*>
 * for (auto* node : nodes_span) {
 *   // Use node...
 * }
 * gsl::span<OrtValue*> regular_span = nodes_span; // Implicit conversion
 * // Array is automatically released when nodes_span goes out of scope
 * @endcode
 */
template <typename T> class OrtArraySpan : public gsl::span<T> {
private:
  const ApiPtrs* api_ptrs_;
  OrtArrayOfConstObjects* array_;

public:
  /**
   * @brief Constructs an OrtArraySpan taking ownership of the given array
   * @param api_ptrs Pointer to ApiPtrs for accessing ORT API functions
   * @param array The OrtArrayOfConstObjects to wrap (takes ownership)
   */
  OrtArraySpan(const ApiPtrs* api_ptrs, OrtArrayOfConstObjects* array);
  // Non-copyable but movable
  OrtArraySpan(const OrtArraySpan&) = delete;
  OrtArraySpan& operator=(const OrtArraySpan&) = delete;

  /**
   * @brief Move constructor
   */
  OrtArraySpan(OrtArraySpan&& other) noexcept;

  /**
   * @brief Move assignment operator
   */
  OrtArraySpan& operator=(OrtArraySpan&& other) noexcept;

  /**
   * @brief Destructor automatically releases the array
   */
  ~OrtArraySpan();

private:
  void release();
};

} // namespace morphizen
