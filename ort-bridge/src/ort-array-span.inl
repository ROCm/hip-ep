/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

namespace morphizen {

template <typename T>
OrtArraySpan<T>::OrtArraySpan(const ApiPtrs* api_ptrs,
                              OrtArrayOfConstObjects* array)
    : gsl::span<T>(), api_ptrs_(api_ptrs), array_(array) {
  if (!array_) {
    return;
  }

  size_t size = 0;
  api_ptrs_->throw_if_error(
      api_ptrs_->ort_api.ArrayOfConstObjects_GetSize(array_, &size));

  if (size == 0) {
    return;
  }

  const void* const* data = nullptr;
  api_ptrs_->throw_if_error(
      api_ptrs_->ort_api.ArrayOfConstObjects_GetData(array_, &data));

  // Reinitialize the base span with the actual data
  static_cast<gsl::span<T>&>(*this) =
      gsl::span<T>(reinterpret_cast<T*>(const_cast<void**>(data)), size);
}

template <typename T>
OrtArraySpan<T>::OrtArraySpan(OrtArraySpan&& other) noexcept
    : gsl::span<T>(other), api_ptrs_(other.api_ptrs_), array_(other.array_) {
  other.array_ = nullptr;
  static_cast<gsl::span<T>&>(other) = gsl::span<T>{};
}

template <typename T>
OrtArraySpan<T>& OrtArraySpan<T>::operator=(OrtArraySpan&& other) noexcept {
  if (this != &other) {
    release();
    static_cast<gsl::span<T>&>(*this) = other;
    api_ptrs_ = other.api_ptrs_;
    array_ = other.array_;
    other.array_ = nullptr;
    static_cast<gsl::span<T>&>(other) = gsl::span<T>{};
  }
  return *this;
}

template <typename T>
OrtArraySpan<T>::~OrtArraySpan() {
  release();
}

template <typename T>
void OrtArraySpan<T>::release() {
  if (array_ && api_ptrs_) {
    api_ptrs_->ort_api.ReleaseArrayOfConstObjects(array_);
    array_ = nullptr;
  }
}

} // namespace morphizen
