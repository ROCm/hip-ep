/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
#include "./version_info.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>
#include <memory>

// Note: api() and set_the_global_api() are defined in morphizen-ort-api-ext.cpp
// Removed redundant wrapper functions that were causing duplicate symbols

extern "C" {
MORPHIZEN_DLL_SPEC
const morphizen::OrtApiForMorphizen* get_the_global_api() {
  // The test program is using this interface
  return morphizen::api();
}
const morphizen::OrtApiForMorphizen* get_the_global_api_unsafe() {
  // this is used by ort-bridge
  return morphizen::get_the_global_api_unsafe();
}
} // extern "C"
