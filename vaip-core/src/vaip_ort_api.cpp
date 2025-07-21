/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
#include "./version_info.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include "morphizen/vaip-ort-api-ext.hpp"
#include <glog/logging.h>
#include <memory>
namespace vaip_core {

VAIP_DLL_SPEC const OrtApiForVaip* api() { return morphizen::api(); }
VAIP_DLL_SPEC void set_the_global_api(OrtApiForVaip* api) {
  morphizen::set_the_global_api(api);
}
} // namespace vaip_core

extern "C" {
VAIP_DLL_SPEC
const vaip_core::OrtApiForVaip* get_the_global_api() {
  // The test program is using this interface
  return morphizen::api();
}
const vaip_core::OrtApiForVaip* get_the_global_api_unsafe() {
  // this is used by ort-bridge
  return morphizen::get_the_global_api_unsafe();
}
} // extern "C"
