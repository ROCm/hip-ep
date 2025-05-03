/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
#include "./version_info.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include <glog/logging.h>
#include <memory>
#include <vaip/vaip_ort_api.h>
namespace vaip_core {

OrtApiForVaip* the_global_api = nullptr;
const OrtApiForVaip& __api() {
  DCHECK(the_global_api != nullptr)
      << "please set_the_global_api() before invoking this function";
  return *the_global_api;
}
VAIP_DLL_SPEC const OrtApiForVaip* api() { return &__api(); }
VAIP_DLL_SPEC void set_the_global_api(OrtApiForVaip* api) {
  if (the_global_api == api) {
    // python reset the api. If the api is the same, don't shift the address.
    return;
  }
  uint32_t onnx_major_version = 1;
  const char* magic = "VAIP";
  bool cmp =
      std::strncmp(reinterpret_cast<char*>(&api->magic), magic, strlen(magic));
  if (cmp == 0) {
    onnx_major_version = api->major;
  } else {
    // new vaip with old onnx, shift by version field
    uint64_t addr = reinterpret_cast<uint64_t>(&(api));
    uint64_t old_addr = reinterpret_cast<uint64_t>(&(api->host_));
    uint64_t diff = old_addr - addr;
    api = reinterpret_cast<OrtApiForVaip*>(addr - diff);
  }
  if (onnx_major_version != get_vaip_version_major()) {
    LOG(FATAL) << "version is not compatible: onnxruntime api version is: "
               << onnx_major_version
               << ", but vaip version is: " << get_vaip_version_major();
  }
  the_global_api = api;
  Ort::Global<void>::api_ = api->ort_api_;
  typedef void* void_ptr_t;
  auto p = (void_ptr_t*)(&(api->host_)); // first api addr
  size_t api_size =
      reinterpret_cast<size_t>(api + 1) - reinterpret_cast<size_t>(p);
  for (auto i = 0u; i < api_size / sizeof(void_ptr_t); ++i) {
    // coverity[ptr_arith]
    if (p[i] == nullptr) {
      LOG(FATAL) << "the_global_api[" << i << "] is not set";
    }
  }
}
} // namespace vaip_core
