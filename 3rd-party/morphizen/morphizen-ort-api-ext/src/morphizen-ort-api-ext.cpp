/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//  ERROR macro is defined. Define GLOG_NO_ABBREVIATED_SEVERITIES
//  before including logging.h. See the document for
//  detail.
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#  define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
// disable all warnings in this file on Windows
#if defined(_WIN32)
#  pragma warning(push, 0)
#endif
#include <glog/logging.h>
#if defined(_WIN32)
#  pragma warning(pop)
#endif
#ifndef ORT_API_MANUAL_INIT
#  define ORT_API_MANUAL_INIT 1
#endif
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
//
#include "morphizen/morphizen-ort-api-ext.hpp"
//
#include "morphizen-utils/morphizen_plugin.hpp"

namespace morphizen {

unsigned int get_morphizen_version_major() {
#ifdef MORPHIZEN_ORT_API_MAJOR
  return MORPHIZEN_ORT_API_MAJOR;
#else
  return 1;
#endif
}

unsigned int get_morphizen_version_minor() {
#ifdef MORPHIZEN_ORT_API_MINOR
  return MORPHIZEN_ORT_API_MINOR;
#else
  return 0;
#endif
}

unsigned int get_morphizen_version_patch() {
#ifdef MORPHIZEN_ORT_API_PATCH
  return MORPHIZEN_ORT_API_PATCH;
#else
  return 0;
#endif
}
using OrtApiForMorphizen = morphizen::OrtApiForMorphizen;
static OrtApiForMorphizen* the_global_api = nullptr;

const OrtApiForMorphizen& __api() {
  DCHECK(the_global_api != nullptr)
      << "please set_the_global_api() before invoking this function";
  return *the_global_api;
}

MORPHIZEN_DLL_SPEC const OrtApiForMorphizen* api() { return &__api(); }

MORPHIZEN_DLL_SPEC void set_the_global_api(OrtApiForMorphizen* api) {
  if (the_global_api == api) {
    // python reset the api. If the api is the same, don't shift the address.
    return;
  }
  uint32_t onnx_major_version = 1;
  const char magic[4] = {0x56, 0x41, 0x49,
                         0x50}; // Binary format identifier bytes
  bool cmp =
      std::strncmp(reinterpret_cast<char*>(&api->magic), magic, sizeof(magic));
  if (cmp == 0) {
    onnx_major_version = api->major;
  } else {
    // new morphizen with old onnx, shift by version field
    uint64_t addr = reinterpret_cast<uint64_t>(&(api));
    uint64_t old_addr = reinterpret_cast<uint64_t>(&(api->host_));
    uint64_t diff = old_addr - addr;
    api = reinterpret_cast<OrtApiForMorphizen*>(addr - diff);
  }
  if (onnx_major_version != get_morphizen_version_major()) {
    LOG(FATAL) << "version is not compatible: onnxruntime api version is: "
               << onnx_major_version << ", but morphizen version is: "
               << get_morphizen_version_major();
  }
  the_global_api = api;
  Ort::InitApi(api->ort_api_);
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
const morphizen::OrtApiForMorphizen*
get_global_morphizen_ort_api(const char* ir_backend_name) {
  auto plugin = morphizen::Plugin::get(ir_backend_name);
  CHECK(plugin != nullptr) << "Failed to get plugin for ir_backend_name: "
                           << ir_backend_name;
  auto method = plugin->get_method<const morphizen::OrtApiForMorphizen*>(
      "morphizen_ort_api_imp");
  CHECK(method != nullptr)
      << "Failed to get method 'morphizen_ort_api_imp' from plugin: "
      << ir_backend_name;
  auto api = method();
  CHECK(api != nullptr) << "Failed to get morphizen_ort_api from plugin: "
                        << ir_backend_name;
  return api;
}

std::shared_ptr<void>
setup_global_morphizen_ort_api(const char* ir_backend_name) {
  auto old_global_api = the_global_api;
  auto deferred_recover_the_global_api =
      std::shared_ptr<void>((void*)old_global_api, [](void* old_global_api) {
        if (old_global_api != nullptr) {
          set_the_global_api(
              static_cast<morphizen::OrtApiForMorphizen*>(old_global_api));
        }
      });
  auto new_global_api = get_global_morphizen_ort_api(ir_backend_name);
  set_the_global_api(
      const_cast<morphizen::OrtApiForMorphizen*>(new_global_api));
  return deferred_recover_the_global_api;
}

const morphizen::OrtApiForMorphizen* get_the_global_api_unsafe() {
  return the_global_api;
}

} // namespace morphizen
