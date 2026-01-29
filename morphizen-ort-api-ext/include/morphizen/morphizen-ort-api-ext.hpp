/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#ifndef ORT_API_MANUAL_INIT
#  define ORT_API_MANUAL_INIT 1
#endif
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>
#undef ORT_API_MANUAL_INIT

#include <morphizen/morphizen_ort_api.h>

namespace morphizen {
static constexpr const char* kONNXIRBackend = "onnx-ir-imp";
static constexpr const char* kMLIRBackend = "mlir-backend";

uint32_t get_vaip_version_major();
uint32_t get_vaip_version_minor();
uint32_t get_vaip_version_patch();

struct VaipOrtApiExt : public morphizen::OrtApiForMorphizen {
  morphizen::TensorProto* (*tensor_proto_new_with_external_data)(
      const std::string& name,           //
      const std::vector<int64_t>& shape, //
      int element_type,                  //
      const std::string& location_file,  //
      size_t location_size,              //
      size_t location_offset             //
  );
  morphizen::TensorProto* (*tensor_proto_new_raw_data)(
      const std::string& name,           //
      const std::vector<int64_t>& shape, //
      int element_type,                  //
      const void* data,                  //
      size_t size);
};

/**
 * @brief Create and initialize the VAIP ORT API
 * @return an opaque pointer which restore api.
 */
std::shared_ptr<void> setup_global_vaip_ort_api(const char* backend_ir);

const morphizen::OrtApiForMorphizen*
get_global_vaip_ort_api(const char* ir_backend_name);

const morphizen::OrtApiForMorphizen* get_the_global_api_unsafe();
const morphizen::OrtApiForMorphizen* get_global_vaip_ort_api();

MORPHIZEN_DLL_SPEC void set_the_global_api(morphizen::OrtApiForMorphizen* api);
MORPHIZEN_DLL_SPEC const morphizen::OrtApiForMorphizen* api();

} // namespace morphizen

#define MORPHIZEN_ORT_API_EXT(name)                                            \
  (static_cast<const ::morphizen::VaipOrtApiExt*>(::morphizen::api())->name != \
           nullptr                                                             \
       ? static_cast<const ::morphizen::VaipOrtApiExt*>(::morphizen::api())    \
             ->name                                                            \
       : (assert(false && #name " is not set"), nullptr))
