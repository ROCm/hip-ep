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

// New morphizen-prefixed API
uint32_t get_morphizen_version_major();
uint32_t get_morphizen_version_minor();
uint32_t get_morphizen_version_patch();

// Deprecated VAIP aliases for backward compatibility
[[deprecated("Use get_morphizen_version_major()")]] inline uint32_t
get_vaip_version_major() {
  return get_morphizen_version_major();
}
[[deprecated("Use get_morphizen_version_minor()")]] inline uint32_t
get_vaip_version_minor() {
  return get_morphizen_version_minor();
}
[[deprecated("Use get_vaip_version_patch()")]] inline uint32_t
get_vaip_version_patch() {
  return get_morphizen_version_patch();
}

struct MorphizenOrtApiExt : public morphizen::OrtApiForMorphizen {
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

// Deprecated VAIP alias for backward compatibility
using VaipOrtApiExt [[deprecated("Use MorphizenOrtApiExt")]] =
    MorphizenOrtApiExt;

/**
 * @brief Create and initialize the MorphiZen ORT API
 * @return an opaque pointer which restore api.
 */
std::shared_ptr<void> setup_global_morphizen_ort_api(const char* backend_ir);

const morphizen::OrtApiForMorphizen*
get_global_morphizen_ort_api(const char* ir_backend_name);

const morphizen::OrtApiForMorphizen* get_the_global_api_unsafe();

// Deprecated VAIP aliases for backward compatibility
[[deprecated(
    "Use setup_global_morphizen_ort_api()")]] inline std::shared_ptr<void>
setup_global_vaip_ort_api(const char* backend_ir) {
  return setup_global_morphizen_ort_api(backend_ir);
}

[[deprecated("Use get_global_morphizen_ort_api()")]] inline const morphizen::
    OrtApiForMorphizen*
    get_global_vaip_ort_api(const char* ir_backend_name) {
  return get_global_morphizen_ort_api(ir_backend_name);
}

MORPHIZEN_DLL_SPEC void set_the_global_api(morphizen::OrtApiForMorphizen* api);
MORPHIZEN_DLL_SPEC const morphizen::OrtApiForMorphizen* api();

} // namespace morphizen

#define MORPHIZEN_ORT_API_EXT(name)                                            \
  (static_cast<const ::morphizen::MorphizenOrtApiExt*>(::morphizen::api())     \
               ->name != nullptr                                               \
       ? static_cast<const ::morphizen::MorphizenOrtApiExt*>(                  \
             ::morphizen::api())                                               \
             ->name                                                            \
       : (assert(false && #name " is not set"), nullptr))
