/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
// must include graph.hpp first, because `main_graph` return vaip_cxx::Graph object by value.
#include "morphizen/graph.hpp"
// clang-format on
#include "morphizen/model.hpp"
#include "glog/logging.h"

#include "morphizen/env_config.hpp"
#include <vaip/vaip_ort_api.h>
DEF_ENV_PARAM(DEBUG_VAIP_MODEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_MODEL) >= n)
namespace vaip_core {
VAIP_DLL_SPEC ModelPtr model_load(const std::string& filename) {
  return ModelPtr(VAIP_ORT_API(model_load)(filename));
}

VAIP_DLL_SPEC void model_set_meta_data(Model& model, const std::string& key,
                                       const std::string& value) {
  VAIP_ORT_API(model_set_meta_data)(model, key, value);
}

VAIP_DLL_SPEC ModelPtr model_clone(const Model& model,
                                   int64_t external_data_threshold) {
#if VAIP_ORT_API_MAJOR >= 7
  return ModelPtr(VAIP_ORT_API(model_clone)(model, external_data_threshold));
#else
  return ModelPtr(VAIP_ORT_API(model_clone)(model));
#endif
}
VAIP_DLL_SPEC void ModelDeleter::operator()(Model* model) const {
  MY_LOG(1) << "destroy model(" << ((void*)model) << ") "
            << graph_get_name(model_main_graph(*model));
  VAIP_ORT_API(model_delete)(model);
}
} // namespace vaip_core

namespace vaip_cxx {
ModelConstRef::ModelConstRef(const vaip_core::Model& model) : self_(model) {}
const std::string& ModelConstRef::name() const {
  return vaip_core::graph_get_name(
      vaip_core::model_main_graph(const_cast<vaip_core::Model&>(self_)));
}
std::string ModelConstRef::get_metadata(const std::string& name) const {
  return vaip_core::model_get_meta_data(self_, name);
}
bool ModelConstRef::has_metadata(const std::string& name) const {
  return vaip_core::model_has_meta_data(self_, name);
}
std::unique_ptr<Model>
ModelConstRef::clone(int64_t external_data_threshold) const {
  return std::unique_ptr<Model>(new Model(vaip_core::model_clone(
      const_cast<onnxruntime::Model&>(self_), external_data_threshold)));
}
std::unique_ptr<Model> Model::load(const std::filesystem::path& model_path) {
  return std::unique_ptr<Model>(
      new Model(vaip_core::model_load(model_path.u8string())));
}
std::unique_ptr<Model>
Model::create(const std::filesystem::path& model_path,
              const std::vector<std::pair<std::string, int64_t>>& opset) {
  return std::unique_ptr<Model>(new Model(vaip_core::ModelPtr(
      VAIP_ORT_API(create_empty_model)(model_path, opset))));
}
Model::Model(vaip_core::ModelPtr&& ptr) : self_{std::move(ptr)} {}

Model::~Model() {
  MY_LOG(1) << "dtor " << (void*)this << "." << (void*)self_.get()
            << " name:" << ref().name();
}

Model& Model::set_metadata(const std::string& name, const std::string& value) {
  vaip_core::model_set_meta_data(*self_, name, value);
  return *this;
}

GraphRef Model::main_graph() {
  return GraphRef(vaip_core::model_main_graph(*self_));
}
GraphConstRef ModelConstRef::main_graph() const {
  return GraphConstRef(
      vaip_core::model_main_graph(const_cast<vaip_core::Model&>(self_)));
}
std::filesystem::path ModelConstRef::model_path() const {
  return VAIP_ORT_API(get_model_path)(main_graph());
}
} // namespace vaip_cxx
