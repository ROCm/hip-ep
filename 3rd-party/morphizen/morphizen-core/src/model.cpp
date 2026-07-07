/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// clang-format off
// must include graph.hpp first, because `main_graph` return morphizen_cxx::Graph object by value.
#include "morphizen/graph.hpp"
// clang-format on
#include "morphizen/model.hpp"
#include "glog/logging.h"

#include "morphizen/env_config.hpp"
#include <morphizen/morphizen_ort_api.h>
DEF_ENV_PARAM(DEBUG_MORPHIZEN_MODEL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_MODEL) >= n)
namespace morphizen {
MORPHIZEN_DLL_SPEC ModelPtr model_load(const std::string &filename) {
  return ModelPtr(MORPHIZEN_ORT_API(model_load)(filename));
}

MORPHIZEN_DLL_SPEC void model_set_meta_data(Model &model,
                                            const std::string &key,
                                            const std::string &value) {
  MORPHIZEN_ORT_API(model_set_meta_data)(model, key, value);
}

MORPHIZEN_DLL_SPEC ModelPtr model_clone(const Model &model,
                                        int64_t external_data_threshold) {
#if MORPHIZEN_ORT_API_MAJOR >= 7
  return ModelPtr(
      MORPHIZEN_ORT_API(model_clone)(model, external_data_threshold));
#else
  return ModelPtr(MORPHIZEN_ORT_API(model_clone)(model));
#endif
}
MORPHIZEN_DLL_SPEC void ModelDeleter::operator()(Model *model) const {
  auto &main_graph = model_main_graph(*model);
  MY_LOG(1) << "destroy model(" << ((void *)model) << ") "
            << morphizen_cxx::GraphConstRef(main_graph).name();
  MORPHIZEN_ORT_API(model_delete)(model);
}
} // namespace morphizen

namespace morphizen_cxx {
ModelConstRef::ModelConstRef(const morphizen::Model &model) : self_(model) {}
const std::string &ModelConstRef::name() const {
  auto &main_graph =
      morphizen::model_main_graph(const_cast<morphizen::Model &>(self_));
  return morphizen_cxx::GraphConstRef(main_graph).name();
}
std::string ModelConstRef::get_metadata(const std::string &name) const {
  return morphizen::model_get_meta_data(self_, name);
}
bool ModelConstRef::has_metadata(const std::string &name) const {
  return morphizen::model_has_meta_data(self_, name);
}
std::unique_ptr<Model>
ModelConstRef::clone(int64_t external_data_threshold) const {
  return std::unique_ptr<Model>(new Model(morphizen::model_clone(
      const_cast<onnxruntime::Model &>(self_), external_data_threshold)));
}
std::unique_ptr<Model> Model::load(const std::filesystem::path &model_path) {
  return std::unique_ptr<Model>(
      new Model(morphizen::model_load(model_path.u8string())));
}
std::unique_ptr<Model>
Model::create(const std::filesystem::path &model_path,
              const std::vector<std::pair<std::string, int64_t>> &opset) {
  return std::unique_ptr<Model>(new Model(morphizen::ModelPtr(
      MORPHIZEN_ORT_API(create_empty_model)(model_path, opset))));
}
Model::Model(morphizen::ModelPtr &&ptr) : self_{std::move(ptr)} {}

Model::~Model() {
  MY_LOG(1) << "dtor " << (void *)this << "." << (void *)self_.get()
            << " name:" << ref().name();
}

Model &Model::set_metadata(const std::string &name, const std::string &value) {
  morphizen::model_set_meta_data(*self_, name, value);
  return *this;
}

GraphRef Model::main_graph() {
  return GraphRef(morphizen::model_main_graph(*self_));
}
GraphConstRef ModelConstRef::main_graph() const {
  return GraphConstRef(
      morphizen::model_main_graph(const_cast<morphizen::Model &>(self_)));
}
std::filesystem::path ModelConstRef::model_path() const {
  return MORPHIZEN_ORT_API(get_model_path)(main_graph());
}
} // namespace morphizen_cxx
