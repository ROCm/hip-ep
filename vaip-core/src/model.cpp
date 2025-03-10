/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
// clang-format off
// must include graph.hpp first, because `main_graph` return vaip_cxx::Graph object by value.
#include "vaip/graph.hpp"
// clang-format on
#include "vaip/model.hpp"
#include "glog/logging.h"

#include "vaip/vaip_ort_api.h"
#include <vitis/ai/env_config.hpp>
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
void ModelDeleter::operator()(Model* model) const {
  MY_LOG(1) << "destroy model(" << ((void*)model) << ") "
            << VAIP_ORT_API(graph_get_name)(
                   VAIP_ORT_API(model_main_graph)(*model));
  VAIP_ORT_API(model_delete)(model);
}
} // namespace vaip_core

namespace vaip_cxx {
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
const std::string& Model::name() const {
  return VAIP_ORT_API(graph_get_name)(VAIP_ORT_API(model_main_graph)(*self_));
}

Model::~Model() {
  MY_LOG(1) << "dtor " << (void*)this << "." << (void*)self_.get()
            << " name:" << name();
}

Model& Model::set_metadata(const std::string& name, const std::string& value) {
  vaip_core::model_set_meta_data(*self_, name, value);
  return *this;
}
std::string Model::get_metadata(const std::string& name) const {
  return *VAIP_ORT_API(model_get_meta_data)(*self_, name);
}
bool Model::has_metadata(const std::string& name) const {
  return VAIP_ORT_API(model_has_meta_data)(*self_, name);
}

GraphRef Model::main_graph() {
  return GraphRef(VAIP_ORT_API(model_main_graph)(*self_));
}

const GraphRef Model::main_graph() const {
  return GraphRef(VAIP_ORT_API(model_main_graph)(*self_));
}

std::unique_ptr<Model> Model::clone(int64_t external_data_threshold) const {
  return std::unique_ptr<Model>(
      new Model(vaip_core::model_clone(*self_, external_data_threshold)));
}
} // namespace vaip_cxx
