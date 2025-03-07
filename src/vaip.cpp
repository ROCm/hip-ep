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

#include "./xir_ops/xir_ops_defs.hpp"
//
#include <exception>
#include <glog/logging.h>
// include glog/logging.h to define CHECK before include vaip_plugin.hpp

#include "./config.hpp"
#include "./pass_imp.hpp"
#include "vaip/transpose.hpp"
#include "vaip/util.hpp"
#include "vaip/vaip_ort.hpp"
#include "vaip/vaip_plugin.hpp"
#include "version_info.hpp"
#include "vitis/ai/env_config.hpp"
#include <vaip/custom_op.h>
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>

// #include "core/common/status.h"
#include <memory>
#include <xir/graph/graph.hpp>

namespace vaip_core {
// TODO: defined vitisai_compile_model.cpp
void compile_onnx_model_2(std::shared_ptr<PassContextImp> context,
                          onnxruntime::Graph& graph);

OrtApiForVaip* the_global_api = nullptr;
const OrtApiForVaip& __api() {
  DCHECK(the_global_api != nullptr)
      << "please set_the_global_api() before invoking this function";
  return *the_global_api;
}

void AttributeProtoDeleter::operator()(AttributeProto* p) const {
  VAIP_ORT_API(attr_proto_delete)(p);
}

void NodeAttributesDeleter::operator()(NodeAttributes* p) const {
  VAIP_ORT_API(node_attributes_delete)(p);
}

VAIP_DLL_SPEC void
initialize_onnxruntime_vitisai_ep(OrtApiForVaip* api,
                                  std::vector<OrtCustomOpDomain*>& ret_domain) {
  vaip_core::set_the_global_api(api);
  ret_domain.emplace_back(vaip_core::register_xir_ops());

  return;
}

VAIP_DLL_SPEC void deinitialize_onnxruntime_vitisai_ep() {
  LOG(INFO) << "deinitialize_onnxruntime_vitisai_ep";
  deinitialize_transpose();
}

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

VAIP_DLL_SPEC const OrtApiForVaip* api() { return &__api(); }

VAIP_DLL_SPEC AttributeProtoPtr attr_proto_clone(const AttributeProto& attr) {
  return AttributeProtoPtr(VAIP_ORT_API(attr_proto_clone)(attr));
}

VAIP_DLL_SPEC AttributeProtoPtr attr_proto_new_ints(
    const std::string& name, const std::vector<int64_t>& value) {
  return AttributeProtoPtr(VAIP_ORT_API(attr_proto_new_ints)(name, value));
}

VAIP_DLL_SPEC NodeAttributesPtr node_attributes_new() {
  return NodeAttributesPtr(VAIP_ORT_API(node_attributes_new)());
}

VAIP_DLL_SPEC NodeAttributesPtr node_clone_attributes(const Node& node) {
  auto ret = node_attributes_new();
  for (auto& attr : node_get_attributes(node)) {
    auto cloned_attr = attr_proto_clone(*attr);
    VAIP_ORT_API(node_attributes_add)(*ret, std::move(*cloned_attr));
  }
  return ret;
}
} // namespace vaip_core
