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

//
#include <exception>
#include <glog/logging.h>
// include glog/logging.h to define CHECK before include vaip_plugin.hpp

#include "./config.hpp"
#include "./pass_imp.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/transpose.hpp"
#include "morphizen/util.hpp"
#include "morphizen/vaip_ort.hpp"
#include "morphizen/vaip_plugin.hpp"
#include "version_info.hpp"
#include <vaip/custom_op.h>
#include <vaip/my_ort.h>
#include <vaip/vaip_ort_api.h>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_DEINITIALIZE) >= n)

#include <memory>

namespace vaip_core {
// TODO: defined vitisai_compile_model.cpp
void compile_onnx_model_2(std::shared_ptr<PassContextImp> context,
                          onnxruntime::Graph& graph);

VAIP_DLL_SPEC void
initialize_onnxruntime_vitisai_ep(OrtApiForVaip* api,
                                  std::vector<OrtCustomOpDomain*>& ret_domain) {
  vaip_core::set_the_global_api(api);

  return;
}
namespace {
std::vector<std::pair<std::string, std::function<void()>>> g_at_exits;
}

void add_cleanup_function(const std::string& name,
                          std::function<void()> cleanup_function) {
  g_at_exits.emplace_back(name, cleanup_function);
}

VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep() {
  MY_LOG(1) << "deinitialize_onnxruntime_vitisai_ep";
  deinitialize_transpose();
  for (auto& item : g_at_exits) {
    MY_LOG(1) << " deinitialize " << item.first;
  }
  g_at_exits.clear();
}
} // namespace vaip_core
