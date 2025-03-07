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

#pragma once
#include "./_sanity_check.hpp"
#if __has_include(<filesystem>)  && __cplusplus > 201700
#  include <filesystem>
#else
#  error "must enable c++17"
#endif

#include <vaip/custom_op.h>
#include <vaip/export.h>
#include <vaip/vaip_ort_api.h>
/// header file used by ort VITISAI execution providers.

using EventInfo = std::tuple<std::string, // name
                             int,         // pid
                             int,         // tid
                             long long,   // timestamp
                             long long    // duration
                             >;

namespace onnxruntime {
using ProviderOptions = std::unordered_map<std::string, std::string>;
}
namespace vaip_core {
class PassContextImp;
using vaip_error_report_func = void (*)(
    void*, int, const char*); // should be same as vaip_core::error_report_func,
                              // for compile issue we  defined it

VAIP_DLL_SPEC void
initialize_onnxruntime_vitisai_ep(OrtApiForVaip* api,
                                  std::vector<OrtCustomOpDomain*>& ret_domain);

VAIP_DLL_SPEC void deinitialize_onnxruntime_vitisai_ep();

VAIP_DLL_SPEC void set_the_global_api(OrtApiForVaip* api);

VAIP_DLL_SPEC const OrtApiForVaip* api();

VAIP_DLL_SPEC std::vector<std::unique_ptr<ExecutionProvider>>
compile_onnx_model_5(const std::filesystem::path& model_path,
                     const Graph& graph,
                     const onnxruntime::ProviderOptions& options, void* status,
                     vaip_error_report_func func);

VAIP_DLL_SPEC void profiler_collect(std::vector<EventInfo>& api_events,
                                    std::vector<EventInfo>& kernel_events);

int optimize_onnx_model(const std::filesystem::path& model_path_in,
                        const std::filesystem::path& model_path_out,
                        const char* json_config);

void initialize_graph_optimizer(const std::string& json_path);

std::shared_ptr<PassContextImp>
initialize_context(const std::string& model_path, const Graph& onnx_graph,
                   const char* json_config);

VAIP_DLL_SPEC void
get_compilation_cache(const std::string& model_path, const Graph& graph,
                      const char* json_config, uint8_t compiler_codes,
                      std::string& cache_dir, std::string& cache_key,
                      std::string& cache_data);

VAIP_DLL_SPEC void restore_compilation_cache(const std::string& cache_dir,
                                             const std::string& cache_key,
                                             const std::string& cache_data,
                                             const std::string& model_path);
} // namespace vaip_core
