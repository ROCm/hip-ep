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
#define _CRT_SECURE_NO_WARNINGS 1
#include <gtest/gtest.h>
#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include "debug_logger.hpp"
#include <glog/logging.h>
#include <onnxruntime_cxx_api.h>
#if _WIN32
#  ifdef _DEBUG
#    include <crtdbg.h>
#  endif
#endif
#include "morphizen/vaip.hpp"
template <typename... Args> void* morphizen_main_cmd(Args... args) {
  auto ep_dll = vaip_core::Plugin::get("onnxruntime_vitisai_ep");
  if (ep_dll == nullptr) {
    LOG(ERROR) << "Failed to load Vitis AI EP";
    return nullptr;
  }
  const char* argv[] = {
      args...,
  };
  int argc = (int)(sizeof(argv) / sizeof(argv[0]));
  return ep_dll->get_method<void*, int, const char*[]>("morphizen_main")(argc,
                                                                         argv);
}

namespace gtest_example {
TEST(GTest, hello) { LOG(INFO) << "Hello GTest"; }
} // namespace gtest_example

int main(int argc, char** argv) {
#if _WIN32
#  ifdef _DEBUG
  auto env_ci = getenv("CI");
  auto ci = std::string(env_ci ? env_ci : "");
  if (ci == "1") {
    // Disable assertion dialog in CI
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    const char* cmd[] = {
        "disable_crt_diag",
    };
    morphizen_main_cmd("disable_crt_diag");
  }
#  endif
#endif
  {
    auto env =
        std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "vaip_unit_test");
    vaip_core::StaticPluginRegister::sync_static_plugin_into_module(
        "onnxruntime_vitisai_ep");

    Ort::SessionOptions().AppendExecutionProvider_VitisAI();
    vaip_core::set_the_global_api(
        vaip_core::Plugin::invoke<vaip_core::OrtApiForVaip*>(
            "onnxruntime_vitisai_ep", "get_the_global_api"));
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
}
