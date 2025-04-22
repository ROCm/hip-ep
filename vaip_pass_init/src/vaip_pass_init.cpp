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

#include <glog/logging.h>

#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
using namespace vaip_core;
DEF_ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL, "0")

struct InitPass {
  InitPass(IPass& self) {}
  void process(IPass& self, Graph& graph) {
    if (ENV_PARAM(XLNX_ENABLE_DUMP_ONNX_MODEL)) {
      auto log_dir = self.get_log_path();
      if (log_dir.empty()) {
        LOG(WARNING) << "log dir is empty, call saving onnx.onnx";
        return;
      }
      auto file = log_dir / "onnx.onnx";
      auto dat_file = "onnx.dat";
      vaip_cxx::GraphConstRef(graph).save(file.u8string(), dat_file,
                                          std::numeric_limits<size_t>::max());
      LOG(INFO) << "save origin onnx model to " << file << " data in "
                << dat_file;
    }
    return;
  }
};

DEFINE_VAIP_PASS(InitPass, vaip_pass_init)
