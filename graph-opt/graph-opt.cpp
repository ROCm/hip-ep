/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *Copyright (C) 2023 – 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <exception>
#include <filesystem>
#include <glog/logging.h>

#include "initialize_vaip.hpp"
#include <limits>

//
#include "morphizen/vaip.hpp"
//

extern "C" {
#include "getopt.h"
}

using namespace vaip_core;
using namespace std;
int main(int argc, char* argv[]) {
  try {
    auto opt_input_file = std::string();
    auto opt_cache = std::string();
    auto opt_output_file = std::string();
    auto opt_output_txt_file = std::string();
    auto opt_module_name = std::string();
    auto opt_method_name = std::string("rules");
    auto opt_pass = std::vector<std::string>();
    int opt = 0;
    while ((opt = getopt(argc, argv, "m:i:f:o:t:c:p:")) != -1) {
      switch (opt) {
      case 'i': {
        opt_input_file = std::string(optarg);
        break;
      }
      case 'o': {
        opt_output_file = std::string(optarg);
        break;
      }
      case 't': {
        opt_output_txt_file = std::string(optarg);
        break;
      }
      case 'm': {
        opt_module_name = std::string(optarg);
        break;
      }
      case 'f': {
        opt_method_name = std::string(optarg);
        break;
      }
      case 'p': {
        opt_pass.push_back(std::string(optarg));
        break;
      }
      case 'c': {
        opt_cache = std::string(optarg);
        break;
      }
      }
    }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "voe_py_pass");
    initialize_vaip();
    std::shared_ptr<PassContext> context = PassContext::create();
    if (!opt_cache.empty()) {
      context = load_context(opt_cache);
    }
    auto protos = std::vector<std::unique_ptr<PassProto>>{};
    auto passes = std::vector<std::shared_ptr<vaip_core::IPass>>{};
    for (auto& opt_pass_i : opt_pass) {
      protos.emplace_back(std::make_unique<PassProto>());
      auto& pass_proto = *protos.back();
      pass_proto.set_name("test");
      pass_proto.set_plugin(opt_pass_i);
      passes.emplace_back(IPass::create_pass(context, pass_proto));
    }

    auto model = vaip_core::model_load(opt_input_file);
    auto& graph = VAIP_ORT_API(model_main_graph)(*model);
    graph_resolve(graph);

    IPass::run_passes(passes, graph);

    if (!opt_output_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_file;
      VAIP_ORT_API(graph_save)
      (graph, opt_output_file, opt_output_file + ".dat",
       std::numeric_limits<size_t>::max());
    }
    if (!opt_output_txt_file.empty()) {
      LOG(INFO) << "write output file to " << opt_output_txt_file;
      vaip_core::dump_graph(graph, opt_output_txt_file);
    }
  } catch (const std::exception& e) {
    std::cerr << "exception occurs : " << e.what() << "\n";
  }
  return 0;
}

#include "getopt.c"
