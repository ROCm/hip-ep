/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
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

#include "debug_logger.hpp"
#include "unit_test_env_params.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
//
#include "vaip/vaip.hpp"
class NodeBuilderTest : public DebugLogger {};

TEST_F(NodeBuilderTest, SkipSimplifiedLayerNormalization) {
  auto output_dir = std::filesystem::path(ENV_PARAM(CMAKE_CURRENT_BINARY_DIR));
  auto onnx_file = output_dir / ".." / ".." / ".." / "vaip_regression" /
                   "llama2-7b-int4-gs128-asym-mha" / "model.onnx";
  onnx_file.make_preferred();
  if (!std::filesystem::exists(onnx_file)) {
    LOG(INFO) << "ignore SkipSimplifiedLayerNormalization";
    return;
  }
  LOG(INFO) << "found " << onnx_file;

  auto model = vaip_cxx::Model::load(onnx_file.u8string());
  auto graph = model->main_graph();
  graph.resolve();
  std::shared_ptr<vaip_core::PassContext> context =
      vaip_core::PassContext::create();

  auto pass_proto = std::make_unique<vaip_core::PassProto>();
  pass_proto->set_plugin("vaip-pass_init");
  pass_proto->set_name("NodeBuilderTest.SkipSimplifiedLayerNormalization");
  auto pass = vaip_core::IPass::create_pass(context, *pass_proto);

  auto node_arg0 =
      graph.find_node_arg("/model/layers.0/post_attention_layernorm/output_0");
  auto node_arg1 =
      graph.find_node_arg("/model/layers.1/post_attention_layernorm/output_0");
  ASSERT_TRUE(node_arg0.has_value());
  ASSERT_TRUE(node_arg1.has_value());
  auto node_0 = node_arg0.value().find_producer();
  auto node_1 = node_arg1.value().find_producer();
  ASSERT_TRUE(node_0.has_value());
  ASSERT_TRUE(node_1.has_value());
  auto node_1_outputs = node_1.value().outputs();
  ASSERT_EQ(node_1_outputs.size(), 4u)
      << "node 1 must have four outputs. " << node_1.value();
  ASSERT_TRUE(node_1_outputs[0].has_value());
  ASSERT_TRUE(!node_1_outputs[1].has_value());
  ASSERT_TRUE(!node_1_outputs[2].has_value());
  ASSERT_TRUE(node_1_outputs[3].has_value());
  auto builder = graph.node_builder(*pass);

  auto newly_node = builder.clone_inputs(node_0.value())
                        .clone_op_type(node_1.value())
                        .clone_attrs(node_1.value())
                        // arg 0
                        .set_anchor_point1(node_1_outputs[0].value())
                        .add_output() // arg 1
                        .skip_optional_output()
                        .add_output() // arg 2
                        .skip_optional_output()
                        .add_output() // arg3
                        .set_anchor_point1(node_1_outputs[3].value())
                        .build_ex();
  LOG(INFO) << "newly_node = " << newly_node;
  /*graph.save((output_dir / "llama2-7b-int4-gs128-asym-mha.onnx").u8string(),
             "llama2-7b-int4-gs128-asym-mha.dat", 128u);*/
}
