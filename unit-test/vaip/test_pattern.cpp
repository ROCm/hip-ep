/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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

#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>

//
#include "debug_logger.hpp"
#include "vaip/vaip.hpp"

class PatternTest : public DebugLogger {};

static std::tuple<std::shared_ptr<vaip_core::Pattern>,
                  std::shared_ptr<vaip_core::Pattern>,
                  std::shared_ptr<vaip_core::Pattern>>
get_commutable_add_pattern() {
  auto ret = std::shared_ptr<vaip_core::Pattern>();
  vaip_core::PatternBuilder builder;
#include "pt_resnet50_add.h.inc"
  auto ret0 =
      builder.commutable_node("Add", DequantizeLinear_2, DequantizeLinear_3);
  auto ret1 =
      builder.commutable_node("Add", DequantizeLinear_3, DequantizeLinear_2);
  builder.bind("Add", ret);
  builder.bind("Add0", ret0);
  builder.bind("Add1", ret1);
  return std::make_tuple(ret, ret0, ret1);
}

TEST_F(PatternTest, CommutableNode) {
  auto model = vaip_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("287");
  auto node = graph.find_node(node_arg_name);
  ASSERT_TRUE(node.has_value()) << "cannot find node " << node_arg_name;
  auto [add, add0, add1] = get_commutable_add_pattern();
  auto binder = add->match(node.value());
  vaip_core::Pattern::enable_trace(1);
  EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";
  auto binder0 = add0->match(node.value());
  EXPECT_TRUE(binder0 != nullptr) << "cannot match the pattern";
  auto binder1 = add1->match(node.value());
  EXPECT_TRUE(binder1 != nullptr) << "cannot match the pattern";
  auto match_node_input = (*binder0)("Add0");
  ASSERT_TRUE(match_node_input.has_value());
  LOG(INFO) << "matched node arg: " << match_node_input.value();
  EXPECT_EQ(match_node_input.value().as_node_arg().name(), "287")
      << "name must be " << match_node_input.value();
  auto match_node = match_node_input.value().as_node();
  ASSERT_TRUE(match_node.has_value());
  LOG(INFO) << "matched node" << match_node.value();
  EXPECT_EQ(match_node.value().name(), "Add_178")
      << "name must be " << match_node.value();
}
/*
TEST_F(PatternTest, SequenceNode) {
  auto model = vaip_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("287");
  auto node = graph.find_node(node_arg_name);
  ASSERT_TRUE(node.has_value()) << "cannot find node " << node_arg_name;
  auto [add, add0, add1] = get_sequence_pattern();
  auto binder = add->match(node.value());
  vaip_core::Pattern::enable_trace(1);
  EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";
  auto binder0 = add0->match(node.value());
  EXPECT_TRUE(binder0 != nullptr) << "cannot match the pattern";
  auto binder1 = add1->match(node.value());
  EXPECT_TRUE(binder1 != nullptr) << "cannot match the pattern";
  auto match_node_input = (*binder0)("Add0");
  ASSERT_TRUE(match_node_input.has_value());
  LOG(INFO) << "matched node arg: " << match_node_input.value();
  EXPECT_EQ(match_node_input.value().as_node_arg().name(), "287")
      << "name must be " << match_node_input.value();
  auto match_node = match_node_input.value().as_node();
  ASSERT_TRUE(match_node.has_value());
  LOG(INFO) << "matched node" << match_node.value();
  EXPECT_EQ(match_node.value().name(), "Add_178")
      << "name must be " << match_node.value();
}
*/

TEST_F(PatternTest, LoadSaveBinary) {
  auto model = vaip_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("287");
  auto node = graph.find_node(node_arg_name);

  auto ret = std::shared_ptr<vaip_core::Pattern>();
  vaip_core::PatternBuilder builder;
#include "pt_resnet50_add.h.inc"
  builder.bind("Add", ret);
  auto encoded_pattern = ret->to_binary();
  auto new_ret = vaip_core::PatternBuilder().create_from_binary(
      encoded_pattern.data(), encoded_pattern.size());

  //
  vaip_core::Pattern::enable_trace(1);

  auto binder = ret->match(node.value());
  vaip_core::Pattern::enable_trace(1);
  EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";
  //
  auto binder2 = new_ret->match(node.value());
  EXPECT_TRUE(binder2 != nullptr) << "cannot match the pattern";
}