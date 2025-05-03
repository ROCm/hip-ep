/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>

//
#include "debug_logger.hpp"
#include "morphizen/vaip.hpp"

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
