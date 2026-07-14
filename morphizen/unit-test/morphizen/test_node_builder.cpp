/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test_environment.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
class NodeBuilderTest : public ::testing::Test {};

TEST_F(NodeBuilderTest, SkipSimplifiedLayerNormalization) {
  auto output_dir = CMAKE_CURRENT_BINARY_PATH;
  auto onnx_file = output_dir / ".." / ".." / ".." / "morphizen_regression" /
                   "llama2-7b-int4-gs128-asym-mha" / "model.onnx";
  onnx_file.make_preferred();
  if (!std::filesystem::exists(onnx_file)) {
    LOG(INFO) << "ignore SkipSimplifiedLayerNormalization";
    return;
  }
  LOG(INFO) << "found " << onnx_file;

  auto model = morphizen_cxx::Model::load(onnx_file.u8string());
  auto graph = model->main_graph();
  graph.resolve();
  std::shared_ptr<morphizen::PassContext> context =
      morphizen::PassContext::create();

  auto pass_proto = std::make_unique<morphizen::PassProto>();
  pass_proto->set_plugin("morphizen-pass_init");
  pass_proto->set_name("NodeBuilderTest.SkipSimplifiedLayerNormalization");
  auto pass = morphizen::IPass::create_pass(context, *pass_proto);

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
  auto builder = morphizen_cxx::graph_node_builder(graph, *pass);

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
