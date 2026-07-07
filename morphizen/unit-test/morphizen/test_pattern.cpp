/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test_environment.hpp"
#include <filesystem>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
//
#include "morphizen/morphizen.hpp"

class PatternTest : public ::testing::Test {};

static std::tuple<std::shared_ptr<morphizen::Pattern>,
                  std::shared_ptr<morphizen::Pattern>,
                  std::shared_ptr<morphizen::Pattern>>
get_commutable_add_pattern() {
  auto ret = std::shared_ptr<morphizen::Pattern>();
  morphizen::PatternBuilder builder;
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
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("287");
  auto node = graph.find_node(node_arg_name);
  ASSERT_TRUE(node.has_value()) << "cannot find node " << node_arg_name;
  auto [add, add0, add1] = get_commutable_add_pattern();
  auto binder = add->match(node.value());
  morphizen::Pattern::enable_trace(1);
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

#if MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT
TEST_F(PatternTest, NamedArgs) {
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("127"); // Conv output
  auto node = graph.find_node(node_arg_name);
  ASSERT_TRUE(node.has_value()) << "cannot find node " << node_arg_name;

  morphizen::Pattern::enable_trace(1);
  morphizen::PatternBuilder builder;
  auto X = builder.wildcard();
  auto W = builder.wildcard();
  auto B = builder.wildcard();
  auto conv =
      builder.node_with_named_args("Conv", {{"X", X}, {"W", W}, {"B", B}});
  ASSERT_TRUE(conv != nullptr) << "cannot build Conv pattern";
  builder.bind("conv_0", conv);

  auto binder = conv->match(node.value());
  EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";

  auto match_node_input = (*binder)("conv_0");
  ASSERT_TRUE(match_node_input.has_value());
  LOG(INFO) << "matched node arg: " << match_node_input.value();
  EXPECT_EQ(match_node_input.value().as_node_arg().name(), "127")
      << "name must be " << match_node_input.value();

  auto match_node = match_node_input.value().as_node();
  ASSERT_TRUE(match_node.has_value());
  LOG(INFO) << "matched node" << match_node.value();
  EXPECT_EQ(match_node.value().name(), "Conv_18")
      << "name must be " << match_node.value();
}
#endif // MORPHIZEN_HAS_ONNX_SCHEMA_SUPPORT

static std::shared_ptr<morphizen::Pattern>
save_and_load_pattern(std::shared_ptr<morphizen::Pattern> pattern) {
  auto encoded_pattern = pattern->to_binary();
  return morphizen::PatternBuilder().create_from_binary(encoded_pattern.data(),
                                                        encoded_pattern.size());
}

TEST_F(PatternTest, MultipleOutputs) {
  // Test description patterns: output_arg and is_graph_output
  // Creates a model with LayerNormalization (3 outputs) -> Relu (graph outputs)
  //
  // Graph structure:
  //   LayerNormalization (outputs: output_0, output_1, output_2)
  //     -> Relu (g_output_0) [graph output]
  //     -> Relu (g_output_1) [graph output]
  //     -> Relu (g_output_2) [graph output]

  // create test model
  auto path =
      CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("multi_outputs.onnx");
  auto data_path = std::filesystem::path("multi_outputs.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"test", 1}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  std::vector<std::vector<int64_t>> i_shapes = {{8}, {1}, {1}};
  std::vector<std::vector<int64_t>> o_shapes = {{8}, {1}, {1}};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> i = {
      graph.new_node_arg("input_0", i_shapes[0],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("input_1", i_shapes[1],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("input_2", i_shapes[2],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> o = {
      graph.new_node_arg("output_0", o_shapes[0],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("output_1", o_shapes[1],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("output_2", o_shapes[2],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> g_o = {
      graph.new_node_arg("g_output_0", o_shapes[0],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("g_output_1", o_shapes[1],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
      graph.new_node_arg("g_output_2", o_shapes[2],
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  graph.add_node("node_0", "", "LayerNormalization", "", i, o,
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("node_1", "", "Relu", "", {o[0]}, {g_o[0]},
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("node_2", "", "Relu", "", {o[1]}, {g_o[1]},
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("node_3", "", "Relu", "", {o[2]}, {g_o[2]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({i[0].value(), i[1].value(), i[2].value()});
  graph.set_outputs({g_o[0].value(), g_o[1].value(), g_o[2].value()});
  graph.resolve();

  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));

  {
    SCOPED_TRACE("OutputArg");

    morphizen::PatternBuilder builder;
    auto node_outputs = builder.node_with_multiple_outputs(
        "LayerNormalization",
        {builder.wildcard(), builder.wildcard(), builder.wildcard()},
        {false, false, false}, "", 3);

    std::vector<std::string> output_names = {"output_0", "output_1",
                                             "output_2"};
    std::vector<std::string> g_output_names = {"g_output_0", "g_output_1",
                                               "g_output_2"};

    for (size_t i = 0; i < g_output_names.size(); ++i) {
      for (size_t j = 0; j < g_output_names.size(); ++j) {
        auto node = graph.find_node(g_output_names[i]);
        ASSERT_TRUE(node.has_value())
            << "cannot find node " << g_output_names[i];

        auto p_output_arg = node_outputs.at(j);
        auto g_output = builder.node2("Relu", {p_output_arg});
        auto binder = g_output->match(node.value());

        auto g_output_2 = save_and_load_pattern(g_output);
        auto binder_2 = g_output_2->match(node.value());

        if (i == j) {
          ASSERT_TRUE(binder != nullptr) << "cannot match the pattern";
          ASSERT_TRUE(binder_2 != nullptr) << "cannot match the pattern";

          auto match_node_input = (*binder)(p_output_arg->get_id());
          ASSERT_TRUE(match_node_input.has_value());
          EXPECT_EQ(match_node_input.value().as_node_arg().name(),
                    output_names[j])
              << "name must be " << output_names[j];

          auto match_node_input_2 =
              (*binder_2)(std::to_string(p_output_arg->get_id()));
          ASSERT_TRUE(match_node_input_2.has_value());
          EXPECT_EQ(match_node_input_2.value().as_node_arg().name(),
                    output_names[j])
              << "name must be " << output_names[j];
        } else {
          EXPECT_TRUE(binder == nullptr) << "should not match the pattern";
          EXPECT_TRUE(binder_2 == nullptr) << "should not match the pattern";
        }
      }
    }
  }

  {
    SCOPED_TRACE("GraphOutput");

    morphizen::PatternBuilder builder;
    auto node_outputs = builder.node_with_multiple_outputs(
        "LayerNormalization",
        {builder.wildcard(), builder.wildcard(), builder.wildcard()},
        {false, false, false}, "", 3);

    std::vector<std::string> g_output_names = {"g_output_0", "g_output_1",
                                               "g_output_2"};

    {
      SCOPED_TRACE("all");
      // Test is_graph_output() without index/name: should match any graph
      // output

      for (size_t i = 0; i < g_output_names.size(); ++i) {
        auto node = graph.find_node(g_output_names[i]);
        ASSERT_TRUE(node.has_value())
            << "cannot find node " << g_output_names[i];

        auto p_output_arg = node_outputs.at(i);
        auto relu_output = builder.node2("Relu", {p_output_arg});
        auto g_output = builder.is_graph_output(relu_output);
        auto binder = g_output->match(node.value());

        auto g_output_2 = save_and_load_pattern(g_output);
        auto binder_2 = g_output_2->match(node.value());

        EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";
        EXPECT_TRUE(binder_2 != nullptr) << "cannot match the pattern";
      }
    }

    {
      SCOPED_TRACE("index");
      // Test is_graph_output() with index: should match only the specified
      // graph output index

      for (size_t i = 0; i < g_output_names.size(); ++i) {
        for (size_t j = 0; j < g_output_names.size(); ++j) {
          auto node = graph.find_node(g_output_names[i]);
          ASSERT_TRUE(node.has_value())
              << "cannot find node " << g_output_names[i];

          auto p_output_arg = node_outputs.at(i);
          auto relu_output = builder.node2("Relu", {p_output_arg});
          auto g_output = builder.is_graph_output(relu_output, j);
          auto binder = g_output->match(node.value());

          auto g_output_2 = save_and_load_pattern(g_output);
          auto binder_2 = g_output_2->match(node.value());

          if (i == j) {
            ASSERT_TRUE(binder != nullptr) << "cannot match the pattern";
            ASSERT_TRUE(binder_2 != nullptr) << "cannot match the pattern";

            auto match_node_input = (*binder)(g_output->get_id());
            ASSERT_TRUE(match_node_input.has_value());
            EXPECT_EQ(match_node_input.value().as_node_arg().name(),
                      g_output_names[j])
                << "name must be " << g_output_names[j];

            auto match_node_input_2 = (*binder_2)(g_output_2->get_id());
            ASSERT_TRUE(match_node_input_2.has_value());
            EXPECT_EQ(match_node_input_2.value().as_node_arg().name(),
                      g_output_names[j])
                << "name must be " << g_output_names[j];
          } else {
            EXPECT_TRUE(binder == nullptr) << "should not match the pattern";
            EXPECT_TRUE(binder_2 == nullptr) << "should not match the pattern";
          }
        }
      }
    }

    {
      SCOPED_TRACE("name");
      // Test is_graph_output() with name: should match only the specified
      // graph output name

      for (size_t i = 0; i < g_output_names.size(); ++i) {
        for (size_t j = 0; j < g_output_names.size(); ++j) {
          auto node = graph.find_node(g_output_names[i]);
          ASSERT_TRUE(node.has_value())
              << "cannot find node " << g_output_names[i];

          auto p_output_arg = node_outputs.at(i);
          auto relu_output = builder.node2("Relu", {p_output_arg});
          auto g_output =
              builder.is_graph_output(relu_output, g_output_names[j]);
          auto binder = g_output->match(node.value());

          auto g_output_2 = save_and_load_pattern(g_output);
          auto binder_2 = g_output_2->match(node.value());

          if (i == j) {
            ASSERT_TRUE(binder != nullptr) << "cannot match the pattern";
            ASSERT_TRUE(binder_2 != nullptr) << "cannot match the pattern";

            auto match_node_input = (*binder)(g_output->get_id());
            ASSERT_TRUE(match_node_input.has_value());
            EXPECT_EQ(match_node_input.value().as_node_arg().name(),
                      g_output_names[j])
                << "name must be " << g_output_names[j];

            auto match_node_input_2 = (*binder_2)(g_output_2->get_id());
            ASSERT_TRUE(match_node_input_2.has_value());
            EXPECT_EQ(match_node_input_2.value().as_node_arg().name(),
                      g_output_names[j])
                << "name must be " << g_output_names[j];
          } else {
            EXPECT_TRUE(binder == nullptr) << "should not match the pattern";
            EXPECT_TRUE(binder_2 == nullptr) << "should not match the pattern";
          }
        }
      }
    }
  }
}

TEST_F(PatternTest, LoadSaveBinary) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  // TODO(Issue #058): MLIR model node names differ from ONNX
  GTEST_SKIP() << "MLIR backend: node names mismatch (Issue #058)";
#endif
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();
  auto node_arg_name = std::string("287");
  auto node = graph.find_node(node_arg_name);

  auto ret = std::shared_ptr<morphizen::Pattern>();
  morphizen::PatternBuilder builder;
#include "pt_resnet50_add.h.inc"
  builder.bind("Add", ret);
  auto encoded_pattern = ret->to_binary();
  auto new_ret = morphizen::PatternBuilder().create_from_binary(
      encoded_pattern.data(), encoded_pattern.size());

  //
  morphizen::Pattern::enable_trace(1);

  auto binder = ret->match(node.value());
  morphizen::Pattern::enable_trace(1);
  EXPECT_TRUE(binder != nullptr) << "cannot match the pattern";
  //
  auto binder2 = new_ret->match(node.value());
  EXPECT_TRUE(binder2 != nullptr) << "cannot match the pattern";
}
