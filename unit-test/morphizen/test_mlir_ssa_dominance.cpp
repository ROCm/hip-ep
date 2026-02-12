/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test_environment.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>

class MLIRSSADominanceTest : public ::testing::Test {};

// Test 1: Node with all graph inputs + optional (nullopt) inputs.
// This is the exact scenario that triggers the SSA dominance bug:
// - All explicit inputs are BlockArguments (graph inputs) -> no definingOp
// - latestInputOp stays nullptr -> insertion point at block start
// - nullopt inputs map to none_->getResult(0)
// - Without the fix, the node would be placed before none_, violating SSA
TEST_F(MLIRSSADominanceTest, AllGraphInputsWithOptionalNone) {
  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("ssa_test_optional.onnx");
  auto data_path = std::filesystem::path("ssa_test_optional.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create graph inputs (these become BlockArguments in MLIR - no definingOp)
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> inputs_q = {
      graph.new_node_arg("query", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> inputs_k = {
      graph.new_node_arg("key", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> inputs_v = {
      graph.new_node_arg("value", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};

  // Create output node arg
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> outputs = {
      graph.new_node_arg("output", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};

  // Build input list: 3 real graph inputs + 4 optional (nullopt -> none_)
  // This mimics GQA where past_key, past_value, seqlens_k, total_seq_len
  // are optional and may be absent.
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> node_inputs = {
      inputs_q[0],  // input 0: graph input (BlockArgument)
      inputs_k[0],  // input 1: graph input (BlockArgument)
      inputs_v[0],  // input 2: graph input (BlockArgument)
      std::nullopt, // input 3: optional (maps to none_)
      std::nullopt, // input 4: optional (maps to none_)
      std::nullopt, // input 5: optional (maps to none_)
      std::nullopt, // input 6: optional (maps to none_)
  };

  // This call triggers the SSA dominance issue without the fix:
  // All inputs are graph arguments (no definingOp), so latestInputOp ==
  // nullptr, insertion point goes to block start, but none_ operands require
  // the node to be after none_.
  graph.add_node("gqa_node", "com.microsoft", "GroupQueryAttention", "test GQA",
                 node_inputs, outputs,
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs(
      {inputs_q[0].value(), inputs_k[0].value(), inputs_v[0].value()});
  graph.set_outputs({outputs[0].value()});

  // resolve() triggers MLIR verification which checks SSA dominance.
  // Without the fix, this would fail because the GQA node uses
  // none_->getResult(0) but is placed before the none_ operation.
  graph.resolve();

  // If we reach here without assertion/crash, SSA dominance is maintained.
  LOG(INFO) << "SSA dominance test passed: node with all graph inputs + "
               "optional none_ operands";

  // Also verify save works (additional MLIR verification)
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
}

// Test 2: Node with mixed inputs (some from ops, some graph inputs, some none).
// This tests the case where latestInputOp is set but is before none_.
TEST_F(MLIRSSADominanceTest, MixedInputsWithOptionalNone) {
  auto path =
      CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("ssa_test_mixed.onnx");
  auto data_path = std::filesystem::path("ssa_test_mixed.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Graph inputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input_a = {
      graph.new_node_arg("input_a", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input_b = {
      graph.new_node_arg("input_b", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Intermediate output from a Relu node (this creates a definingOp)
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> relu_out = {
      graph.new_node_arg("relu_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Add Relu node first
  graph.add_node("relu_0", "", "Relu", "", {input_a[0]}, {relu_out[0]},
                 morphizen::NodeAttributesBuilder().build());

  // Now add a node that takes:
  // - relu_out (has a definingOp)
  // - input_b (graph input, no definingOp)
  // - nullopt (maps to none_)
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> final_out = {
      graph.new_node_arg("final_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> mixed_inputs = {
      relu_out[0],  // from Relu op (has definingOp)
      input_b[0],   // graph input (BlockArgument, no definingOp)
      std::nullopt, // optional (maps to none_)
  };

  graph.add_node("add_0", "", "Add", "", mixed_inputs, {final_out[0]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input_a[0].value(), input_b[0].value()});
  graph.set_outputs({final_out[0].value()});
  graph.resolve();

  LOG(INFO) << "SSA dominance test passed: mixed inputs with none_ operands";

  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
}

// Test 3: Multiple nodes with only graph inputs and none_ operands.
// Ensures the fix works correctly for consecutive nodes.
TEST_F(MLIRSSADominanceTest, MultipleNodesAllGraphInputsWithNone) {
  auto path =
      CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("ssa_test_multi.onnx");
  auto data_path = std::filesystem::path("ssa_test_multi.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // All graph inputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input_x = {
      graph.new_node_arg("input_x", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input_y = {
      graph.new_node_arg("input_y", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> out_1 = {
      graph.new_node_arg("out_1", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> out_2 = {
      graph.new_node_arg("out_2", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // First node: graph input + nullopt
  graph.add_node("node_1", "", "Relu", "", {input_x[0], std::nullopt},
                 {out_1[0]}, morphizen::NodeAttributesBuilder().build());

  // Second node: different graph input + nullopt
  graph.add_node("node_2", "", "Relu", "", {input_y[0], std::nullopt},
                 {out_2[0]}, morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input_x[0].value(), input_y[0].value()});
  graph.set_outputs({out_1[0].value(), out_2[0].value()});
  graph.resolve();

  LOG(INFO) << "SSA dominance test passed: multiple nodes with graph inputs "
               "and none_ operands";

  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
}
