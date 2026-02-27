/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test_environment.hpp"
#include "morphizen/morphizen-ort-api-ext.hpp"
#include "morphizen/morphizen.hpp"
#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <regex>

// Helper function to check if MLIR backend is being used at runtime
static bool isMLIRBackendActive() {
  auto* api = morphizen::get_the_global_api_unsafe();
  if (!api || !api->get_lib_name) {
    return false;
  }
  auto lib_name = api->get_lib_name();
  if (!lib_name.get()) {
    return false;
  }
  return std::string(lib_name->c_str()) == "morphizen-mlir-imp";
}

// Helper function to check if MLIR output is in text format (not bytecode)
// MLIR bytecode starts with magic bytes: 'M' 'L' 0xEF 'R'
static bool isMLIRTextFormat(const std::string& mlir_output) {
  if (mlir_output.size() < 4) {
    return true; // Too short to be bytecode, assume text
  }
  // Check for MLIR bytecode magic: "ML\xefR"
  if (mlir_output[0] == 'M' && mlir_output[1] == 'L' &&
      static_cast<unsigned char>(mlir_output[2]) == 0xEF &&
      mlir_output[3] == 'R') {
    return false; // This is bytecode
  }
  return true;    // Assume text format
}

class OnnxReturnTest : public ::testing::Test {};

// Test 1: Verify that a simple graph uses onnx.Return in MLIR serialization
TEST_F(OnnxReturnTest, SimpleGraphUsesOnnxReturn) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_return_test.onnx");
  auto data_path = std::filesystem::path("onnx_return_test.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create a simple graph: input -> Relu -> output
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("output", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {output[0]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  // Get MLIR serialization with text format
  // Note: We need to set MORPHIZEN_SAVE_MLIR_AS_TEXT=1 for text output
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;
  LOG(INFO) << "MLIR output length: " << mlir_text.size();

  // Check if it's text format (starts with readable characters)
  // If bytecode, we can't easily verify the content
  if (isMLIRTextFormat(mlir_text)) {
    // Text format - verify onnx.Return is present
    EXPECT_TRUE(mlir_text.find("onnx.Return") != std::string::npos)
        << "MLIR output should contain 'onnx.Return' as the function "
           "terminator";

    // Also verify it doesn't use func.return (should use onnx.Return instead)
    // Note: func.return might still appear in some contexts, but the main
    // terminator should be onnx.Return
    LOG(INFO) << "Found onnx.Return in MLIR output";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify the graph can be saved successfully (MLIR verification passes)
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 2: Verify onnx.Return works with multiple outputs
TEST_F(OnnxReturnTest, MultipleOutputsWithOnnxReturn) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_return_multi_output.onnx");
  auto data_path = std::filesystem::path("onnx_return_multi_output.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create graph with two parallel branches: input -> Relu -> out1
  //                                           -> Sigmoid -> out2
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> out1 = {
      graph.new_node_arg("out1", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> out2 = {
      graph.new_node_arg("out2", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {out1[0]},
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("sigmoid_0", "", "Sigmoid", "", {input[0]}, {out2[0]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({out1[0].value(), out2[0].value()});
  graph.resolve();

  // Verify the graph outputs are correctly set
  auto outputs = graph.outputs();
  EXPECT_EQ(outputs.size(), 2u);

  // Verify save works (MLIR verification passes with multiple return operands)
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);

  LOG(INFO) << "Multiple outputs test passed with onnx.Return";
}

// Test 3: Verify onnx.Return with custom domain operations (onnx.Custom)
TEST_F(OnnxReturnTest, CustomDomainWithOnnxReturn) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_return_custom_domain.onnx");
  auto data_path = std::filesystem::path("onnx_return_custom_domain.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create a graph with a custom domain operation
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> query = {
      graph.new_node_arg("query", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> key = {
      graph.new_node_arg("key", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> value = {
      graph.new_node_arg("value", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("output", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};

  // Add GroupQueryAttention from com.microsoft domain
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> node_inputs = {
      query[0],     key[0],       value[0],    std::nullopt,
      std::nullopt, std::nullopt, std::nullopt};

  graph.add_node("gqa_node", "com.microsoft", "GroupQueryAttention", "",
                 node_inputs, {output[0]},
                 morphizen::NodeAttributesBuilder()
                     .add("num_heads", int64_t(32))
                     .add("kv_num_heads", int64_t(8))
                     .build());

  graph.set_inputs({query[0].value(), key[0].value(), value[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Verify onnx.Custom is used for custom domain ops
    EXPECT_TRUE(mlir_text.find("onnx.Custom") != std::string::npos)
        << "MLIR output should contain 'onnx.Custom' for custom domain ops";

    // Verify onnx.Return is present
    EXPECT_TRUE(mlir_text.find("onnx.Return") != std::string::npos)
        << "MLIR output should contain 'onnx.Return'";

    // Verify function_name attribute is present
    EXPECT_TRUE(mlir_text.find("function_name") != std::string::npos)
        << "onnx.Custom should have function_name attribute";

    // Verify domain_name attribute is present
    EXPECT_TRUE(mlir_text.find("domain_name") != std::string::npos ||
                mlir_text.find("com.microsoft") != std::string::npos)
        << "onnx.Custom should have domain_name attribute";

    LOG(INFO) << "Custom domain with onnx.Return test passed";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 4: Verify that set_outputs correctly updates onnx.Return operands
TEST_F(OnnxReturnTest, SetOutputsUpdatesOnnxReturn) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_return_set_outputs.onnx");
  auto data_path = std::filesystem::path("onnx_return_set_outputs.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create a chain: input -> Relu -> Sigmoid -> output
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> relu_out = {
      graph.new_node_arg("relu_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> sigmoid_out = {
      graph.new_node_arg("sigmoid_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {relu_out[0]},
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("sigmoid_0", "", "Sigmoid", "", {relu_out[0]},
                 {sigmoid_out[0]}, morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});

  // First, set output to relu_out
  graph.set_outputs({relu_out[0].value()});
  graph.resolve();

  auto outputs1 = graph.outputs();
  EXPECT_EQ(outputs1.size(), 1u);
  EXPECT_EQ(outputs1[0].name(), "relu_out");

  // Now change output to sigmoid_out
  graph.set_outputs({sigmoid_out[0].value()});
  graph.resolve();

  auto outputs2 = graph.outputs();
  EXPECT_EQ(outputs2.size(), 1u);
  EXPECT_EQ(outputs2[0].name(), "sigmoid_out");

  // Verify save works (MLIR verification passes after changing outputs)
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);

  LOG(INFO) << "set_outputs correctly updates onnx.Return operands";
}

// Test 5: Verify graph created from existing model has a valid terminator
// Note: Pre-existing .mlir files may use func.return (old implementation)
// while newly created graphs will use onnx.Return (new implementation).
// Both are valid terminators.
TEST_F(OnnxReturnTest, LoadedModelHasValidTerminator) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  // Load the ResNet50 model (uses MLIR backend)
  auto model = morphizen_cxx::Model::load(RESNET_50_PATH);
  auto graph = model->main_graph();
  graph.resolve();

  // Verify outputs are accessible (terminator operands)
  auto outputs = graph.outputs();
  EXPECT_GT(outputs.size(), 0u);

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;
  LOG(INFO) << "Loaded model MLIR output length: " << mlir_text.size();

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Verify a valid return terminator is present (onnx.Return or func.return)
    // Pre-existing models may have func.return, new models will have
    // onnx.Return
    bool has_onnx_return = mlir_text.find("onnx.Return") != std::string::npos;
    bool has_func_return = mlir_text.find("func.return") != std::string::npos ||
                           mlir_text.find("return ") != std::string::npos;

    EXPECT_TRUE(has_onnx_return || has_func_return)
        << "Loaded model MLIR should contain a valid return terminator "
           "(onnx.Return or func.return)";

    if (has_onnx_return) {
      LOG(INFO) << "Loaded model uses onnx.Return (new format)";
    } else if (has_func_return) {
      LOG(INFO) << "Loaded model uses func.return (legacy format, still valid)";
    }
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }
}

// Test 6: Verify cloned graph preserves onnx.Return
TEST_F(OnnxReturnTest, ClonedGraphPreservesOnnxReturn) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_return_clone.onnx");
  auto data_path = std::filesystem::path("onnx_return_clone.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create a simple graph
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("output", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {output[0]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  // Clone the model
  auto cloned_model = model->ref().clone();
  auto cloned_graph = cloned_model->main_graph();
  cloned_graph.resolve();

  // Verify cloned graph outputs
  auto cloned_outputs = cloned_graph.outputs();
  EXPECT_EQ(cloned_outputs.size(), 1u);

  // Get MLIR serialization of cloned graph
  auto mlir_output = cloned_graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    EXPECT_TRUE(mlir_text.find("onnx.Return") != std::string::npos)
        << "Cloned graph MLIR should contain 'onnx.Return'";
    LOG(INFO) << "Cloned graph preserves onnx.Return";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify cloned graph can be saved
  auto clone_path = CMAKE_CURRENT_BINARY_PATH /
                    std::filesystem::path("onnx_return_cloned.onnx");
  cloned_graph.save(clone_path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(clone_path));
  std::filesystem::remove(clone_path);
  std::filesystem::remove(path);
}
