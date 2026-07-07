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

class OnnxCustomTest : public ::testing::Test {};

// Test 1: Verify custom domain operations use onnx.Custom in MLIR
TEST_F(OnnxCustomTest, CustomDomainUsesOnnxCustom) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_custom_test.onnx");
  auto data_path = std::filesystem::path("onnx_custom_test.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create inputs for GroupQueryAttention
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

  // Check if it's text format (not bytecode)
  if (isMLIRTextFormat(mlir_text)) {
    // Verify onnx.Custom is used for custom domain ops
    EXPECT_TRUE(mlir_text.find("onnx.Custom") != std::string::npos)
        << "MLIR output should contain 'onnx.Custom' for custom domain ops";

    // Verify function_name attribute is present with correct value
    EXPECT_TRUE(mlir_text.find("function_name") != std::string::npos)
        << "onnx.Custom should have function_name attribute";
    EXPECT_TRUE(mlir_text.find("GroupQueryAttention") != std::string::npos)
        << "function_name should be 'GroupQueryAttention'";

    // Verify domain_name attribute is present
    EXPECT_TRUE(mlir_text.find("domain_name") != std::string::npos)
        << "onnx.Custom should have domain_name attribute";
    EXPECT_TRUE(mlir_text.find("com.microsoft") != std::string::npos)
        << "domain_name should be 'com.microsoft'";

    LOG(INFO) << "Custom domain operation correctly uses onnx.Custom";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 2: Verify standard ONNX operations do NOT use onnx.Custom
TEST_F(OnnxCustomTest, StandardOnnxDoesNotUseOnnxCustom) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_standard_test.onnx");
  auto data_path = std::filesystem::path("onnx_standard_test.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create a simple graph with standard ONNX operations
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> relu_out = {
      graph.new_node_arg("relu_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> sigmoid_out = {
      graph.new_node_arg("sigmoid_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Add standard ONNX operations (empty domain = standard ONNX)
  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {relu_out[0]},
                 morphizen::NodeAttributesBuilder().build());
  graph.add_node("sigmoid_0", "", "Sigmoid", "", {relu_out[0]},
                 {sigmoid_out[0]}, morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({sigmoid_out[0].value()});
  graph.resolve();

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Verify onnx.Custom is NOT used for standard ops
    EXPECT_TRUE(mlir_text.find("onnx.Custom") == std::string::npos)
        << "Standard ONNX operations should NOT use onnx.Custom";

    // Verify standard operation names are used
    EXPECT_TRUE(mlir_text.find("onnx.Relu") != std::string::npos)
        << "Standard Relu should use 'onnx.Relu'";
    EXPECT_TRUE(mlir_text.find("onnx.Sigmoid") != std::string::npos)
        << "Standard Sigmoid should use 'onnx.Sigmoid'";

    LOG(INFO) << "Standard ONNX operations correctly use onnx.OpName format";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 3: Verify onnx.Custom preserves operation attributes
TEST_F(OnnxCustomTest, OnnxCustomPreservesAttributes) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_custom_attrs.onnx");
  auto data_path = std::filesystem::path("onnx_custom_attrs.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create inputs
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

  // Add custom operation with multiple attributes
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> node_inputs = {
      query[0],     key[0],       value[0],    std::nullopt,
      std::nullopt, std::nullopt, std::nullopt};

  graph.add_node("gqa_node", "com.microsoft", "GroupQueryAttention", "",
                 node_inputs, {output[0]},
                 morphizen::NodeAttributesBuilder()
                     .add("num_heads", int64_t(32))
                     .add("kv_num_heads", int64_t(8))
                     .add("scale", 0.0883883461f)
                     .add("do_rotary", int64_t(0))
                     .add("rotary_interleaved", int64_t(0))
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
    // Verify attributes are preserved
    EXPECT_TRUE(mlir_text.find("num_heads") != std::string::npos)
        << "num_heads attribute should be preserved";
    EXPECT_TRUE(mlir_text.find("kv_num_heads") != std::string::npos)
        << "kv_num_heads attribute should be preserved";
    EXPECT_TRUE(mlir_text.find("scale") != std::string::npos)
        << "scale attribute should be preserved";
    EXPECT_TRUE(mlir_text.find("do_rotary") != std::string::npos)
        << "do_rotary attribute should be preserved";

    LOG(INFO) << "onnx.Custom correctly preserves operation attributes";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 4: Verify mixed graph with both custom and standard operations
TEST_F(OnnxCustomTest, MixedCustomAndStandardOperations) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path =
      CMAKE_CURRENT_BINARY_PATH / std::filesystem::path("onnx_mixed.onnx");
  auto data_path = std::filesystem::path("onnx_mixed.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Input
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Standard ONNX operation output
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> relu_out = {
      graph.new_node_arg("relu_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Custom domain operation output (using a simpler custom op for test)
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> custom_out = {
      graph.new_node_arg("custom_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Final standard operation
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> final_out = {
      graph.new_node_arg("final_out", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Add standard ONNX Relu (empty domain)
  graph.add_node("relu_0", "", "Relu", "", {input[0]}, {relu_out[0]},
                 morphizen::NodeAttributesBuilder().build());

  // Add custom domain operation
  graph.add_node("custom_0", "com.microsoft", "CustomOp", "", {relu_out[0]},
                 {custom_out[0]},
                 morphizen::NodeAttributesBuilder()
                     .add("custom_attr", int64_t(42))
                     .build());

  // Add standard ONNX Sigmoid
  graph.add_node("sigmoid_0", "", "Sigmoid", "", {custom_out[0]},
                 {final_out[0]}, morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({final_out[0].value()});
  graph.resolve();

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Verify standard ops use onnx.OpName format
    EXPECT_TRUE(mlir_text.find("onnx.Relu") != std::string::npos)
        << "Standard Relu should use onnx.Relu";
    EXPECT_TRUE(mlir_text.find("onnx.Sigmoid") != std::string::npos)
        << "Standard Sigmoid should use onnx.Sigmoid";

    // Verify custom op uses onnx.Custom
    EXPECT_TRUE(mlir_text.find("onnx.Custom") != std::string::npos)
        << "Custom domain op should use onnx.Custom";

    // Verify custom op attributes
    EXPECT_TRUE(mlir_text.find("function_name") != std::string::npos)
        << "Custom op should have function_name attribute";
    EXPECT_TRUE(mlir_text.find("CustomOp") != std::string::npos)
        << "function_name should be 'CustomOp'";

    LOG(INFO) << "Mixed custom and standard operations handled correctly";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 5: Verify node properties can be retrieved from onnx.Custom operations
TEST_F(OnnxCustomTest, RetrieveNodePropertiesFromOnnxCustom) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_custom_props.onnx");
  auto data_path = std::filesystem::path("onnx_custom_props.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create inputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("output", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Add custom domain operation
  graph.add_node(
      "my_custom_node", "com.microsoft", "MyCustomOp", "test desc", {input[0]},
      {output[0]},
      morphizen::NodeAttributesBuilder().add("param1", int64_t(100)).build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  // Get the nodes and verify properties
  auto nodes = graph.nodes();
  ASSERT_EQ(nodes.size(), 1u);

  auto node = nodes[0];

  // Verify node properties can be retrieved
  std::string op_type = node.op_type();
  std::string domain = node.op_domain();

  // The op_type should be extracted from function_name attribute
  EXPECT_EQ(op_type, "MyCustomOp")
      << "op_type should be 'MyCustomOp' from function_name attribute";

  // The domain should be extracted from domain_name attribute
  EXPECT_EQ(domain, "com.microsoft")
      << "domain should be 'com.microsoft' from domain_name attribute";

  LOG(INFO) << "Node properties: op_type=" << op_type
            << ", op_domain=" << domain;

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 6: Verify multiple custom domain operations in same graph
TEST_F(OnnxCustomTest, MultipleCustomDomainOperations) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_multi_custom.onnx");
  auto data_path = std::filesystem::path("onnx_multi_custom.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {
      {"com.microsoft", 1}, {"com.custom", 1}, {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create inputs and outputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> input = {
      graph.new_node_arg("input", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> mid1 = {
      graph.new_node_arg("mid1", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> mid2 = {
      graph.new_node_arg("mid2", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> output = {
      graph.new_node_arg("output", {1, 8},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT)};

  // Add first custom domain operation (com.microsoft)
  graph.add_node("ms_op", "com.microsoft", "MicrosoftOp", "", {input[0]},
                 {mid1[0]}, morphizen::NodeAttributesBuilder().build());

  // Add second custom domain operation (com.custom)
  graph.add_node("custom_op", "com.custom", "CustomOp", "", {mid1[0]},
                 {mid2[0]}, morphizen::NodeAttributesBuilder().build());

  // Add standard ONNX operation
  graph.add_node("relu", "", "Relu", "", {mid2[0]}, {output[0]},
                 morphizen::NodeAttributesBuilder().build());

  graph.set_inputs({input[0].value()});
  graph.set_outputs({output[0].value()});
  graph.resolve();

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Count onnx.Custom occurrences (should be 2 for the custom domain ops)
    size_t custom_count = 0;
    size_t pos = 0;
    while ((pos = mlir_text.find("onnx.Custom", pos)) != std::string::npos) {
      ++custom_count;
      pos += 11; // length of "onnx.Custom"
    }
    EXPECT_EQ(custom_count, 2u)
        << "Should have exactly 2 onnx.Custom operations";

    // Verify both domains are present
    EXPECT_TRUE(mlir_text.find("com.microsoft") != std::string::npos)
        << "com.microsoft domain should be present";
    EXPECT_TRUE(mlir_text.find("com.custom") != std::string::npos)
        << "com.custom domain should be present";

    // Verify both function names are present
    EXPECT_TRUE(mlir_text.find("MicrosoftOp") != std::string::npos)
        << "MicrosoftOp function_name should be present";
    EXPECT_TRUE(mlir_text.find("CustomOp") != std::string::npos)
        << "CustomOp function_name should be present";

    // Verify standard Relu is still onnx.Relu
    EXPECT_TRUE(mlir_text.find("onnx.Relu") != std::string::npos)
        << "Standard Relu should still use onnx.Relu";

    LOG(INFO) << "Multiple custom domain operations handled correctly";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}

// Test 7: Verify onnx.Custom with multiple outputs
TEST_F(OnnxCustomTest, OnnxCustomWithMultipleOutputs) {
#ifndef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP() << "MLIR backend not enabled at compile time";
#endif
  if (!isMLIRBackendActive()) {
    GTEST_SKIP() << "MLIR backend not active at runtime";
  }

  auto path = CMAKE_CURRENT_BINARY_PATH /
              std::filesystem::path("onnx_custom_multi_out.onnx");
  auto data_path = std::filesystem::path("onnx_custom_multi_out.dat");
  std::vector<std::pair<std::string, int64_t>> opset = {{"com.microsoft", 1},
                                                        {"", 17}};

  auto model = morphizen_cxx::Model::create(path, opset);
  auto graph = model->main_graph();

  // Create inputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> query = {
      graph.new_node_arg("query", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> key = {
      graph.new_node_arg("key", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> value = {
      graph.new_node_arg("value", {1, 1, 1024},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};

  // Create multiple outputs (GQA returns output + present_key + present_value)
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> out_attn = {
      graph.new_node_arg("out_attn", {1, 1, 4096},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> present_key = {
      graph.new_node_arg("present_key", {1, 8, 128, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> present_value = {
      graph.new_node_arg("present_value", {1, 8, 128, 128},
                         ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)};

  // Add GQA with multiple outputs
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> node_inputs = {
      query[0],     key[0],       value[0],    std::nullopt,
      std::nullopt, std::nullopt, std::nullopt};
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> node_outputs = {
      out_attn[0], present_key[0], present_value[0]};

  graph.add_node("gqa", "com.microsoft", "GroupQueryAttention", "", node_inputs,
                 node_outputs,
                 morphizen::NodeAttributesBuilder()
                     .add("num_heads", int64_t(32))
                     .add("kv_num_heads", int64_t(8))
                     .build());

  graph.set_inputs({query[0].value(), key[0].value(), value[0].value()});
  graph.set_outputs(
      {out_attn[0].value(), present_key[0].value(), present_value[0].value()});
  graph.resolve();

  // Verify outputs
  auto outputs = graph.outputs();
  EXPECT_EQ(outputs.size(), 3u);

  // Get MLIR serialization
  auto mlir_output = graph.save_string();
  ASSERT_TRUE(mlir_output.get() != nullptr);
  ASSERT_FALSE(mlir_output->empty());

  std::string mlir_text = *mlir_output;

  // Check if it's text format
  if (isMLIRTextFormat(mlir_text)) {
    // Verify onnx.Custom is used
    EXPECT_TRUE(mlir_text.find("onnx.Custom") != std::string::npos)
        << "Custom domain op should use onnx.Custom";

    LOG(INFO) << "onnx.Custom with multiple outputs handled correctly";
  } else {
    LOG(INFO)
        << "MLIR output is in bytecode format, skipping text verification";
  }

  // Verify save works
  graph.save(path, data_path, 999999);
  ASSERT_TRUE(std::filesystem::exists(path));
  std::filesystem::remove(path);
}
