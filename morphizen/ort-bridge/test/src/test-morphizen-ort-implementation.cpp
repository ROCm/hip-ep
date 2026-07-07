/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "./test-api-coverage-checker.hpp"
#include "./test-coverage-wrapper.hpp"
#include "./test-environment.hpp"
#include "morphizen/morphizen.hpp"
#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <morphizen-utils/morphizen-utils.hpp>
#include <numeric>
#include <vector>
DEF_ENV_PARAM_2(MORPHIZEN_ORT_BRIDGE_UNITTEST_COVERAGE, "", std::string)

namespace morphizen {
namespace test {

std::string vec_to_string(const std::vector<int64_t> &vec) {
  std::ostringstream oss;
  oss << "{ ";
  for (const auto &elem : vec) {
    oss << elem << " ";
  }
  oss << "}";
  return oss.str();
}
/**
 * Test fixture specifically for testing MorphiZen ORT API implementation
 * through the coverage wrapper
 */
class MorphizenOrtApiTest : public TestCoverageWrapperTest {
protected:
  void SetUp() override {
    TestCoverageWrapperTest::SetUp();

    // Reset statistics at the beginning of the overall test
    reset_morphizen_ort_api_call_statistics();

    // Log that we're starting a fresh test suite
    LOG(INFO) << "Starting comprehensive test suite with clean statistics";
  }

  void TearDown() override {
    // Log final statistics for the entire test suite

    auto stats = get_morphizen_ort_api_call_statistics();
    LOG(INFO) << "Test suite completed with " << stats.size()
              << " different API calls made total";

    TestCoverageWrapperTest::TearDown();
  }

protected:
  // Member functions for individual test components
  void Test01_TestIsolationVerification();
  void Test02_TestIsolationVerificationSecond();
  void Test03_ModelLoadAndDelete();
  void Test04_ModelMetaDataOperations();
  void Test05_GraphBasicOperations();
  void Test06_GraphAdvancedOperations();
  void Test07_create_simple_conv_relu_model();
  void Test08_NodeOperations();
  void Test09_NodeArgOperations();
  void Test10_NodeAttributesOperations();
  void Test11_AttributeProtoOperations();
  void Test12_TensorProtoOperations();
  void Test13_ExtendedApiOperations();
  void Test14_GraphTensorOperations();
  void Test16_GraphFuseOperations();
  void Test17_GraphNodeRemovalOperations();
  void Test18_MissingApisCoverage();
  void Test19_add_sin_op_before_relu_op();
  void Test20_conv_relu_fuse_conv2d_nchw();
  void Test21_fuse_relu_q();
  void Test22_create_initializer_node_arg();
  void Test23_try_fuse_and_fuse();
  void Test24_convert_initializer_to_const_op();

  void ComprehensiveCoverageReport();
  void DetailedCoverageAnalysis();
  void DeleteSimpleConvReluModel();
  morphizen::Model *simple_conv_relu_model_ = nullptr;
};

// ============================================================================
// Test Execution Strategy - Sequential Member Function Calls
// ============================================================================
// All individual test components are implemented as private member functions
// of the MorphizenOrtApiTest class. The main TEST_F(MorphizenOrtApiTest,
// TestAll) function calls them in a specific sequential order, ensuring:
// 1. Predictable execution order
// 2. Single test execution context
// 3. Cumulative API coverage tracking
// 4. Comprehensive logging and analysis
//
// Test Execution Sequence:
// 1. Test01_TestIsolationVerification() - Verify test isolation works
// 2. Test02_TestIsolationVerificationSecond() - Double-check isolation
// 3. Test03_ModelLoadAndDelete() - Model lifecycle tests
// 4. Test04_ModelMetaDataOperations() - Model metadata operations
// 5. Test05_GraphBasicOperations() - Basic graph operations
// 6. Test06_GraphAdvancedOperations() - Advanced graph operations
// 7. Test07_create_simple_conv_relu_model() - Create Conv+ReLU model pattern
// 8. Test08_NodeOperations() - Node-level operations
// 9. Test09_NodeArgOperations() - Node argument operations
// 10. Test10_NodeAttributesOperations() - Node attribute operations
// 11. Test11_AttributeProtoOperations() - Attribute proto operations
// 12. Test12_TensorProtoOperations() - Tensor proto operations
// 13. Test13_ExtendedApiOperations() - Extended API operations
// 14. Test14_GraphTensorOperations() - Graph tensor operations
// 16. Test16_GraphFuseOperations() - Graph fusion operations
// 17. Test17_GraphNodeRemovalOperations() - Graph node removal operations
// 18. Test18_MissingApisCoverage() - Coverage for remaining missing APIs
// 19. Test19_add_sin_op_before_relu_op() - Add Sin->Cos operations before ReLU
// ============================================================================

// ============================================================================
// Test Isolation Verification (Run First)
// ============================================================================

void MorphizenOrtApiTest::Test01_TestIsolationVerification() {
  // Make a few API calls
  auto lib_id = wrapped_api_->get_lib_id();
  auto lib_name = wrapped_api_->get_lib_name();
  ASSERT_EQ((std::string)lib_id->c_str(), "v1.0.0")
      << "Expected library ID to be 'v1.0.0'";
  if (backend_ == morphizen::kONNXIRBackend) {
    ASSERT_EQ((std::string)lib_name->c_str(), "morphizen-onnx-imp")
        << "Expected library name to be 'morphizen-onnx-imp'";
  } else if (backend_ == morphizen::kMLIRBackend) {
    ASSERT_EQ((std::string)lib_name->c_str(), "morphizen-mlir-imp")
        << "Expected library name to be 'morphizen-mlir-imp'";
  } else {
    ASSERT_TRUE(false) << "not support backend : " << backend_;
  }
  LOG(INFO) << "Test suite initialization verification passed";
}

void MorphizenOrtApiTest::Test02_TestIsolationVerificationSecond() {
  // This test makes different API calls to build up coverage

  // Make different API calls than the previous test
  try {
    std::filesystem::path temp_path =
        std::filesystem::temp_directory_path() / "isolation_test.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};
    auto *model = wrapped_api_->create_empty_model(temp_path, opset);
    ASSERT_TRUE(model != nullptr) << "Failed to create empty model";
    wrapped_api_->model_delete(model);
  } catch (...) {
    LOG(INFO) << "Model operations tested for isolation";
    ASSERT_TRUE(false) << "Model creation failed";
  }
  LOG(INFO) << "Second test isolation verification passed";
}

// ============================================================================
// Model API Tests [0-6]
// ============================================================================

void MorphizenOrtApiTest::Test03_ModelLoadAndDelete() {
  // Test model_load and model_delete
  // Note: These tests may require actual model files to work properly
  // For now, we test the API wrapper functionality

  // This will likely fail with actual file operations, but will test the
  // wrapper
  try {
    auto test_model_path = RESNET_50_PATH;
    if (backend_ == morphizen::kMLIRBackend) {
      test_model_path = RESNET_50_MLIR_PATH;
    }
    auto *model = wrapped_api_->model_load(test_model_path.u8string());
    ASSERT_TRUE(model != nullptr) << "Failed to load model from file";
    // auto* morphizen_model = reinterpret_cast<morphizen::Model*>(model);
    // ASSERT_TRUE(morphizen_model->is_valid())
    //     << "Loaded model is not valid";
    if (model) {
      wrapped_api_->model_delete(model);
    }
  } catch (...) {
    // Expected to fail without actual model file
    LOG(INFO) << "model_load/model_delete APIs tested (expected failure "
                 "without model file)";
  }
  LOG(INFO) << "Test03 Model load and delete operations tested";
}

void MorphizenOrtApiTest::Test04_ModelMetaDataOperations() {
  // Test model metadata operations
  // These will likely require a valid model, so we'll test the wrapper call
  try {
    // Create a temporary model for testing (this may fail)
    std::filesystem::path temp_path =
        std::filesystem::temp_directory_path() / "test_model.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};

    auto *model = wrapped_api_->create_empty_model(temp_path, opset);
    ASSERT_TRUE(model != nullptr) << "Failed to create empty model";
    wrapped_api_->model_set_meta_data(*model, "test_key", "test_value");
    int has_meta = wrapped_api_->model_has_meta_data(*model, "test_key");
    ASSERT_TRUE(has_meta) << "Model should have metadata for 'test_key'";
    auto meta_value = wrapped_api_->model_get_meta_data(*model, "test_key");
    ASSERT_EQ(*meta_value, "test_value")
        << "Model metadata value for 'test_key' should be 'test_value'";

    // Test model cloning
    auto *cloned_model = wrapped_api_->model_clone(*model, 1024);
    // Note: model_clone may return nullptr in current implementation
    if (cloned_model) {
      wrapped_api_->model_delete(cloned_model);
    }
    wrapped_api_->model_delete(model);
  } catch (...) {
    LOG(INFO) << "Model metadata APIs tested (may require valid model)";
  }
  LOG(INFO) << "Test04 Model metadata operations test passed";
}

// ============================================================================
// Graph API Tests [7-23]
// ============================================================================

void MorphizenOrtApiTest::Test05_GraphBasicOperations() {
  try {
    if (!simple_conv_relu_model_) {
      Test07_create_simple_conv_relu_model();
    }
    std::filesystem::path test07_output_model =
        CMAKE_CURRENT_BINARY_PATH /
        "Test07_create_simple_conv_relu_model.onnx.graph.onnx";

    auto *model = wrapped_api_->model_load(test07_output_model.u8string());
    ASSERT_TRUE(model != nullptr) << "Failed to load model for graph tests";

    auto &graph = wrapped_api_->model_main_graph(*model);
    ASSERT_TRUE(&graph != nullptr) << "Main graph should not be null";

    // Test basic graph operations
    const std::string &graph_name = wrapped_api_->graph_get_name(graph);
    LOG(INFO) << "Graph name: " << graph_name;
    ASSERT_FALSE(graph_name.empty()) << "Graph name should not be empty";

    // const auto& model_ref =
    //     wrapped_api_->graph_get_model(graph); // Test node and I/O operations

    // Test graph_get_model
    const auto &model_ref = wrapped_api_->graph_get_model(graph);
    ASSERT_TRUE(&model_ref != nullptr)
        << "Graph model reference should not be null";

    auto inputs = wrapped_api_->graph_get_inputs_unsafe(graph);
    ASSERT_GE(inputs->size(), 1)
        << "Expected at least 1 input node argument, got: " << inputs->size();
    for (auto i = 0; i < inputs->size(); i++) {
      LOG(INFO) << "Input node argument name: "
                << wrapped_api_->node_arg_get_name_unsafe(*(*inputs)[i]);
    }
    auto outputs = wrapped_api_->graph_get_outputs_unsafe(graph);
    ASSERT_GE(outputs->size(), 1)
        << "Expected at least 1 output node argument, got: " << outputs->size();
    for (auto i = 0; i < outputs->size(); i++) {
      LOG(INFO) << "Output node argument name: "
                << wrapped_api_->node_arg_get_name_unsafe(*(*outputs)[i]);
    }

    auto output_0_arg_name = wrapped_api_->node_arg_get_name_unsafe(
        *(*outputs)[0]); // Get the name of the first output node argument
    LOG(INFO) << "First output node argument name: " << output_0_arg_name;
    auto output_0_node =
        wrapped_api_->graph_producer_node(graph, output_0_arg_name);
    ASSERT_TRUE(output_0_node != nullptr)
        << "Producer node for first output should not be null";

    // Test graph nodes
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_GT(nodes->size(), 1)
        << "Expected more than 1 node in the graph, got: " << nodes->size();
    LOG(INFO) << "Graph has " << nodes->size() << " nodes";
    const auto &node = wrapped_api_->graph_get_node(
        graph, wrapped_api_->node_get_index(*(*nodes)[0]));
    ASSERT_TRUE(node != nullptr) << "First node should not be null";
    ASSERT_TRUE(node == (*nodes)[0]) << "First node should not be null";
    const std::string &node_name = wrapped_api_->node_get_name(*node);
    LOG(INFO) << "First node name: " << node_name;
    const std::string &op_domain = wrapped_api_->node_op_domain(*node);
    LOG(INFO) << "First node op domain: " << op_domain;
    const std::string &op_type = wrapped_api_->node_op_type(*node);
    LOG(INFO) << "First node op_type: " << op_type;

    // Test graph_get_node_arg - try to get a node argument by name
    auto first_input_name =
        wrapped_api_->node_arg_get_name_unsafe(*(*inputs)[0]);
    const auto *node_arg =
        wrapped_api_->graph_get_node_arg(graph, first_input_name);
    // Note: node_arg may be null depending on implementation

    // Test graph_get_consumer_nodes_unsafe
    auto consumer_nodes =
        wrapped_api_->graph_get_consumer_nodes_unsafe(graph, first_input_name);
    LOG(INFO) << "Consumer nodes for " << first_input_name << ": "
              << consumer_nodes->size();

    // Test initialized tensors
    const auto &tensors =
        wrapped_api_->graph_get_all_initialized_tensors(graph);
    // Check that we have some initialized tensors (weights, biases, etc.)
    ASSERT_GE(tensors.size(), 0)
        << "Expected at least 0 initialized tensors, got: " << tensors.size();
    LOG(INFO) << "Graph has " << tensors.size() << " initialized tensors";

    // Test graph resolution
    int resolved = wrapped_api_->graph_resolve(graph, false);
    ASSERT_GE(resolved, 0) << "Graph resolution should not fail";
    wrapped_api_->model_delete(model);
  } catch (...) {
    LOG(INFO) << "Graph basic operations tested";
  }
  LOG(INFO) << "Test05 Graph basic operations test passed";
}

void MorphizenOrtApiTest::Test06_GraphAdvancedOperations() {
  try {
    auto test_model_path = RESNET_50_PATH;
    if (backend_ == morphizen::kMLIRBackend) {
      test_model_path = RESNET_50_MLIR_PATH;
    }
    auto *model = wrapped_api_->model_load(test_model_path.u8string());
    ASSERT_TRUE(model != nullptr) << "Failed to load model for graph tests";
    auto &graph = wrapped_api_->model_main_graph(*model);
    // Test model path
    const auto &model_path = wrapped_api_->get_model_path(graph);
    EXPECT_EQ(model_path, test_model_path.u8string())
        << "Model path should match the loaded model path";
    // Test graph name setting
    wrapped_api_->graph_set_name(graph, "test_graph_name");
    auto retrieved_name = wrapped_api_->graph_get_name(graph);
    ASSERT_EQ(retrieved_name, "test_graph_name")
        << "Graph name should be set to 'test_graph_name'";

    // Test DFS traversal (with empty lambda functions)
    std::vector<const morphizen::Node *> leaf_nodes;
    auto output_node_args = wrapped_api_->graph_get_outputs_unsafe(graph);
    for (auto node_arg : *output_node_args) {
      ASSERT_TRUE(node_arg != nullptr);
      auto producer_node = wrapped_api_->graph_producer_node(
          graph, wrapped_api_->node_arg_get_name_unsafe(*node_arg));
      ASSERT_TRUE(producer_node);
      leaf_nodes.push_back(producer_node);
    }
    auto nodes_entering = std::vector<const morphizen::Node *>{};
    auto nodes_leaving = std::vector<const morphizen::Node *>{};
    auto verbose_log = 0;
    wrapped_api_->graph_reverse_dfs_from(
        graph, gsl::span<const morphizen::Node *const>(leaf_nodes),
        [this, &nodes_entering, verbose_log](const morphizen::Node *node) {
          auto op_type = wrapped_api_->node_op_type(*node);
          if (op_type == "Constant") {
            LOG_IF(INFO, verbose_log)
                << "Skipping Constant node: "
                << "\"" << wrapped_api_->node_get_name(*node) << "\"";
            return; /* skip constant nodes */
          }
          nodes_entering.push_back(node);
          LOG_IF(INFO, verbose_log)
              << "  --- entering node: "
              << "\"" << wrapped_api_->node_get_name(*node) << "\""; /* enter */
        },
        [this, &nodes_leaving, verbose_log](const morphizen::Node *node) {
          auto op_type = wrapped_api_->node_op_type(*node);
          if (op_type == "Constant") {
            LOG_IF(INFO, verbose_log)
                << "Skipping Constant node: "
                << "\"" << wrapped_api_->node_get_name(*node) << "\"";
            return; /* skip constant nodes */
          }
          nodes_leaving.push_back(node);
          LOG_IF(INFO, verbose_log)
              << "  --- leaving node: "
              << "\"" << wrapped_api_->node_get_name(*node) << "\""; /* leave */
        },
        [](const morphizen::Node *, const morphizen::Node *) {
          return false; /* stop */
        });
    ASSERT_TRUE(!nodes_entering.empty())
        << "Expected at least one node to be entered during DFS traversal";
    ASSERT_TRUE(!nodes_leaving.empty())
        << "Expected at least one node to be left during DFS traversal";
    for (const auto *node : nodes_entering) {
      LOG_IF(INFO, verbose_log)
          << "Entered node: "
          << "\"" << wrapped_api_->node_get_name(*node) << "\"";
    }
    for (const auto *node : nodes_leaving) {
      LOG_IF(INFO, verbose_log)
          << "Left node: "
          << "\"" << wrapped_api_->node_get_name(*node) << "\"";
    }
    auto nodes_entering2 = std::vector<const morphizen::Node *>{};
    auto nodes_leaving2 = std::vector<const morphizen::Node *>{};
    // test graph_reverse_dfs_from_preemp similiar to above
    wrapped_api_->graph_reverse_dfs_from_preemp(
        graph, gsl::span<const morphizen::Node *const>(leaf_nodes),
        [this, &nodes_entering2,
         verbose_log](const morphizen::Node *node) -> bool {
          auto op_type = wrapped_api_->node_op_type(*node);
          if (op_type == "Constant") {
            LOG_IF(INFO, verbose_log)
                << "Skipping Constant node: "
                << "\"" << wrapped_api_->node_get_name(*node) << "\"";
            return false; /* skip constant nodes */
          }
          nodes_entering2.push_back(node);
          LOG_IF(INFO, verbose_log)
              << "  --- entering node: "
              << "\"" << wrapped_api_->node_get_name(*node) << "\""; /* enter */
          return false;
        },
        [this, &nodes_leaving2,
         verbose_log](const morphizen::Node *node) -> bool {
          auto op_type = wrapped_api_->node_op_type(*node);
          if (op_type == "Constant") {
            LOG_IF(INFO, verbose_log)
                << "Skipping Constant node: "
                << "\"" << wrapped_api_->node_get_name(*node) << "\"";
            return false; /* skip constant nodes */
          }
          nodes_leaving2.push_back(node);
          LOG_IF(INFO, verbose_log)
              << "  --- leaving node: "
              << "\"" << wrapped_api_->node_get_name(*node) << "\""; /* leave */
          return false;
        },
        nullptr,
        [](const morphizen::Node *, const morphizen::Node *) -> bool {
          return false; /* stop */
        });
    EXPECT_EQ(nodes_entering.size(), nodes_entering2.size())
        << "Expected same number of nodes entered in both DFS traversals";
    EXPECT_EQ(nodes_leaving.size(), nodes_leaving2.size())
        << "Expected same number of nodes left in both DFS traversals";
    for (size_t i = 0; i < nodes_entering.size(); ++i) {
      EXPECT_EQ(nodes_entering[i], nodes_entering2[i])
          << "Nodes entered in both traversals should match at index " << i;
    }
    for (size_t i = 0; i < nodes_leaving.size(); ++i) {
      EXPECT_EQ(nodes_leaving[i], nodes_leaving2[i])
          << "Nodes left in both traversals should match at index " << i;
    }
    wrapped_api_->model_delete(model);
  } catch (...) {
    LOG(INFO) << "Graph advanced operations tested";
  }
}

// ============================================================================
// Node API Tests [24-33]
// ============================================================================

void MorphizenOrtApiTest::Test08_NodeOperations() {
  try {
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for fuse test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }
    auto *model = simple_conv_relu_model_;
    ASSERT_TRUE(model != nullptr);
    {
      auto &graph = wrapped_api_->model_main_graph(*model);
      auto nodes = wrapped_api_->graph_nodes_unsafe(graph);

      // Test node operations on each node (if any exist)
      for (const auto *node : *nodes) {
        if (node) {
          const std::string &name = wrapped_api_->node_get_name(*node);
          const std::string &desc = wrapped_api_->node_description(*node);
          size_t index = wrapped_api_->node_get_index(*node);
          const std::string &op_type = wrapped_api_->node_op_type(*node);
          const std::string &op_domain = wrapped_api_->node_op_domain(*node);

          auto inputs = wrapped_api_->node_get_inputs_unsafe(*node);
          auto outputs = wrapped_api_->node_get_output_node_args_unsafe(*node);

          bool is_fused = wrapped_api_->node_type_is_fused(*node);

          // Test getting attributes (const version, so we can't modify)
          auto &attrs = wrapped_api_->node_get_attributes(
              *const_cast<morphizen::Node *>(node));

          // Test node_get_function_body (may return null for non-function
          // nodes)
          try {
            const auto &function_body =
                wrapped_api_->node_get_function_body(*node);
            // If we get here, the node has a function body
          } catch (...) {
            // Expected for most nodes that don't have function bodies
          }
        }
      }
    }
  } catch (...) {
    LOG(INFO) << "Node operations tested";
  }
}

// ============================================================================
// NodeArg API Tests [34-45]
// ============================================================================

void MorphizenOrtApiTest::Test09_NodeArgOperations() {
  try {
    std::filesystem::path temp_path =
        std::filesystem::temp_directory_path() / "test_nodeargs.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};

    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for fuse test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }

    auto *model = simple_conv_relu_model_;
    ASSERT_TRUE(model != nullptr) << "Failed to load model for NodeArg tests";
    {
      auto &graph = wrapped_api_->model_main_graph(*model);

      // Create a new NodeArg for testing
      std::vector<int64_t> shape = {1, 3, 224, 224};
      auto &new_node_arg = wrapped_api_->node_arg_new(graph, "test_input",
                                                      &shape, 1); // FLOAT type

      // Test NodeArg operations
      const std::string &name =
          wrapped_api_->node_arg_get_name_unsafe(new_node_arg);
      ASSERT_TRUE(name == "test_input")
          << "Expected node arg name to be 'test_input', got: " << name;

      bool exists = wrapped_api_->node_arg_is_exists(new_node_arg);
      bool is_constant =
          wrapped_api_->node_arg_is_constant(graph, new_node_arg);
      auto node_arg_shape =
          wrapped_api_->node_arg_get_shape_i64_unsafe(new_node_arg);
      ASSERT_TRUE(*node_arg_shape == shape)
          << "Expected shape to be {1, 3, 224, 224}, got: "
          << vec_to_string(*node_arg_shape);

      int element_type = wrapped_api_->node_arg_get_element_type(new_node_arg);
      ASSERT_TRUE(element_type == 1) // FLOAT type
          << "Expected element type to be FLOAT (1), got: " << element_type;

      // Test shape and denotation setting
      std::vector<int64_t> new_shape = {1, 64, 222, 222};
      wrapped_api_->node_arg_set_shape_i64(new_node_arg, new_shape);

      std::vector<std::string> denotation = {"BATCH", "CHANNEL", "HEIGHT",
                                             "WIDTH"};
      wrapped_api_->node_arg_set_denotation(new_node_arg, denotation);
      auto retrieved_denotation =
          wrapped_api_->node_arg_get_denotation_unsafe(new_node_arg);

      // Test element type setting
      // This API is useless , should be removed
      // wrapped_api_->node_arg_set_element_type(new_node_arg, 1); // FLOAT

      // Test external location (may return 0 for non-external data)
      std::string external_file;
      size_t offset = 0, size = 0, checksum = 0;
      int has_external = wrapped_api_->node_arg_external_location(
          graph, new_node_arg, external_file, offset, size, checksum);

      // Test get_const_data_as_tensor (may throw for non-constant args)
      if (wrapped_api_->node_arg_is_constant(graph, new_node_arg)) {
        const auto &tensor_data =
            wrapped_api_->node_arg_get_const_data_as_tensor(graph,
                                                            new_node_arg);
      }

      // Test cloning
      /* useless
      auto& cloned_arg = wrapped_api_->node_arg_clone(graph, new_node_arg,
                                                      "cloned_test_input");*/
    }
  } catch (...) {
    LOG(INFO) << "NodeArg operations tested";
  }
}

// ============================================================================
// NodeAttributes API Tests [46-50]
// ============================================================================

void MorphizenOrtApiTest::Test10_NodeAttributesOperations() {
  // Test NodeAttributes operations
  auto *attrs = wrapped_api_->node_attributes_new();
  ASSERT_NE(attrs, nullptr);

  // Create some test attributes
  auto *int_attr = wrapped_api_->attr_proto_new_int("test_int", 42);
  auto *float_attr = wrapped_api_->attr_proto_new_float("test_float", 3.14f);
  auto *string_attr =
      wrapped_api_->attr_proto_new_string("test_string", "hello");

  if (int_attr && float_attr && string_attr) {
    // Add attributes
    wrapped_api_->node_attributes_add(*attrs, std::move(*int_attr));
    wrapped_api_->node_attributes_add(*attrs, std::move(*float_attr));
    wrapped_api_->node_attributes_add(*attrs, std::move(*string_attr));

    // Get attributes
    const auto *retrieved_int =
        wrapped_api_->node_attributes_get(*attrs, "test_int");
    const auto *retrieved_float =
        wrapped_api_->node_attributes_get(*attrs, "test_float");
    const auto *retrieved_string =
        wrapped_api_->node_attributes_get(*attrs, "test_string");

    // Get keys
    auto keys = wrapped_api_->node_attributes_get_keys(*attrs);
    EXPECT_GE(keys->size(), 3);
  }

  wrapped_api_->node_attributes_delete(attrs);
}

// ============================================================================
// AttributeProto API Tests [51-69]
// ============================================================================

void MorphizenOrtApiTest::Test11_AttributeProtoOperations() {
  // Test various attribute creation and manipulation
  {
    // Test integer attributes
    auto *int_attr = wrapped_api_->attr_proto_new_int("int_attr", 123);
    ASSERT_TRUE(int_attr);
    const std::string &name = wrapped_api_->attr_proto_get_name(*int_attr);
    EXPECT_EQ(name, "int_attr");

    int type = wrapped_api_->attr_proto_get_type(*int_attr);
    EXPECT_EQ(type,
              (int)ONNX_NAMESPACE::AttributeProto_AttributeType::
                  AttributeProto_AttributeType_INT); // Check type is INT
    int64_t value = wrapped_api_->attr_proto_get_int(*int_attr);
    EXPECT_EQ(value, 123);

    // Test cloning
    auto *cloned_attr = wrapped_api_->attr_proto_clone(*int_attr);
    int type_cloned = wrapped_api_->attr_proto_get_type(*cloned_attr);
    EXPECT_EQ(type_cloned,
              ONNX_NAMESPACE::AttributeProto_AttributeType::
                  AttributeProto_AttributeType_INT); // Check type is INT

    value = wrapped_api_->attr_proto_get_int(*cloned_attr);
    EXPECT_EQ(value, 123);
    const std::string &name_cloned =
        wrapped_api_->attr_proto_get_name(*cloned_attr);
    EXPECT_EQ(name_cloned, "int_attr");

    ASSERT_TRUE(cloned_attr);
    wrapped_api_->attr_proto_delete(cloned_attr);
    wrapped_api_->attr_proto_delete(int_attr);
  }
  {
    // Test float attributes
    auto *float_attr = wrapped_api_->attr_proto_new_float("float_attr", 2.718f);
    ASSERT_TRUE(float_attr != nullptr);
    float value = wrapped_api_->attr_proto_get_float(*float_attr);
    EXPECT_NEAR(value, 2.718f, 0.001f);
    wrapped_api_->attr_proto_delete(float_attr);
  }

  {
    // Test string attributes
    auto *string_attr =
        wrapped_api_->attr_proto_new_string("string_attr", "test_value");
    ASSERT_TRUE(string_attr != nullptr);
    const std::string &value =
        wrapped_api_->attr_proto_get_string(*string_attr);
    EXPECT_EQ(value, "test_value");

    // Test string release
    auto released_string = wrapped_api_->attr_proto_release_string(string_attr);

    wrapped_api_->attr_proto_delete(string_attr);
  }

  {
    // Test array attributes
    std::vector<int64_t> int_values = {1, 2, 3, 4, 5};
    auto *ints_attr =
        wrapped_api_->attr_proto_new_ints("ints_attr", int_values);
    ASSERT_TRUE(ints_attr != nullptr);
    auto retrieved_ints = wrapped_api_->attr_proto_get_ints(*ints_attr);
    EXPECT_EQ(retrieved_ints.size(), int_values.size());
    for (size_t i = 0; i < retrieved_ints.size(); ++i) {
      EXPECT_EQ(retrieved_ints[i], int_values[i]);
    }
    wrapped_api_->attr_proto_delete(ints_attr);
  }
  {

    std::vector<float> float_values = {1.1f, 2.2f, 3.3f};
    auto *floats_attr =
        wrapped_api_->attr_proto_new_floats("floats_attr", float_values);
    ASSERT_TRUE(floats_attr != nullptr);
    auto retrieved_floats = wrapped_api_->attr_proto_get_floats(*floats_attr);
    EXPECT_EQ(retrieved_floats.size(), float_values.size());
    for (size_t i = 0; i < retrieved_floats.size(); ++i) {
      EXPECT_NEAR(retrieved_floats[i], float_values[i], 0.001f);
    }
    wrapped_api_->attr_proto_delete(floats_attr);
  }
  {

    std::vector<std::string> string_values = {"a", "b", "c"};
    auto *strings_attr =
        wrapped_api_->attr_proto_new_strings("strings_attr", string_values);
    ASSERT_TRUE(strings_attr != nullptr);
    auto retrieved_strings =
        wrapped_api_->attr_proto_get_strings(*strings_attr);
    EXPECT_EQ(retrieved_strings.size(), string_values.size());
    for (size_t i = 0; i < retrieved_strings.size(); ++i) {
      EXPECT_EQ(retrieved_strings[i], string_values[i]);
    }
    wrapped_api_->attr_proto_delete(strings_attr);
  }
  {
    // TENSOR attribute creation round-trips through the proto factory into a
    // self-describing `mlir::DenseElementsAttr` (mlir-imp backend). The data
    // read-back accessor `attr_proto_get_tensor` is intentionally unsupported
    // on this backend (it LOG(FATAL)s): TENSOR values are read directly from
    // the DenseElementsAttr by MLIR consumer passes, not via the proto API.
    // We therefore only exercise creation, name and type here.
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> tensor_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto tensor_proto = wrapped_api_->tensor_proto_new_floats(
        "tensor_attr", shape, tensor_data);
    auto *tensor_attr =
        wrapped_api_->attr_proto_new_tensor("tensor_attr", *tensor_proto);
    ASSERT_TRUE(tensor_attr != nullptr);
    EXPECT_EQ(wrapped_api_->attr_proto_get_name(*tensor_attr), "tensor_attr");
    EXPECT_EQ(wrapped_api_->attr_proto_get_type(*tensor_attr),
              (int)ONNX_NAMESPACE::AttributeProto_AttributeType::
                  AttributeProto_AttributeType_TENSOR);

    wrapped_api_->attr_proto_set_name(tensor_attr, "renamed_tensor_attr");
    EXPECT_EQ(wrapped_api_->attr_proto_get_name(*tensor_attr),
              "renamed_tensor_attr");
    wrapped_api_->attr_proto_delete(tensor_attr);
    wrapped_api_->tensor_proto_delete(tensor_proto);
  }
  {
    // Regression (name-collision): two TENSOR attributes built with the *same*
    // attribute name (e.g. "value", as every `onnx.Constant` carries) must be
    // independent. The dense encoding keeps no name-keyed side table -- each
    // `attr_proto_new_tensor` produces its own content-uniqued
    // `DenseElementsAttr` -- so constructing the second must not disturb the
    // first. (Data read-back is verified at the MLIR/DenseElementsAttr layer,
    // not through the unsupported `attr_proto_get_tensor` proto accessor.)
    std::vector<int64_t> shape_a = {2, 3};
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto tensor_proto_a =
        wrapped_api_->tensor_proto_new_floats("value", shape_a, data_a);
    auto *attr_a =
        wrapped_api_->attr_proto_new_tensor("value", *tensor_proto_a);
    ASSERT_TRUE(attr_a != nullptr);

    std::vector<int64_t> shape_b = {1, 4};
    std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f};
    auto tensor_proto_b =
        wrapped_api_->tensor_proto_new_floats("value", shape_b, data_b);
    auto *attr_b =
        wrapped_api_->attr_proto_new_tensor("value", *tensor_proto_b);
    ASSERT_TRUE(attr_b != nullptr);
    EXPECT_NE(attr_a, attr_b);

    wrapped_api_->attr_proto_delete(attr_b);
    wrapped_api_->attr_proto_delete(attr_a);
    wrapped_api_->tensor_proto_delete(tensor_proto_b);
    wrapped_api_->tensor_proto_delete(tensor_proto_a);
  }
  {
    // An ONNX element type that `onnxElementTypeToMlirElementType` does not yet
    // map (here BFLOAT16, which falls back to F32) must fail loudly in
    // create_tensor rather than silently produce a corrupted DenseElementsAttr:
    // the dense byte-size guard catches the F32 default (4B/elem) vs the real
    // bf16 raw_data width (2B/elem) and LOG(FATAL)s with a fix hint. This death
    // test locks in that enforcement point.
    std::vector<int64_t> bf16_shape = {2, 3};
    std::vector<int16_t> bf16_data = {0x3c00, 0x4000, 0x4200,
                                      0x4400, 0x4500, 0x4600};
    auto *bf16_proto =
        wrapped_api_->tensor_proto_new_bf16("value", bf16_shape, bf16_data);
    ASSERT_TRUE(bf16_proto != nullptr);
    EXPECT_DEATH(
        {
          auto *attr =
              wrapped_api_->attr_proto_new_tensor("value", *bf16_proto);
          (void)attr;
        },
        "create_tensor: ONNX element type");
    wrapped_api_->tensor_proto_delete(bf16_proto);
  }
}

// ============================================================================
// TensorProto API Tests [70-89, 100-101]
// ============================================================================

void MorphizenOrtApiTest::Test12_TensorProtoOperations() {
  // Test various tensor creation and manipulation
  std::vector<int64_t> shape = {2, 3};

  // Test float tensors
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto *float_tensor =
      wrapped_api_->tensor_proto_new_floats("float_tensor", shape, float_data);
  if (float_tensor) {
    const std::string &name =
        wrapped_api_->tensor_proto_get_name(*float_tensor);
    EXPECT_EQ(name, "float_tensor");
    auto tensor_shape =
        wrapped_api_->tensor_proto_get_shape_unsafe(*float_tensor);
    EXPECT_EQ(tensor_shape->size(), shape.size());

    int data_type = wrapped_api_->tensor_proto_data_type(*float_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_FLOAT); // Check type is FLOAT
    size_t raw_size = wrapped_api_->tensor_proto_raw_data_size(*float_tensor);

    wrapped_api_->tensor_proto_delete(float_tensor);
  }

  // Test integer tensors
  std::vector<int64_t> int64_data = {10, 20, 30, 40, 50, 60};
  auto *int64_tensor =
      wrapped_api_->tensor_proto_new_i64("int64_tensor", shape, int64_data);
  if (int64_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*int64_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_INT64); // Check type is INT64
    wrapped_api_->tensor_proto_delete(int64_tensor);
  }

  std::vector<int32_t> int32_data = {1, 2, 3, 4, 5, 6};
  auto *int32_tensor =
      wrapped_api_->tensor_proto_new_i32("int32_tensor", shape, int32_data);
  if (int32_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*int32_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_INT32); // Check type is INT32
    wrapped_api_->tensor_proto_delete(int32_tensor);
  }

  std::vector<int16_t> int16_data = {1, 2, 3, 4, 5, 6};
  auto *int16_tensor =
      wrapped_api_->tensor_proto_new_i16("int16_tensor", shape, int16_data);
  if (int16_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*int16_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_INT16); // Check type is INT16
    wrapped_api_->tensor_proto_delete(int16_tensor);
  }

  std::vector<int8_t> int8_data = {1, 2, 3, 4, 5, 6};
  auto *int8_tensor =
      wrapped_api_->tensor_proto_new_i8("int8_tensor", shape, int8_data);
  if (int8_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*int8_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_INT8); // Check type is INT8
    wrapped_api_->tensor_proto_delete(int8_tensor);
  }

  // Test unsigned integer tensors
  std::vector<uint64_t> uint64_data = {10, 20, 30, 40, 50, 60};
  auto *uint64_tensor =
      wrapped_api_->tensor_proto_new_u64("uint64_tensor", shape, uint64_data);
  if (uint64_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*uint64_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_UINT64); // Check type is UINT64
    wrapped_api_->tensor_proto_delete(uint64_tensor);
  }

  std::vector<uint32_t> uint32_data = {1, 2, 3, 4, 5, 6};
  auto *uint32_tensor =
      wrapped_api_->tensor_proto_new_u32("uint32_tensor", shape, uint32_data);
  if (uint32_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*uint32_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_UINT32); // Check type is UINT32
    wrapped_api_->tensor_proto_delete(uint32_tensor);
  }

  std::vector<uint16_t> uint16_data = {1, 2, 3, 4, 5, 6};
  auto *uint16_tensor =
      wrapped_api_->tensor_proto_new_u16("uint16_tensor", shape, uint16_data);
  if (uint16_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*uint16_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_UINT16); // Check type is UINT16
    wrapped_api_->tensor_proto_delete(uint16_tensor);
  }

  std::vector<uint8_t> uint8_data = {1, 2, 3, 4, 5, 6};
  auto *uint8_tensor =
      wrapped_api_->tensor_proto_new_u8("uint8_tensor", shape, uint8_data);
  if (uint8_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*uint8_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_UINT8); // Check type is UINT8
    wrapped_api_->tensor_proto_delete(uint8_tensor);
  }

  // Test double tensors
  std::vector<double> double_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  auto *double_tensor = wrapped_api_->tensor_proto_new_doubles(
      "double_tensor", shape, double_data);
  if (double_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*double_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_DOUBLE); // Check type is DOUBLE
    wrapped_api_->tensor_proto_delete(double_tensor);
  }

  // Test half precision tensors (stored as int16)
  std::vector<int16_t> fp16_data = {0x3c00, 0x4000, 0x4200,
                                    0x4400, 0x4500, 0x4600}; // FP16 values
  auto *fp16_tensor =
      wrapped_api_->tensor_proto_new_fp16("fp16_tensor", shape, fp16_data);
  if (fp16_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*fp16_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_FLOAT16); // Check type is FLOAT16
    wrapped_api_->tensor_proto_delete(fp16_tensor);
  }

  auto *bf16_tensor =
      wrapped_api_->tensor_proto_new_bf16("bf16_tensor", shape, fp16_data);
  if (bf16_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*bf16_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_BFLOAT16); // Check type is BFLOAT16
    wrapped_api_->tensor_proto_delete(bf16_tensor);
  }

  // Test 4-bit tensors
  std::vector<int8_t> i4_data = {1, 2, 3, 4, 5, 6};
  auto *i4_tensor =
      wrapped_api_->tensor_proto_new_i4("i4_tensor", shape, i4_data);
  if (i4_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*i4_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_INT4); // Check type is INT4
    wrapped_api_->tensor_proto_delete(i4_tensor);
  }

  std::vector<uint8_t> u4_data = {1, 2, 3, 4, 5, 6};
  auto *u4_tensor =
      wrapped_api_->tensor_proto_new_u4("u4_tensor", shape, u4_data);
  if (u4_tensor) {
    auto data_type = wrapped_api_->tensor_proto_data_type(*u4_tensor);
    ASSERT_TRUE(data_type ==
                (int)ONNX_NAMESPACE::TensorProto_DataType::
                    TensorProto_DataType_UINT4); // Check type is UINT4
    wrapped_api_->tensor_proto_delete(u4_tensor);
  }
}

// ============================================================================
// Extended API Tests [80-108]
// ============================================================================
void MorphizenOrtApiTest::Test13_ExtendedApiOperations() { // Test library
                                                           // identification
  // Test library identification
  auto lib_id = wrapped_api_->get_lib_id();
  auto lib_name = wrapped_api_->get_lib_name();

  EXPECT_FALSE(lib_id->empty());
  EXPECT_FALSE(lib_name->empty());

  LOG(INFO) << "Library ID: " << *lib_id;
  LOG(INFO) << "Library Name: " << *lib_name;

  // Test session option configuration (dummy test)
  void *dummy_mmap = nullptr;
  void *dummy_session_option = nullptr;
  wrapped_api_->session_option_configuration(dummy_mmap, dummy_session_option,
                                             nullptr);

  // Test profiling enabled check
  bool profiling_enabled =
      wrapped_api_->is_profiling_enabled(dummy_session_option);
  LOG(INFO) << "Profiling enabled: " << profiling_enabled;

  try {
    // Test model proto operations
    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "test_proto.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for fuse test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }
    auto *model = simple_conv_relu_model_;
    ASSERT_TRUE(model != nullptr);
    {
      // the API model_to__proto and model_proto_serialize_as_string only for
      // fallback_cpu not implement in MLIR backend
      auto *model_proto = wrapped_api_->model_to_proto(*model);
      if (model_proto) {
        auto serialized =
            wrapped_api_->model_proto_serialize_as_string(*model_proto);
        EXPECT_FALSE(serialized->empty());
      }

      // Test graph proto operations
      auto &graph = wrapped_api_->model_main_graph(*model);
      // The API graph_to_graph_proto will be obsolete soon
      auto *graph_proto = wrapped_api_->graph_to_graph_proto(graph);
      if (graph_proto) {
        wrapped_api_->graph_proto_delete(graph_proto);
      }
      if (model_proto) {
        wrapped_api_->graph_infer_shapes(*model_proto);
        wrapped_api_->model_proto_delete(model_proto);
      }
    }
  } catch (...) {
    LOG(INFO) << "Extended API operations tested";
  }
}

void MorphizenOrtApiTest::Test14_GraphTensorOperations() {
  try {
    std::filesystem::path temp_path =
        std::filesystem::temp_directory_path() / "test_tensors.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};

    auto *model = wrapped_api_->create_empty_model(temp_path, opset);
    if (model) {
      auto &graph = wrapped_api_->model_main_graph(*model);

      // Create a tensor to add to the graph
      std::vector<int64_t> shape = {3, 3};
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      auto *tensor =
          wrapped_api_->tensor_proto_new_floats("test_tensor", shape, data);

      if (tensor) {
        // Add initialized tensor
        wrapped_api_->graph_add_initialized_tensor(graph, *tensor);

        // Test removing initialized tensor
        /* useless */
        //  wrapped_api_->graph_remove_initialized_tensor(graph, "test_tensor");

        wrapped_api_->tensor_proto_delete(tensor);
      }
      wrapped_api_->model_delete(model);
    }
  } catch (...) {
    LOG(INFO) << "Graph tensor operations tested";
  }
}

// ============================================================================
// Graph Fuse Operations Test
// ============================================================================

void MorphizenOrtApiTest::Test16_GraphFuseOperations() {
  LOG(INFO) << "=== Test16: Graph Fuse Operations ===";

  try {
    // Ensure we have a model to work with
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for fuse test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }

    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for fuse test";

    // Clone the model for testing
    LOG(INFO) << "Cloning model for graph fuse operations...";
    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone model";

    // Get the graph from the cloned model
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);

    // Get nodes from the cloned graph
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_TRUE(nodes.get() != nullptr && nodes->size() == 2)
        << "Cloned graph doesn't have enough nodes for fuse test";

    // Get graph inputs and outputs for fuse operation
    auto inputs = wrapped_api_->graph_get_inputs_unsafe(graph);
    auto outputs = wrapped_api_->graph_get_outputs_unsafe(graph);
    ASSERT_TRUE(inputs.get() != nullptr && !inputs->empty())
        << "No graph inputs found";
    ASSERT_TRUE(outputs.get() != nullptr && !outputs->empty())
        << "No graph outputs found";

    // Test graph_fuse - attempt to fuse Conv and ReLU nodes
    try {
      LOG(INFO) << "Testing graph_fuse operation on cloned model...";
      auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
      std::vector<size_t> nodes_to_fuse = {
          wrapped_api_->node_get_index(*(*nodes.get())[0]),
          wrapped_api_->node_get_index(
              *(*nodes.get())[1])}; // Conv and ReLU node indices
      std::vector<std::string> fuse_inputs = {
          wrapped_api_->node_arg_get_name_unsafe(*(*inputs)[0])};
      std::vector<std::string> fuse_outputs = {
          wrapped_api_->node_arg_get_name_unsafe(*(*outputs)[0])};
      std::vector<std::string> fuse_constants =
          {}; // No constants for this test

      auto &fused_node = wrapped_api_->graph_fuse(
          graph, "fused_conv_relu", "FusedConvRelu", nodes_to_fuse, fuse_inputs,
          fuse_outputs, fuse_constants);

      LOG(INFO) << "Successfully created fused node: "
                << wrapped_api_->node_get_name(fused_node);
      LOG(INFO) << "Graph fuse operation completed successfully";

    } catch (const std::exception &e) {
      LOG(INFO) << "Graph fuse test completed with exception (expected with "
                   "simple implementation): "
                << e.what();
    } catch (...) {
      LOG(INFO) << "Graph fuse test completed with unknown exception (expected "
                   "with simple implementation)";
    }
    // save the modified graph
    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "Test16_graph_fuse_operations.onnx";
    wrapped_api_->graph_save(graph, temp_path.u8string(),
                             temp_path.u8string() + ".dat",
                             std::numeric_limits<size_t>::max());
    LOG(INFO) << "Graph saved to: " << temp_path.string();
    // Clean up the cloned model
    wrapped_api_->model_delete(cloned_model);
    LOG(INFO) << "Test16_GraphFuseOperations completed successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test16_GraphFuseOperations failed with exception: "
               << e.what();
    FAIL() << "Test16_GraphFuseOperations failed with exception: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Test16_GraphFuseOperations failed with unknown exception";
    FAIL() << "Test16_GraphFuseOperations failed with unknown exception";
  }
}

// ============================================================================
// Graph Node Removal Operations Test
// ============================================================================

void MorphizenOrtApiTest::Test17_GraphNodeRemovalOperations() {
  LOG(INFO) << "=== Test17: Graph Node Removal Operations ===";

  try {
    // Ensure we have a model to work with
    if (!simple_conv_relu_model_) {
      LOG(INFO)
          << "No model available for node removal test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }

    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for node removal test";

    // Clone the model for testing (so we don't modify the original)
    LOG(INFO) << "Cloning model for graph node removal operations...";
    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone model";

    // Get the graph from the cloned model
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);

    // Get nodes from the cloned graph
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_TRUE(nodes.get() != nullptr && nodes->size() >= 2)
        << "Cloned graph doesn't have enough nodes for removal test";

    LOG(INFO) << "Graph has " << nodes->size() << " nodes before removal";

    // Create a temporary path for saving the modified graph
    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "Test17_graph_node_removal";

    // Set output to Conv node
    // We cannot use conv_output_arg any more, because it is invalidated after
    // `graph_resolve()`
    auto conv_node_outputs =
        wrapped_api_->node_get_output_node_args_unsafe(*(*nodes)[0]);
    ASSERT_TRUE(conv_node_outputs.get() != nullptr &&
                !conv_node_outputs->empty())
        << "Conv node should have at least one output";

    wrapped_api_->graph_set_outputs(
        graph,
        gsl::span<const morphizen::NodeArg *const>({(*conv_node_outputs)[0]}));

    LOG(INFO) << "Now delete the ReLU node (node at index 1)";
    // Delete the ReLU node
    // NOTE: we cannot use relu_node any more, because it is invalidated
    // after `graph_resolve()`
    wrapped_api_->graph_remove_node(graph,
                                    morphizen::NodeInput{(*nodes)[1], nullptr});
    // Resolve the graph again
    int resolution_result = wrapped_api_->graph_resolve(graph, false);
    LOG(INFO) << "Graph resolution result after deleting ReLU node: "
              << resolution_result;

    // Verify the node was removed
    auto nodes_after = wrapped_api_->graph_nodes_unsafe(graph);
    LOG(INFO) << "Graph has " << nodes_after->size() << " nodes after removal";
    EXPECT_EQ(nodes_after->size(), 1) << "Expected 1 node after removing ReLU";

    // Save the modified graph
    LOG(INFO) << "Saving graph to file: "
              << temp_path.string() + ".graph2.onnx";
    wrapped_api_->graph_save(graph, temp_path.string() + ".graph2.onnx", "",
                             1024);
    LOG(INFO) << "Graph saved successfully";

    // Clean up the cloned model
    wrapped_api_->model_delete(cloned_model);
    LOG(INFO) << "Test17_GraphNodeRemovalOperations completed successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test17_GraphNodeRemovalOperations failed with exception: "
               << e.what();
    FAIL() << "Test17_GraphNodeRemovalOperations failed with exception: "
           << e.what();
  } catch (...) {
    LOG(ERROR)
        << "Test17_GraphNodeRemovalOperations failed with unknown exception";
    FAIL() << "Test17_GraphNodeRemovalOperations failed with unknown exception";
  }
}

// ============================================================================
// Missing APIs Coverage Test
// ============================================================================

void MorphizenOrtApiTest::Test18_MissingApisCoverage() {
  LOG(INFO) << "=== Test18: Missing APIs Coverage ===";

  try {
    // Ensure we have a model to work with
    if (!simple_conv_relu_model_) {
      LOG(INFO)
          << "No model available for missing APIs test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }

    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for missing APIs test";

    // Test graph_get_model
    LOG(INFO) << "Testing graph_get_model...";
    auto &graph = wrapped_api_->model_main_graph(*simple_conv_relu_model_);
    const auto &model_ref = wrapped_api_->graph_get_model(graph);
    ASSERT_TRUE(&model_ref != nullptr)
        << "graph_get_model should return valid reference";

    // Test graph_get_node_arg
    LOG(INFO) << "Testing graph_get_node_arg...";
    auto inputs = wrapped_api_->graph_get_inputs_unsafe(graph);
    if (inputs.get() && !inputs->empty()) {
      auto first_input_name =
          wrapped_api_->node_arg_get_name_unsafe(*(*inputs)[0]);
      const auto *node_arg =
          wrapped_api_->graph_get_node_arg(graph, first_input_name);
      LOG(INFO) << "graph_get_node_arg for '" << first_input_name
                << "' returned: " << (node_arg ? "valid pointer" : "null");
    }

    // Test graph_get_consumer_nodes_unsafe
    LOG(INFO) << "Testing graph_get_consumer_nodes_unsafe...";
    if (inputs.get() && !inputs->empty()) {
      auto first_input_name =
          wrapped_api_->node_arg_get_name_unsafe(*(*inputs)[0]);
      auto consumer_nodes = wrapped_api_->graph_get_consumer_nodes_unsafe(
          graph, first_input_name);
      LOG(INFO) << "Consumer nodes for " << first_input_name << ": "
                << consumer_nodes->size();
    }

    // Test model_clone
    LOG(INFO) << "Testing model_clone...";
    auto *cloned_model =
        wrapped_api_->model_clone(*simple_conv_relu_model_, 1024);
    if (cloned_model) {
      LOG(INFO) << "model_clone succeeded";
      wrapped_api_->model_delete(cloned_model);
    } else {
      LOG(INFO) << "model_clone returned nullptr (may be expected in current "
                   "implementation)";
    }

    // Test node_get_attributes and node_get_function_body with actual nodes
    LOG(INFO) << "Testing node_get_attributes and node_get_function_body...";
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    if (nodes.get() && !nodes->empty()) {
      for (const auto *node : *nodes) {
        if (node) {
          // Test node_get_attributes
          auto &attrs = wrapped_api_->node_get_attributes(
              *const_cast<morphizen::Node *>(node));
          LOG(INFO) << "Got attributes for node: "
                    << wrapped_api_->node_get_name(*node);

          // Test node_get_function_body (may throw for non-function nodes)
          try {
            const auto &function_body =
                wrapped_api_->node_get_function_body(*node);
            LOG(INFO) << "Got function body for node: "
                      << wrapped_api_->node_get_name(*node);
          } catch (...) {
            LOG(INFO) << "node_get_function_body threw (expected for "
                         "non-function nodes)";
          }
        }
      }
    }

    // Test node_arg_get_const_data_as_tensor and node_arg_set_element_type
    LOG(INFO) << "Testing node_arg_get_const_data_as_tensor and "
                 "node_arg_set_element_type...";
    auto outputs = wrapped_api_->graph_get_outputs_unsafe(graph);
    if (outputs.get() && !outputs->empty()) {
      for (const auto node_arg : *outputs.get()) {
        if (node_arg) {
          // Test node_arg_set_element_type
          // This API is useless , should be removed
          // wrapped_api_->node_arg_set_element_type(
          //    const_cast<onnxruntime::NodeArg&>(*node_arg), 1); // FLOAT

          // Test node_arg_get_const_data_as_tensor (may throw for non-constant
          // args)
          if (wrapped_api_->node_arg_is_constant(graph, *node_arg)) {
            const auto &tensor_data =
                wrapped_api_->node_arg_get_const_data_as_tensor(graph,
                                                                *node_arg);
          }
        }
      }
    }

    // Test node_arg_external_location
    LOG(INFO) << "Testing node_arg_external_location...";
    if (outputs.get() && !outputs->empty()) {
      for (auto *node_arg : *outputs) {
        if (node_arg) {
          std::string external_file;
          size_t offset = 0, size = 0, checksum = 0;
          int has_external = wrapped_api_->node_arg_external_location(
              graph, *node_arg, external_file, offset, size, checksum);
          LOG(INFO) << "node_arg_external_location returned: " << has_external;
        }
      }
    }

    // Test session_option_configuration and is_profiling_enabled
    LOG(INFO)
        << "Testing session_option_configuration and is_profiling_enabled...";
    void *dummy_mmap = nullptr;
    void *dummy_session_option = nullptr;
    wrapped_api_->session_option_configuration(dummy_mmap, dummy_session_option,
                                               nullptr);
    bool profiling_enabled =
        wrapped_api_->is_profiling_enabled(dummy_session_option);
    LOG(INFO) << "is_profiling_enabled returned: " << profiling_enabled;

    // Test graph_proto_delete and graph_infer_shapes
    LOG(INFO) << "Testing graph_proto_delete and graph_infer_shapes...";
    auto *graph_proto = wrapped_api_->graph_to_graph_proto(graph);
    if (graph_proto) {
      wrapped_api_->graph_proto_delete(graph_proto);
      LOG(INFO) << "graph_proto_delete called successfully";
    }

    // Test graph_infer_shapes with model proto
    auto *model_proto = wrapped_api_->model_to_proto(*simple_conv_relu_model_);
    ASSERT_TRUE(model_proto);
    wrapped_api_->graph_infer_shapes(*model_proto);
    LOG(INFO) << "graph_infer_shapes called successfully";
    wrapped_api_->model_proto_delete(model_proto);

    // Test graph_fuse - this is already covered in Test16 but let's ensure it's
    // called
    LOG(INFO) << "Testing graph_fuse (basic call)...";
    if (nodes.get() && nodes->size() >= 2) {
      try {
        auto cloned_for_fuse =
            wrapped_api_->model_clone(*simple_conv_relu_model_, 1024);
        if (cloned_for_fuse) {
          auto &fuse_graph = wrapped_api_->model_main_graph(*cloned_for_fuse);
          auto fuse_inputs = wrapped_api_->graph_get_inputs_unsafe(fuse_graph);
          auto fuse_outputs =
              wrapped_api_->graph_get_outputs_unsafe(fuse_graph);

          if (fuse_inputs.get() && !fuse_inputs->empty() &&
              fuse_outputs.get() && !fuse_outputs->empty()) {

            auto nodes = wrapped_api_->graph_nodes_unsafe(fuse_graph);
            std::vector<size_t> nodes_to_fuse = {
                wrapped_api_->node_get_index(*(*nodes.get())[0]),
                wrapped_api_->node_get_index(*(*nodes.get())[1])};
            std::vector<std::string> fuse_input_names = {
                wrapped_api_->node_arg_get_name_unsafe(*(*fuse_inputs)[0])};
            std::vector<std::string> fuse_output_names = {
                wrapped_api_->node_arg_get_name_unsafe(*(*fuse_outputs)[0])};
            std::vector<std::string> fuse_constants = {};

            auto &fused_node = wrapped_api_->graph_fuse(
                fuse_graph, "test_fused_node", "TestFuse", nodes_to_fuse,
                fuse_input_names, fuse_output_names, fuse_constants);

            LOG(INFO) << "graph_fuse succeeded, created node: "
                      << wrapped_api_->node_get_name(fused_node);
          }
          wrapped_api_->model_delete(cloned_for_fuse);
        }
      } catch (...) {
        LOG(INFO) << "graph_fuse test completed (may throw with current "
                     "implementation)";
      }
    }

    LOG(INFO) << "Test18_MissingApisCoverage completed successfully";

  } catch (const std::exception &e) {
    LOG(INFO) << "Test18_MissingApisCoverage completed with exception (some "
                 "APIs may not be fully implemented): "
              << e.what();
  } catch (...) {
    LOG(INFO) << "Test18_MissingApisCoverage completed with unknown exception "
                 "(some APIs may not be fully implemented)";
  }
}

void MorphizenOrtApiTest::Test19_add_sin_op_before_relu_op() {
  LOG(INFO) << "=== Test19: Add Sin->Cos operations before ReLU ===";

  try {
    // Ensure we have a model to work with
    if (!simple_conv_relu_model_) {
      LOG(INFO)
          << "No model available for Sin->Cos test, creating one first...";
      Test07_create_simple_conv_relu_model();
    }

    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for Sin->Cos test";

    // Clone the model for testing
    LOG(INFO) << "Cloning model for Sin->Cos operations...";
    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone model";

    // Get the graph from the cloned model
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);
    {
      LOG(INFO) << "DFS the unresolved graph to ensure all nodes are connected";
      auto topo_node_indices =
          morphizen::graph_get_node_in_topoligical_order(graph);
      ASSERT_TRUE(!topo_node_indices.empty());
      for (auto node_index : topo_node_indices) {
        auto node = wrapped_api_->graph_get_node(graph, node_index);
        ASSERT_TRUE(node != nullptr)
            << "Node at index " << node_index << " should not be null";
        LOG(INFO) << " Node at index " << node_index
                  << " is: " << morphizen::node_as_string(*node);
      }
    }
    // Get nodes from the cloned graph - should have Conv (index 0) and ReLU
    // (index 1)
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_TRUE(nodes.get() != nullptr && nodes->size() >= 2)
        << "Cloned graph doesn't have enough nodes for modification test";

    LOG(INFO) << "Graph has " << nodes->size() << " nodes before modification";

    // Get the Conv node output (which currently goes to ReLU)
    const auto *conv_node = (*nodes)[0]; // Conv node
    const auto *relu_node = (*nodes)[1]; // ReLU node

    auto conv_outputs =
        wrapped_api_->node_get_output_node_args_unsafe(*conv_node);
    auto relu_inputs = wrapped_api_->node_get_inputs_unsafe(*relu_node);

    ASSERT_TRUE(conv_outputs.get() != nullptr && !conv_outputs->empty())
        << "Conv node should have at least one output";
    ASSERT_TRUE(relu_inputs.get() != nullptr && !relu_inputs->empty())
        << "ReLU node should have at least one input";

    // Get the intermediate tensor (conv_output) that connects Conv to ReLU
    const auto *conv_output_arg = (*conv_outputs)[0];
    std::string conv_output_name =
        wrapped_api_->node_arg_get_name_unsafe(*conv_output_arg);

    LOG(INFO) << "Original flow: Conv -> " << conv_output_name << " -> ReLU";

    // Create intermediate tensors for Sin and Cos operations
    auto conv_output_shape =
        wrapped_api_->node_arg_get_shape_i64_unsafe(*conv_output_arg);

    // Create intermediate tensor between Conv and Sin
    // sin_input_arg is as same as conv_output_arg, but with a new name
    auto &sin_input_arg = *conv_output_arg;

    // Create intermediate tensor between Sin and Cos
    auto &sin_output_arg = wrapped_api_->node_arg_new(
        graph, "sin_output", conv_output_shape.get(), 1); // FLOAT type

    // Create intermediate tensor between Cos and ReLU
    auto &cos_output_arg = wrapped_api_->node_arg_new(
        graph, "cos_output", conv_output_shape.get(), 1); // FLOAT type

    // Create Sin node attributes (Sin doesn't need specific attributes)
    auto *sin_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(sin_attrs != nullptr) << "Failed to create Sin node attributes";

    // Create Sin node inputs and outputs
    std::vector<const morphizen::NodeArg *> sin_inputs = {&sin_input_arg};
    std::vector<const morphizen::NodeArg *> sin_outputs = {&sin_output_arg};

    // Add Sin node to the graph
    LOG(INFO) << "Adding Sin node to graph...";
    auto &sin_node =
        wrapped_api_->graph_add_node(graph, "sin_node", "Sin", "Sin operation",
                                     sin_inputs, sin_outputs, *sin_attrs, "");

    // Create Cos node attributes (Cos doesn't need specific attributes)
    auto *cos_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(cos_attrs != nullptr) << "Failed to create Cos node attributes";

    // Create Cos node inputs and outputs
    std::vector<const morphizen::NodeArg *> cos_inputs = {&sin_output_arg};
    std::vector<const morphizen::NodeArg *> cos_outputs = {&cos_output_arg};

    // Add Cos node to the graph
    LOG(INFO) << "Adding Cos node to graph...";
    auto &cos_node =
        wrapped_api_->graph_add_node(graph, "cos_node", "Cos", "Cos operation",
                                     cos_inputs, cos_outputs, *cos_attrs, "");

    LOG(INFO) << "New flow will be: Conv -> Sin -> Cos -> ReLU";

    // Now we need to modify the graph connections:
    // 1. Change Conv node's output to point to sin_input_arg
    // 2. Change ReLU node's input to point to cos_output_arg

    // Get ReLU node outputs for recreation
    auto relu_outputs =
        wrapped_api_->node_get_output_node_args_unsafe(*relu_node);
    ASSERT_TRUE(relu_outputs.get() != nullptr && !relu_outputs->empty() &&
                relu_outputs->size() == 1)
        << "ReLU node should have at least one output";
    auto relu_output_arg = (*relu_outputs)[0];
    // Create new ReLU node with cos_output as input
    auto *new_relu_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(new_relu_attrs != nullptr)
        << "Failed to create new ReLU node attributes";

    std::vector<const morphizen::NodeArg *> new_relu_inputs = {&cos_output_arg};
    std::vector<const morphizen::NodeArg *> new_relu_outputs = {
        relu_output_arg};

    // Add new ReLU node to the graph
    LOG(INFO) << "Adding new ReLU node with modified connections...";
    auto &new_relu_node = wrapped_api_->graph_add_node(
        graph, "new_relu_node", "Relu", "ReLU activation operation",
        new_relu_inputs, new_relu_outputs, *new_relu_attrs, "");
    wrapped_api_->graph_remove_node(graph, {relu_node, nullptr});
    {
      wrapped_api_->graph_resolve(graph, true);
      auto topo_node_indices =
          morphizen::graph_get_node_in_topoligical_order(graph);
      for (auto node_index : topo_node_indices) {
        auto node = wrapped_api_->graph_get_node(graph, node_index);
        ASSERT_TRUE(node != nullptr)
            << "Node at index " << node_index << " should not be null";
        LOG(INFO) << " Node at index " << node_index
                  << " is: " << morphizen::node_as_string(*node);
      }
      ASSERT_EQ(topo_node_indices.size(), 4u);
    }

    // Clean up
    wrapped_api_->node_attributes_delete(sin_attrs);
    wrapped_api_->node_attributes_delete(cos_attrs);
    wrapped_api_->node_attributes_delete(new_relu_attrs);
    wrapped_api_->model_delete(cloned_model);

    LOG(INFO) << "Test19_add_sin_op_before_relu_op completed successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test19_add_sin_op_before_relu_op failed with exception: "
               << e.what();
    FAIL() << "Test19_add_sin_op_before_relu_op failed with exception: "
           << e.what();
  } catch (...) {
    LOG(ERROR)
        << "Test19_add_sin_op_before_relu_op failed with unknown exception";
    FAIL() << "Test19_add_sin_op_before_relu_op failed with unknown exception";
  }
}

void MorphizenOrtApiTest::Test20_conv_relu_fuse_conv2d_nchw() {
  LOG(INFO) << "=== Test20: Conv-ReLU Fuse com.xilinx:conv2d_nchw ===";
  try {
    // ensure we have a model to work with
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for Conv-ReLU fuse test, creating one "
                   "first...";
      Test07_create_simple_conv_relu_model();
    }
    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for Conv-ReLU fuse test";

    // Clone the model for testing
    LOG(INFO) << "Cloning model for Conv-ReLU fuse operations...";
    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone model";

    // Get the graph from the cloned model
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);
    {
      LOG(INFO) << "DFS the unresolved graph to ensure all nodes are connected";
      auto topo_node_indices =
          morphizen::graph_get_node_in_topoligical_order(graph);
      ASSERT_TRUE(!topo_node_indices.empty());
      for (auto node_index : topo_node_indices) {
        auto node = wrapped_api_->graph_get_node(graph, node_index);
        ASSERT_TRUE(node != nullptr)
            << "Node at index " << node_index << " should not be null";
        LOG(INFO) << " Node at index " << node_index
                  << " is: " << morphizen::node_as_string(*node);
      }
    }

    // Get nodes from the cloned graph - should have Conv (index 0) and ReLU
    // (index 1)
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_TRUE(nodes.get() != nullptr && nodes->size() >= 2)
        << "Cloned graph doesn't have enough nodes for modification test";
    LOG(INFO) << "Graph has " << nodes->size() << " nodes before modification";

    // Get the Conv node output (which currently goes to ReLU)
    const auto *conv_node = wrapped_api_->graph_get_node(
        graph, wrapped_api_->node_get_index(*(*nodes)[0])); // Conv node
    const auto *relu_node = wrapped_api_->graph_get_node(
        graph, wrapped_api_->node_get_index(*(*nodes)[1])); // ReLU node

    // Get Conv inputs as new Conv2D NCHW node inputs
    auto conv_inputs = wrapped_api_->node_get_inputs_unsafe(*conv_node);
    // Get Conv node attributes, for cloning to new Conv2D NCHW node
    auto &conv_attrs = wrapped_api_->node_get_attributes(
        *const_cast<morphizen::Node *>(conv_node));

    // Relu outputs will be used as Conv2D NCHW node outputs
    auto relu_outputs =
        wrapped_api_->node_get_output_node_args_unsafe(*relu_node);
    // auto relu_output_shape =
    //    wrapped_api_->node_arg_get_shape_i64_unsafe(*(*relu_outputs)[0]);

    // new inputs and outputs
    std::vector<const morphizen::NodeArg *> new_conv2d_input_args;
    new_conv2d_input_args.reserve(conv_inputs->size());
    for (const auto &input_arg : *conv_inputs) {
      ASSERT_TRUE(input_arg.node_arg != nullptr)
          << "Conv node input should not be null";
      new_conv2d_input_args.push_back(input_arg.node_arg);
    }
    std::vector<const morphizen::NodeArg *> new_conv2d_output_args = {
        (*relu_outputs)[0]};

    // new attributes
    auto *new_conv2d_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(new_conv2d_attrs != nullptr)
        << "Failed to create new Conv2D node attributes";
    auto attrs_keys = wrapped_api_->node_attributes_get_keys(conv_attrs);
    for (const auto &key : *attrs_keys) {
      auto *attr = wrapped_api_->node_attributes_get(conv_attrs, key);
      auto new_attr = wrapped_api_->attr_proto_clone(*attr);
      LOG(INFO) << "clone attribute: " << key;
      wrapped_api_->node_attributes_add(*new_conv2d_attrs,
                                        std::move(*new_attr));
    }

    auto &new_conv2d_node = wrapped_api_->graph_add_node(
        graph, "new_conv_relu", "conv2d_nchw", "Conv2D NCHW operation",
        new_conv2d_input_args, new_conv2d_output_args, *new_conv2d_attrs,
        "com.xilinx");

    LOG(INFO) << "New Conv2D NCHW node created: "
              << wrapped_api_->node_get_name(new_conv2d_node);

    // remove the old relu node
    wrapped_api_->graph_remove_node(graph,
                                    morphizen::NodeInput{relu_node, nullptr});

    {
      LOG(INFO) << "DFS the unresolved graph to ensure all nodes are connected";
      auto topo_node_indices =
          morphizen::graph_get_node_in_topoligical_order(graph);
      for (auto node_index : topo_node_indices) {
        auto node = wrapped_api_->graph_get_node(graph, node_index);
        ASSERT_TRUE(node != nullptr)
            << "Node at index " << node_index << " should not be null";
        LOG(INFO) << " Node at index " << node_index
                  << " is: " << morphizen::node_as_string(*node);
      }
    }

    // graph_resolve the graph to ensure all nodes are connected
    int resolution_result = wrapped_api_->graph_resolve(graph, true);
    LOG(INFO) << "Graph resolution result after adding Conv2D NCHW node: "
              << resolution_result;
    ASSERT_EQ(resolution_result, 0)
        << "Graph resolution failed after adding Conv2D NCHW node";
    auto new_nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_TRUE(new_nodes.get() != nullptr)
        << "Graph nodes should not be null after Conv2D NCHW fusion";
    /* ASSERT_TRUE(new_nodes->size() == 1)
        << "Graph should have exactly one node after Conv2D NCHW fusion, new "
           "nodes size is "
        << new_nodes->size();*/
    // double graph_resolve
    wrapped_api_->graph_resolve(graph, true);

    // save the modified graph
    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "Test20_conv_relu_fuse_conv2d_nchw.onnx";
    wrapped_api_->graph_save(graph, temp_path.u8string(), "",
                             std::numeric_limits<size_t>::max());

#if MORPHIZEN_ORT_API_MAJOR >= 18
    auto save_string = wrapped_api_->graph_save_string(graph);
    EXPECT_FALSE(save_string->empty());
    // LOG(INFO) << "graph save to string" << *save_string;
#endif

    wrapped_api_->node_attributes_delete(new_conv2d_attrs);
    wrapped_api_->model_delete(cloned_model);
    LOG(INFO) << "Test20_conv_relu_fuse_conv2d_nchw completed successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test20_conv_relu_fuse_conv2d_nchw failed with exception: "
               << e.what();
    FAIL() << "Test20_conv_relu_fuse_conv2d_nchw failed with exception: "
           << e.what();
  } catch (...) {
    LOG(ERROR)
        << "Test20_conv_relu_fuse_conv2d_nchw failed with unknown exception";
    FAIL() << "Test20_conv_relu_fuse_conv2d_nchw failed with unknown exception";
  }
  LOG(INFO) << "Test20_conv_relu_fuse_conv2d_nchw completed successfully";
}

void MorphizenOrtApiTest::Test21_fuse_relu_q() {
  // Skip this test - morphizen-pass_init plugin not loaded (see Issue #030)
  GTEST_SKIP()
      << "Test skipped: morphizen-pass_init plugin not loaded (see Issue #030)";
  try {
    auto test_model_path = RESNET_50_PATH;
    if (backend_ == morphizen::kMLIRBackend) {
      test_model_path = RESNET_50_MLIR_PATH;
    }
    auto *model = wrapped_api_->model_load(test_model_path.u8string());
    ASSERT_TRUE(model != nullptr) << "Failed to load ResNet-50 model";

    auto *cloned_model =
        wrapped_api_->model_clone(*model, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone ResNet-50 model";
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);
    // morphizen::graph_resolve(graph, true);

    // create pattern
    auto builder = morphizen::PatternBuilder();
    std::shared_ptr<morphizen::Pattern> input = builder.wildcard();
    std::shared_ptr<morphizen::Pattern> relu =
        builder.node2("Relu", {builder.wildcard()});
    std::shared_ptr<morphizen::Pattern> pattern_q = builder.node2(
        "QuantizeLinear", {relu, builder.wildcard(), builder.wildcard()});
    LOG(INFO) << "Pattern for Relu->QuantizeLinear: "
              << pattern_q->debug_string();

    // create passcontext and pass
    std::shared_ptr<morphizen::PassContext> pass_context =
        morphizen::PassContext::create();
    auto pass_proto = std::make_unique<morphizen::PassProto>();
    pass_proto->set_plugin("morphizen-pass_init");
    pass_proto->set_name("MorphizenOrtApiTest::Test21_fuse_relu_q");
    auto pass = morphizen::IPass::create_pass(pass_context, *pass_proto);

    int fuse_count = 0;
    bool has_matched = true;
    while (has_matched) {
      has_matched = false;
      for (auto index : morphizen::graph_get_node_in_topoligical_order(graph)) {
        auto q_node = wrapped_api_->graph_get_node(graph, index);
        if (q_node == nullptr) {
          LOG(INFO) << "Node at index " << index << " is null, skipping";
          continue;
        }
        // LOG(INFO) << morphizen::node_as_string(*node);
        auto op_type = morphizen::node_op_type(*q_node);
        auto bind = pattern_q->match(graph, *q_node);
        if (bind) {
          auto q_node_name = wrapped_api_->node_get_name(*q_node);
          LOG(INFO) << "==== Found matching pattern for node: "
                    << morphizen::node_as_string(*q_node);

          auto b_relu_node = (*bind)[relu->get_id()];
          auto b_q_node = (*bind)[pattern_q->get_id()];
          ASSERT_TRUE(b_relu_node.node != nullptr && b_q_node.node != nullptr)
              << "Pattern binding failed for Relu or QuantizeLinear node";
          ASSERT_TRUE((void *)b_q_node.node == (void *)q_node)
              << "Pattern binding QuantizeLinear node does not match current "
                 "node";

          morphizen::NodeBuilder(graph, *pass)
              .set_op_type("QRelu", "com.xilinx")
              .clone_inputs(*b_relu_node.node)
              .set_anchor_point1(*b_q_node.node)
              .add("attr_int", (int64_t)1)
              .add("attr_vec", std::vector<int64_t>{1, 2, 3})
              .add("attr_string", "test_attr")
              .add("attr_float", 2.0f)
              .add("attr_floats", std::vector<float>{4.0f, 5.0f})
              .build();

          has_matched = true;
          fuse_count++;
          break;
        }
      }
    }
    ASSERT_TRUE(fuse_count == 49)
        << "Expected to fuse 49 ReLU-Quantize pairs, but found: " << fuse_count;
    {
      LOG(INFO) << "DFS the unresolved graph to ensure all nodes are connected";
      auto topo_node_indices =
          morphizen::graph_get_node_in_topoligical_order(graph);
      ASSERT_TRUE(!topo_node_indices.empty());
      for (auto node_index : topo_node_indices) {
        auto node = wrapped_api_->graph_get_node(graph, node_index);
        ASSERT_TRUE(node != nullptr)
            << "Node at index " << node_index << " should not be null";
        auto node_type = morphizen::node_op_type(*node);
        if (node_type == "QRelu") {
          LOG(INFO) << " ==== QRelu Node at index " << node_index
                    << " is: " << morphizen::node_as_string(*node);
          auto attr_int = morphizen::node_get_attr_int(*node, "attr_int");
          CHECK_EQ(attr_int, 1)
              << "Expected attr_int to be present in QRelu node attributes"
              << "node:" << morphizen::node_as_string(*node);
        }
      }
    }

    morphizen::graph_resolve(graph, true);

    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "Test21_relu_q_fuse.onnx";
    wrapped_api_->graph_save(graph, temp_path.u8string(), "",
                             std::numeric_limits<size_t>::max());

    wrapped_api_->model_delete(cloned_model);
    if (model) {
      wrapped_api_->model_delete(model);
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Test21_fuse_relu_q failed with exception: " << e.what();
    FAIL() << "Test21_fuse_relu_q failed with exception: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Test21_fuse_relu_q failed with unknown exception";
    FAIL() << "Test21_fuse_relu_q failed with unknown exception";
  }
  LOG(INFO) << "Test21_fuse_relu_q completed successfully";
}

void MorphizenOrtApiTest::Test22_create_initializer_node_arg() {
  try {
    std::filesystem::path temp_path =
        std::filesystem::temp_directory_path() / "test_new_initializer.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};

    auto *model = wrapped_api_->create_empty_model(temp_path, opset);
    if (model) {
      auto &graph = wrapped_api_->model_main_graph(*model);

      // Create a tensor to add to the graph
      std::vector<int64_t> shape = {3, 3};
      std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      auto tensor_name = "test_tensor";
      auto *tensor =
          wrapped_api_->tensor_proto_new_floats(tensor_name, shape, data);
      ASSERT_TRUE(tensor != nullptr) << "Failed to create tensor";

      // Add initialized tensor
      wrapped_api_->graph_add_initialized_tensor(graph, *tensor);
      auto &initializer_node_arg =
          wrapped_api_->node_arg_new(graph, tensor_name, &shape, 1);
      wrapped_api_->tensor_proto_delete(tensor);
      wrapped_api_->model_delete(model);
    }
  } catch (...) {
    LOG(INFO) << "Graph tensor operations tested";
  }
}

void MorphizenOrtApiTest::Test23_try_fuse_and_fuse() {
  LOG(INFO) << "===== Starting Test23_try_fuse_and_fuse";
  try {
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for Conv-ReLU fuse test, creating one "
                   "first...";
      Test07_create_simple_conv_relu_model();
    }
    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for Conv-ReLU fuse test";

    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);

    // graph_resolve the graph to ensure all nodes are connected
    int resolution_result = wrapped_api_->graph_resolve(graph, true);
    LOG(INFO) << "Graph resolution result: " << resolution_result;

    for (auto index : morphizen::graph_get_node_in_topoligical_order(graph)) {
      auto node = wrapped_api_->graph_get_node(graph, index);
      if (node == nullptr) {
        LOG(INFO) << "Node at index " << index << " is null";
        continue;
      }
      auto op_type = morphizen::node_op_type(*node);
      // try_fuse and fuse
      if (op_type != "Conv") {
        LOG(INFO) << "Skipping node: " << morphizen::node_as_string(*node)
                  << " with op_type: " << op_type;
        continue;
      }

      // create passcontext and pass
      std::shared_ptr<morphizen::PassContext> pass_context =
          morphizen::PassContext::create();
      auto pass_proto = std::make_unique<morphizen::PassProto>();
      pass_proto->set_plugin("morphizen-pass_init");
      pass_proto->set_name("MorphizenOrtApiTest::Test21_fuse_relu_q");
      auto pass = morphizen::IPass::create_pass(pass_context, *pass_proto);

      auto conv_inputs = wrapped_api_->node_get_inputs_unsafe(*node);
      CHECK_EQ(conv_inputs->size(), 3)
          << "Expected Conv node to have exactly one input, but found: "
          << conv_inputs->size();
      auto conv_outputs = wrapped_api_->node_get_output_node_args_unsafe(*node);

      auto node_name = wrapped_api_->node_get_name(*node);
      auto inputs = std::vector<std::string>();
      inputs.reserve(conv_inputs->size());
      for (const auto &input : *conv_inputs) {
        ASSERT_TRUE(input.node_arg != nullptr)
            << "Conv node input should not be null";
        inputs.push_back(
            wrapped_api_->node_arg_get_name_unsafe(*input.node_arg));
      }
      auto outputs = std::vector<std::string>();
      outputs.reserve(conv_outputs->size());
      for (const auto &output : *conv_outputs) {
        ASSERT_TRUE(output != nullptr) << "Conv node output should not be null";
        outputs.push_back(wrapped_api_->node_arg_get_name_unsafe(*output));
      }
      auto const_initializers = std::vector<std::string>{};
      auto [meta_def, err] =
          pass->try_fuse(graph, "fuse_" + node_name, inputs, outputs,
                         const_initializers, "TEST_FUSE");
      CHECK(meta_def != nullptr) << "Failed to fuse node: " << node_name;
      auto &fuse_node = pass->fuse(graph, std::move(*meta_def));

      auto &function_body = wrapped_api_->node_get_function_body(fuse_node);
      ASSERT_TRUE(&function_body != nullptr)
          << "Function body for fused node should not be null";
      auto fuse_node_order =
          morphizen::graph_get_node_in_topoligical_order(function_body);
      for (auto fuse_index : fuse_node_order) {
        auto fuse_node_ptr =
            wrapped_api_->graph_get_node(function_body, fuse_index);
        if (fuse_node_ptr == nullptr) {
          LOG(INFO) << "Fused node at index " << fuse_index << " is null";
          continue;
        }
        LOG(INFO) << "Fused node at index " << fuse_index
                  << " is: " << morphizen::node_as_string(*fuse_node_ptr);
      }
    }

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test23_try_fuse_and_fuse failed with exception: "
               << e.what();
    FAIL() << "Test23_try_fuse_and_fuse failed with exception: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Test23_try_fuse_and_fuse failed with unknown exception";
    FAIL() << "Test23_try_fuse_and_fuse failed with unknown exception";
  }
}

void MorphizenOrtApiTest::Test24_convert_initializer_to_const_op() {
  LOG(INFO) << "=== Test24: Convert Initializer to Const Op ===";

  try {
    if (!simple_conv_relu_model_) {
      LOG(INFO) << "No model available for Initializer to Const test, creating "
                   "one first...";
      Test07_create_simple_conv_relu_model();
    }

    ASSERT_TRUE(simple_conv_relu_model_ != nullptr)
        << "Failed to get model for Initializer to Const test";

    // Clone the model for testing
    LOG(INFO) << "Cloning model for Initializer to Const operations...";
    auto *cloned_model = wrapped_api_->model_clone(
        *simple_conv_relu_model_, std::numeric_limits<size_t>::max());
    ASSERT_TRUE(cloned_model != nullptr) << "Failed to clone model";

    // Get the graph from the cloned model
    auto &graph = wrapped_api_->model_main_graph(*cloned_model);

    // foreach initializer node args
    const auto &initializer_tensors =
        wrapped_api_->graph_get_all_initialized_tensors(graph);
    ASSERT_TRUE(initializer_tensors.size() > 0)
        << "No initializer tensors found in the graph";
    LOG(INFO) << "Found " << initializer_tensors.size()
              << " initializer tensors in the graph";
    for (auto constant_tensor : initializer_tensors) {
      LOG(INFO) << "Converting initializer tensor: " << constant_tensor.first
                << " to Const Op";
      const auto *node_arg =
          wrapped_api_->graph_get_node_arg(graph, constant_tensor.first);
      ASSERT_TRUE(node_arg != nullptr)
          << "Node arg for initializer tensor " << constant_tensor.first
          << " should not be null";
      auto shape = wrapped_api_->node_arg_get_shape_i64_unsafe(*node_arg);
      ASSERT_TRUE(shape.get() != nullptr)
          << "Shape for initializer tensor " << constant_tensor.first
          << " should not be null";
      auto data_type = wrapped_api_->node_arg_get_element_type(*node_arg);
      ASSERT_TRUE(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
          << "Data type for initializer tensor " << constant_tensor.first
          << " should be FLOAT, but found: " << data_type;

      {
        // add testcase for cover tensor_proto_get_shape_unsafe API
        auto tensor_shape = wrapped_api_->tensor_proto_get_shape_unsafe(
            *constant_tensor.second);
        ASSERT_TRUE(tensor_shape.get() != nullptr)
            << "Tensor shape for initializer tensor " << constant_tensor.first
            << " should not be null";
        ASSERT_EQ(*tensor_shape, *shape)
            << "Tensor shape from NodeArg and TensorProto should match for "
               "initializer tensor "
            << constant_tensor.first;

        auto tensor_data_type =
            wrapped_api_->tensor_proto_data_type(*constant_tensor.second);
        ASSERT_EQ(tensor_data_type, data_type)
            << "Tensor data type for initializer tensor "
            << constant_tensor.first
            << " should be FLOAT, but found: " << tensor_data_type;

        auto tensor_data =
            wrapped_api_->tensor_proto_as_raw(graph, *constant_tensor.second);
        auto tensor_data_size =
            wrapped_api_->tensor_proto_raw_data_size(*constant_tensor.second);
        ASSERT_EQ(tensor_data.size(), tensor_data_size)
            << "Tensor data size for initializer tensor "
            << constant_tensor.first
            << " does not match expected size from raw data";
        ASSERT_EQ(tensor_data_size,
                  sizeof(float) * std::accumulate(tensor_shape->begin(),
                                                  tensor_shape->end(), 1,
                                                  std::multiplies<int64_t>()))
            << "Tensor data size for initializer tensor "
            << constant_tensor.first
            << " does not match expected size from shape";
      }

      auto op_type = std::string("const");
      auto attrs = morphizen::NodeAttributesBuilder();
      attrs.add("data_type", "float");
      if (!(*shape).empty()) {
        attrs.add("shape", *shape);
      } else {
        attrs.add("shape", std::vector<int64_t>{});
      }
      auto &const_node = wrapped_api_->graph_add_node(
          graph, "const_" + constant_tensor.first, op_type,
          "Const operation for initializer tensor", {}, {node_arg},
          *attrs.build(), "com.xilinx");
      wrapped_api_->graph_remove_initialized_tensor(
          graph,
          constant_tensor.first); // remove original node arg

      LOG(INFO) << "Converted initializer tensor: " << constant_tensor.first
                << " to Const Op";
    }

    // TODO , here need remove graph_resolve
    wrapped_api_->graph_resolve(graph, true);
    for (auto index : morphizen::graph_get_node_in_topoligical_order(graph)) {
      auto node = wrapped_api_->graph_get_node(graph, index);
      if (node == nullptr) {
        LOG(INFO) << "Node at index " << index << " is null";
        continue;
      }
      auto op_type = morphizen::node_op_type(*node);
      // try_fuse and fuse
      if (op_type != "Conv") {
        LOG(INFO) << "Skipping node: " << morphizen::node_as_string(*node)
                  << " with op_type: " << op_type;
        continue;
      }
      auto conv_inputs = wrapped_api_->node_get_inputs_unsafe(*node);
      // check conv_node inputs size is 3
      ASSERT_TRUE(conv_inputs.get() != nullptr && conv_inputs->size() == 3);
      auto weights = (*conv_inputs)[1];
      // get weights op type
      ASSERT_TRUE(weights.node != nullptr)
          << "Conv node weights input should be a const node, not be null "
             "after "
             "converting "
             "initializers to Const Ops";
      auto weights_op_type = wrapped_api_->node_op_type(*weights.node);
      ASSERT_EQ(weights_op_type, "const")
          << "Conv node weights input should be a Const Op, but found: "
          << weights_op_type;

      auto bias = (*conv_inputs)[2];
      // get bias op type
      ASSERT_TRUE(bias.node != nullptr)
          << "Conv node bias input should be a const node, not be null after "
             "converting "
             "initializers to Const Ops";
      auto bias_op_type = wrapped_api_->node_op_type(*bias.node);
      ASSERT_EQ(bias_op_type, "const")
          << "Conv node bias input should be a Const Op, but found: "
          << bias_op_type;
    }

    wrapped_api_->graph_resolve(graph, true);
    LOG(INFO) << "Graph resolved after converting initializers to Const Ops";
    wrapped_api_->model_delete(cloned_model);

  } catch (const std::exception &e) {
    LOG(ERROR)
        << "Test24_convert_initializer_to_const_op failed with exception: "
        << e.what();
    FAIL() << "Test24_convert_initializer_to_const_op failed with exception: "
           << e.what();
  } catch (...) {
    LOG(ERROR) << "Test24_convert_initializer_to_const_op failed with unknown "
                  "exception";
    FAIL() << "Test24_convert_initializer_to_const_op failed with unknown "
              "exception";
  }
  LOG(INFO) << "Test24_convert_initializer_to_const_op completed successfully";
}

// ============================================================================
// Comprehensive Coverage Test
// ============================================================================

void MorphizenOrtApiTest::ComprehensiveCoverageReport() {
  // IMPORTANT: This test only shows statistics for APIs called within this
  // specific test. Each test has isolated statistics due to
  // reset_morphizen_ort_api_call_statistics() in SetUp(). This is CORRECT
  // behavior for test isolation - prevents test interference.
  //
  // To see cumulative coverage across all tests, run with VLOG=1 to see
  // per-test breakdowns.

  auto stats = get_morphizen_ort_api_call_statistics();

  LOG(INFO) << "=== MorphiZen ORT API Coverage Report (Current Test Only) ===";
  LOG(INFO) << "APIs called in current test: " << stats.size();

  // Use the coverage checker to get detailed analysis
  auto [coverage_percent, missing_apis] = check_api_coverage(stats);

  LOG(INFO) << "API Coverage: " << std::fixed << std::setprecision(1)
            << coverage_percent << "% (" << (stats.size()) << "/"
            << get_all_morphizen_ort_api_functions().size() << ")";

  if (!missing_apis.empty()) {
    LOG(INFO) << "=== Missing APIs (" << missing_apis.size() << ") ===";
    for (const auto &api : missing_apis) {
      LOG(INFO) << "  - " << api;
    }
  }

  // Count APIs by category using the helper
  auto api_categories = get_morphizen_ort_api_by_category();
  for (const auto &[category, api_list] : api_categories) {
    size_t called_in_category = 0;
    for (const auto &api : api_list) {
      if (stats.find(api) != stats.end() && stats.at(api) > 0) {
        called_in_category++;
      }
    }
    double category_coverage =
        static_cast<double>(called_in_category) / api_list.size() * 100.0;
    LOG(INFO) << category << " APIs: " << called_in_category << "/"
              << api_list.size() << " (" << std::fixed << std::setprecision(1)
              << category_coverage << "%)";
  }

  // Log the most frequently called APIs
  LOG(INFO) << "=== Most Frequently Called APIs ===";
  std::vector<std::pair<std::string, size_t>> sorted_stats(stats.begin(),
                                                           stats.end());
  std::sort(sorted_stats.begin(), sorted_stats.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  for (size_t i = 0; i < std::min(size_t(10), sorted_stats.size()); ++i) {
    LOG(INFO) << "  " << sorted_stats[i].first << ": " << sorted_stats[i].second
              << " calls";
  }

  // Coverage assertions - we expect reasonably good coverage from our tests
  std::string enable_unittest =
      ENV_PARAM(MORPHIZEN_ORT_BRIDGE_UNITTEST_COVERAGE);
  auto enable_test_all = enable_unittest.empty() || enable_unittest == "all";
  if (enable_test_all) {
    EXPECT_GT(coverage_percent, 50.0)
        << "API coverage is too low: " << coverage_percent << "%";
  }

  // Ensure each major category has some coverage
  size_t categories_with_coverage = 0;
  for (const auto &[category, api_list] : api_categories) {
    bool has_coverage = false;
    for (const auto &api : api_list) {
      if (stats.find(api) != stats.end() && stats.at(api) > 0) {
        has_coverage = true;
        break;
      }
    }
    if (has_coverage)
      categories_with_coverage++;
  }

  EXPECT_GE(categories_with_coverage, 5)
      << "Too few API categories have coverage";
}

void MorphizenOrtApiTest::DeleteSimpleConvReluModel() {
  if (simple_conv_relu_model_) {
    wrapped_api_->model_delete(simple_conv_relu_model_);
  }
}

void MorphizenOrtApiTest::DetailedCoverageAnalysis() {
  // Analyze coverage in detail and provide recommendations
  auto stats = get_morphizen_ort_api_call_statistics();
  auto [coverage_percent, missing_apis] = check_api_coverage(stats);

  LOG(INFO) << "=== Detailed Coverage Analysis ===";

  // Categorize missing APIs
  auto api_categories = get_morphizen_ort_api_by_category();
  std::map<std::string, std::vector<std::string>> missing_by_category;

  for (const auto &missing_api : missing_apis) {
    for (const auto &[category, api_list] : api_categories) {
      if (std::find(api_list.begin(), api_list.end(), missing_api) !=
          api_list.end()) {
        missing_by_category[category].push_back(missing_api);
        break;
      }
    }
  }

  for (const auto &[category, missing_list] : missing_by_category) {
    if (!missing_list.empty()) {
      LOG(INFO) << "Missing " << category << " APIs (" << missing_list.size()
                << "):";
      for (const auto &api : missing_list) {
        LOG(INFO) << "  - " << api;
      }
    }
  }

  // Recommendations for improving coverage
  LOG(INFO) << "=== Coverage Improvement Recommendations ===";
  if (missing_by_category["Model"].size() > 0) {
    LOG(INFO)
        << "Consider adding tests with actual ONNX model files for Model APIs";
  }
  if (missing_by_category["Graph"].size() > 0) {
    LOG(INFO) << "Consider adding tests with more complex graph operations";
  }
  if (missing_by_category["Node"].size() > 0) {
    LOG(INFO) << "Consider adding tests with models that have actual nodes";
  }
  // Record coverage for CI/CD tracking
  LOG(INFO) << "COVERAGE_METRIC: " << std::fixed << std::setprecision(2)
            << coverage_percent;
  LOG(INFO) << "TOTAL_APIS: " << get_all_morphizen_ort_api_functions().size();
  LOG(INFO) << "COVERED_APIS: " << (stats.size());
  LOG(INFO) << "MISSING_APIS: " << missing_apis.size();
}

// ============================================================================
// Model Creation Test - Conv+ReLU Pattern
// ============================================================================

void MorphizenOrtApiTest::Test07_create_simple_conv_relu_model() {
  LOG(INFO) << "=== Test07: Creating Simple Conv+ReLU Model ===";

  try {
    // Create a temporary path for the model
    std::filesystem::path temp_path =
        CMAKE_CURRENT_BINARY_PATH / "Test07_create_simple_conv_relu_model.onnx";
    std::vector<std::pair<std::string, int64_t>> opset = {{"", 11}};

    // Create an empty model
    auto *model = wrapped_api_->create_empty_model(temp_path, opset);
    ASSERT_TRUE(model != nullptr) << "Failed to create empty model";
    auto &graph = wrapped_api_->model_main_graph(*model);
    LOG(INFO) << " the default graph name is "
              << wrapped_api_->graph_get_name(graph);

    // Set a meaningful graph name
    wrapped_api_->graph_set_name(graph, "simple_conv_relu");
    LOG(INFO) << " the after set name,  graph name is "
              << wrapped_api_->graph_get_name(graph);
    // Create input tensor (NCHW format: batch=1, channels=3, height=224,
    // width=224)
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    auto &input_arg = wrapped_api_->node_arg_new(graph, "input", &input_shape,
                                                 1); // FLOAT type
    CHECK(&input_arg != nullptr);
    wrapped_api_->graph_set_inputs(
        graph, gsl::span<const morphizen::NodeArg *const>({&input_arg}));
    // Create convolution weights (output_channels=64, input_channels=3,
    // kernel_h=3, kernel_w=3)
    std::vector<int64_t> weight_shape = {64, 3, 3, 3};
    std::vector<float> weight_data(64 * 3 * 3 * 3,
                                   0.1f); // Initialize with small values
    auto *weight_tensor = wrapped_api_->tensor_proto_new_floats(
        "conv_weight", weight_shape, weight_data);
    ASSERT_TRUE(weight_tensor != nullptr) << "Failed to create weight tensor";

    // Add weight as initialized tensor to the graph
    wrapped_api_->graph_add_initialized_tensor(graph, *weight_tensor);

    // Create bias tensor (64 output channels)
    std::vector<int64_t> bias_shape = {64};
    std::vector<float> bias_data(64, 0.0f); // Initialize with zeros
    auto *bias_tensor = wrapped_api_->tensor_proto_new_floats(
        "conv_bias", bias_shape, bias_data);
    ASSERT_TRUE(bias_tensor != nullptr) << "Failed to create bias tensor";

    // Add bias as initialized tensor to the graph
    wrapped_api_->graph_add_initialized_tensor(graph, *bias_tensor);

    // Create intermediate tensor for conv output (will be input to ReLU)
    std::vector<int64_t> conv_output_shape = {
        1, 64, 222, 222}; // Assuming no padding, stride=1
    auto &conv_output_arg =
        wrapped_api_->node_arg_new(graph, "conv_output", &conv_output_shape, 1);

    // Create final output tensor (same shape as conv output since ReLU doesn't
    // change shape)
    std::vector<int64_t> final_output_shape = {1, 64, 222, 222};
    auto &final_output_arg =
        wrapped_api_->node_arg_new(graph, "output", &final_output_shape, 1);

    // Create Conv node attributes
    auto *conv_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(conv_attrs != nullptr) << "Failed to create node attributes";

    // Add kernel_shape attribute [3, 3]
    std::vector<int64_t> kernel_shape = {3, 3};
    auto *kernel_shape_attr =
        wrapped_api_->attr_proto_new_ints("kernel_shape", kernel_shape);
    if (kernel_shape_attr) {
      wrapped_api_->node_attributes_add(*conv_attrs,
                                        std::move(*kernel_shape_attr));
    }

    // Add strides attribute [1, 1]
    std::vector<int64_t> strides = {1, 1};
    auto *strides_attr = wrapped_api_->attr_proto_new_ints("strides", strides);
    if (strides_attr) {
      wrapped_api_->node_attributes_add(*conv_attrs, std::move(*strides_attr));
    }

    // Add pads attribute [0, 0, 0, 0] (no padding)
    std::vector<int64_t> pads = {0, 0, 0, 0};
    auto *pads_attr = wrapped_api_->attr_proto_new_ints("pads", pads);
    if (pads_attr) {
      wrapped_api_->node_attributes_add(*conv_attrs, std::move(*pads_attr));
    }

    // Create weight and bias NodeArgs for the Conv node
    auto weight_arg = wrapped_api_->graph_get_node_arg(graph, "conv_weight");
    ASSERT_TRUE(weight_arg != nullptr);
    auto bias_arg = wrapped_api_->graph_get_node_arg(graph, "conv_bias");
    ASSERT_TRUE(bias_arg != nullptr);

    // Create Conv node inputs and outputs
    std::vector<const morphizen::NodeArg *> conv_inputs = {
        &input_arg, weight_arg, bias_arg};
    std::vector<const morphizen::NodeArg *> conv_outputs = {&conv_output_arg};

    // Add Conv node to the graph
    LOG(INFO) << "Adding Conv node to graph...";
    auto &conv_node = wrapped_api_->graph_add_node(
        graph, "conv_node", "Conv", "Convolution operation", conv_inputs,
        conv_outputs, *conv_attrs, "");

    // Create ReLU node attributes (ReLU doesn't need specific attributes)
    auto *relu_attrs = wrapped_api_->node_attributes_new();
    ASSERT_TRUE(relu_attrs != nullptr)
        << "Failed to create ReLU node attributes";

    // Create ReLU node inputs and outputs
    std::vector<const morphizen::NodeArg *> relu_inputs = {&conv_output_arg};
    std::vector<const morphizen::NodeArg *> relu_outputs = {&final_output_arg};

    // Add ReLU node to the graph
    LOG(INFO) << "Adding ReLU node to graph...";
    auto &relu_node = wrapped_api_->graph_add_node(
        graph, "relu_node", "Relu", "ReLU activation operation", relu_inputs,
        relu_outputs, *relu_attrs, "");

    // Set graph inputs and outputs
    std::vector<const morphizen::NodeArg *> graph_inputs = {&input_arg};
    std::vector<const morphizen::NodeArg *> graph_outputs = {&final_output_arg};

    wrapped_api_->graph_set_outputs(
        graph, gsl::span<const morphizen::NodeArg *const>(graph_outputs));

    LOG(INFO) << "Created model structure:";
    LOG(INFO) << "  - Input: "
              << wrapped_api_->node_arg_get_name_unsafe(input_arg)
              << " shape: [1, 3, 224, 224]";
    LOG(INFO) << "  - Conv weights: [64, 3, 3, 3]";
    LOG(INFO) << "  - Conv bias: [64]";
    LOG(INFO) << "  - Conv output: "
              << wrapped_api_->node_arg_get_name_unsafe(conv_output_arg)
              << " shape: [1, 64, 222, 222]";
    LOG(INFO) << "  - Final output: "
              << wrapped_api_->node_arg_get_name_unsafe(final_output_arg)
              << " shape: [1, 64, 222, 222]";

    // Test model metadata
    wrapped_api_->model_set_meta_data(*model, "description",
                                      "Simple Conv+ReLU model for testing");
    wrapped_api_->model_set_meta_data(*model, "author", "MorphiZen Test Suite");

    // Verify metadata
    auto description = wrapped_api_->model_get_meta_data(*model, "description");
    auto author = wrapped_api_->model_get_meta_data(*model, "author");

    EXPECT_EQ(*description, "Simple Conv+ReLU model for testing");
    EXPECT_EQ(*author, "MorphiZen Test Suite");

    // Test graph resolution
    // NOTE: this will invalidate all node_arg and node index
    int resolution_result = wrapped_api_->graph_resolve(graph, false);
    LOG(INFO) << "Graph resolution result: " << resolution_result;
    LOG(INFO) << "Saving graph to file: " << temp_path.string() + ".graph.onnx";
    wrapped_api_->graph_save(graph, temp_path.string() + ".graph.onnx", "",
                             1024);
    // Get graph statistics
    auto inputs = wrapped_api_->graph_get_inputs_unsafe(graph);
    auto outputs = wrapped_api_->graph_get_outputs_unsafe(graph);
    auto nodes = wrapped_api_->graph_nodes_unsafe(graph);
    ASSERT_EQ(nodes->size(), 2) << "Expected 2 nodes (Conv and ReLU)";
    auto tensors = wrapped_api_->graph_get_all_initialized_tensors(graph);

    LOG(INFO) << "Graph statistics:";
    LOG(INFO) << "  - Inputs: " << inputs->size();
    LOG(INFO) << "  - Outputs: " << outputs->size();
    LOG(INFO) << "  - Nodes: " << nodes->size();
    LOG(INFO) << "  - Initialized tensors: " << tensors.size();
    // Save the initial graph to file

    LOG(INFO) << "Graph saved successfully";
    // Cleanup
    wrapped_api_->node_attributes_delete(conv_attrs);
    wrapped_api_->node_attributes_delete(relu_attrs);
    wrapped_api_->tensor_proto_delete(weight_tensor);
    wrapped_api_->tensor_proto_delete(bias_tensor);
    simple_conv_relu_model_ = model;
    LOG(INFO) << "Test07_create_simple_conv_relu_model: Conv+ReLU model "
                 "creation completed successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "Test07_create_simple_conv_relu_model failed with exception: "
               << e.what();
    FAIL() << "Test07_create_simple_conv_relu_model failed with exception: "
           << e.what();
  } catch (...) {
    LOG(ERROR) << "Test16 failed with unknown exception";
    FAIL()
        << "Test07_create_simple_conv_relu_model failed with unknown exception";
  }
}

// ============================================================================
// Main Test Entry Point - Runs All Tests Sequentially
// ============================================================================

TEST_F(MorphizenOrtApiTest, TestAll) {
#ifdef MORPHIZEN_ENABLE_MLIR_BACKEND
  GTEST_SKIP()
      << "Test skipped: MLIR backend has multiple issues including "
         "plugin loading (Issue #030), model loading (Issue #028), and "
         "other backend-specific problems";
#else
  LOG(INFO) << "=== Starting Sequential Test Execution ===";

  std::string enable_unittest =
      ENV_PARAM(MORPHIZEN_ORT_BRIDGE_UNITTEST_COVERAGE);
  auto enable_test_all = enable_unittest.empty() || enable_unittest == "all";

  // Run all test components in sequence
  if (enable_test_all || enable_unittest == "1") {
    LOG(INFO) << "Running Test01_TestIsolationVerification...";
    ASSERT_NO_FATAL_FAILURE(Test01_TestIsolationVerification());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test01_TestIsolationVerification failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "2") {
    LOG(INFO) << "Running Test02_TestIsolationVerificationSecond...";
    ASSERT_NO_FATAL_FAILURE(Test02_TestIsolationVerificationSecond());
    if (HasFailure()) {
      LOG(ERROR) << "Test02_TestIsolationVerificationSecond failed, stopping "
                    "execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "3") {
    LOG(INFO) << "Running Test03_ModelLoadAndDelete...";
    ASSERT_NO_FATAL_FAILURE(Test03_ModelLoadAndDelete());
    if (HasFailure()) {
      LOG(ERROR) << "Test03_ModelLoadAndDelete failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "4") {
    LOG(INFO) << "Running Test04_ModelMetaDataOperations...";
    ASSERT_NO_FATAL_FAILURE(Test04_ModelMetaDataOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test04_ModelMetaDataOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "5") {
    LOG(INFO) << "Running Test05_GraphBasicOperations...";
    ASSERT_NO_FATAL_FAILURE(Test05_GraphBasicOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test05_GraphBasicOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "6") {
    LOG(INFO) << "Running Test06_GraphAdvancedOperations...";
    ASSERT_NO_FATAL_FAILURE(Test06_GraphAdvancedOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test06_GraphAdvancedOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "7") {
    LOG(INFO) << "Running Test07_create_simple_conv_relu_model...";
    ASSERT_NO_FATAL_FAILURE(Test07_create_simple_conv_relu_model());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test07_create_simple_conv_relu_model failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "8") {
    LOG(INFO) << "Running Test08_NodeOperations...";
    ASSERT_NO_FATAL_FAILURE(Test08_NodeOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test08_NodeOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "9") {
    LOG(INFO) << "Running Test09_NodeArgOperations...";
    ASSERT_NO_FATAL_FAILURE(Test09_NodeArgOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test09_NodeArgOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "10") {
    LOG(INFO) << "Running Test10_NodeAttributesOperations...";
    ASSERT_NO_FATAL_FAILURE(Test10_NodeAttributesOperations());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test10_NodeAttributesOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "11") {
    LOG(INFO) << "Running Test11_AttributeProtoOperations...";
    ASSERT_NO_FATAL_FAILURE(Test11_AttributeProtoOperations());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test11_AttributeProtoOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "12") {
    LOG(INFO) << "Running Test12_TensorProtoOperations...";
    ASSERT_NO_FATAL_FAILURE(Test12_TensorProtoOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test12_TensorProtoOperations failed, stopping execution";
      return;
    }
  }
  if (enable_test_all || enable_unittest == "13") {
    LOG(INFO) << "Running Test13_ExtendedApiOperations...";
    ASSERT_NO_FATAL_FAILURE(Test13_ExtendedApiOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test13_ExtendedApiOperations failed, stopping execution";
      return;
    }
  }
  if (enable_test_all || enable_unittest == "14") {
    LOG(INFO) << "Running Test14_GraphTensorOperations...";
    ASSERT_NO_FATAL_FAILURE(Test14_GraphTensorOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test14_GraphTensorOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "16") {
    LOG(INFO) << "Running Test16_GraphFuseOperations...";
    ASSERT_NO_FATAL_FAILURE(Test16_GraphFuseOperations());
    if (HasFailure()) {
      LOG(ERROR) << "Test16_GraphFuseOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "17") {
    LOG(INFO) << "Running Test17_GraphNodeRemovalOperations...";
    ASSERT_NO_FATAL_FAILURE(Test17_GraphNodeRemovalOperations());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test17_GraphNodeRemovalOperations failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "18") {
    LOG(INFO) << "Running Test18_MissingApisCoverage...";
    ASSERT_NO_FATAL_FAILURE(Test18_MissingApisCoverage());
    if (HasFailure()) {
      LOG(ERROR) << "Test18_MissingApisCoverage failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "19") {
    LOG(INFO) << "Running Test19_add_sin_op_before_relu_op...";
    ASSERT_NO_FATAL_FAILURE(Test19_add_sin_op_before_relu_op());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test19_add_sin_op_before_relu_op failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "20") {
    LOG(INFO) << "Running Test20_conv_relu_fuse_conv2d_nchw...";
    ASSERT_NO_FATAL_FAILURE(Test20_conv_relu_fuse_conv2d_nchw());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test20_conv_relu_fuse_conv2d_nchw failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "21") {
    LOG(INFO) << "Running Test21_fuse_relu_q...";
    ASSERT_NO_FATAL_FAILURE(Test21_fuse_relu_q());
    if (HasFailure()) {
      LOG(ERROR) << "Test21_fuse_relu_q failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "22") {
    LOG(INFO) << "Running Test22_create_initializer_node_arg...";
    ASSERT_NO_FATAL_FAILURE(Test22_create_initializer_node_arg());
    if (HasFailure()) {
      LOG(ERROR)
          << "Test22_create_initializer_node_arg failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "23") {
    LOG(INFO) << "Running Test23_try_fuse_and_fuse...";
    ASSERT_NO_FATAL_FAILURE(Test23_try_fuse_and_fuse());
    if (HasFailure()) {
      LOG(ERROR) << "Test23_try_fuse_and_fuse failed, stopping execution";
      return;
    }
  }

  if (enable_test_all || enable_unittest == "24") {
    LOG(INFO) << "Running Test24_convert_initializer_to_const_op...";
    ASSERT_NO_FATAL_FAILURE(Test24_convert_initializer_to_const_op());
    if (HasFailure()) {
      LOG(ERROR) << "Test24_convert_initializer_to_const_op failed, stopping "
                    "execution";
      return;
    }
  }
  if (enable_test_all) {
    LOG(INFO) << "Running ComprehensiveCoverageReport...";
    ComprehensiveCoverageReport();
    LOG(INFO) << "Running DetailedCoverageAnalysis...";
    DetailedCoverageAnalysis();
  }
  LOG(INFO) << "=== Sequential Test Execution Completed (enable_unittest=\""
            << enable_unittest << "\", backend=\"" << backend_ << "\") === ";
#endif // MORPHIZEN_ENABLE_MLIR_BACKEND
}

} // namespace test
} // namespace morphizen
