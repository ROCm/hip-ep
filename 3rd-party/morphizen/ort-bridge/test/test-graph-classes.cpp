/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../src/model.hpp"
#include "../src/ort-graph-wrapper.hpp"
#include <gtest/gtest.h>
#include <type_traits>

namespace morphizen {

class GraphClassesTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(GraphClassesTest, BothClassesCanBeInstantiated) {
  // Test that both Graph and OrtGraphWrapper classes exist and can be
  // referenced

  // OrtGraphWrapper is a concrete class that can be used with ORT APIs
  // Note: We can't actually instantiate it without proper ORT setup
  // This test just verifies the class definitions are correct

  // Verify OrtGraphWrapper class exists
  static_assert(std::is_class_v<OrtGraphWrapper>,
                "OrtGraphWrapper should be a class");

  // Verify Graph class exists
  static_assert(std::is_class_v<Graph>, "Graph should be a class");

  // Verify they are different types
  static_assert(!std::is_same_v<Graph, OrtGraphWrapper>,
                "Graph and OrtGraphWrapper should be different classes");

  SUCCEED();
}

TEST_F(GraphClassesTest, GraphClassPlaceholderExists) {
  // Create a simple ONNX graph for testing
  ONNX_NAMESPACE::GraphProto graph_proto;
  graph_proto.set_name("test_graph");

  // Add a simple node: input -> Add -> output
  auto* node = graph_proto.add_node();
  node->set_name("add_node");
  node->set_op_type("Add");
  node->add_input("input1");
  node->add_input("input2");
  node->add_output("output1");

  // Add graph inputs
  auto* input1 = graph_proto.add_input();
  input1->set_name("input1");
  auto* input2 = graph_proto.add_input();
  input2->set_name("input2");

  // Add graph output
  auto* output = graph_proto.add_output();
  output->set_name("output1");

  // Create Graph instance
  Graph graph(graph_proto);

  // Test basic properties
  EXPECT_EQ(graph.name(), "test_graph");
  EXPECT_EQ(graph.node_count(), 1);
  EXPECT_EQ(graph.input_count(), 2);
  EXPECT_EQ(graph.output_count(), 1);
  EXPECT_EQ(graph.initializer_count(), 0);
}

TEST_F(GraphClassesTest, GraphNodeLookup) {
  // Create a graph with multiple nodes
  ONNX_NAMESPACE::GraphProto graph_proto;
  graph_proto.set_name("multi_node_graph");

  // Node 1: input1 -> Relu -> relu_out
  auto* node1 = graph_proto.add_node();
  node1->set_name("relu_node");
  node1->set_op_type("Relu");
  node1->add_input("input1");
  node1->add_output("relu_out");

  // Node 2: relu_out + input2 -> Add -> output1
  auto* node2 = graph_proto.add_node();
  node2->set_name("add_node");
  node2->set_op_type("Add");
  node2->add_input("relu_out");
  node2->add_input("input2");
  node2->add_output("output1");

  // Add inputs and outputs
  auto* input1 = graph_proto.add_input();
  input1->set_name("input1");
  auto* input2 = graph_proto.add_input();
  input2->set_name("input2");
  auto* output = graph_proto.add_output();
  output->set_name("output1");

  Graph graph(graph_proto);

  // Test node lookup
  const auto* relu_node = graph.find_node("relu_node");
  ASSERT_NE(relu_node, nullptr);
  EXPECT_EQ(relu_node->name(), "relu_node");
  EXPECT_EQ(relu_node->op_type(), "Relu");

  const auto* add_node = graph.find_node("add_node");
  ASSERT_NE(add_node, nullptr);
  EXPECT_EQ(add_node->name(), "add_node");
  EXPECT_EQ(add_node->op_type(), "Add");

  // Test non-existent node
  const auto* missing_node = graph.find_node("missing_node");
  EXPECT_EQ(missing_node, nullptr);

  // Test getting all nodes
  auto all_nodes = graph.nodes();
  EXPECT_EQ(all_nodes.size(), 2);
}

TEST_F(GraphClassesTest, GraphDependencyAnalysis) {
  // Create a graph with dependencies: input1 -> Relu -> Add <- input2 ->
  // output1
  ONNX_NAMESPACE::GraphProto graph_proto;
  graph_proto.set_name("dependency_graph");

  // Node 1: input1 -> Relu -> relu_out
  auto* node1 = graph_proto.add_node();
  node1->set_name("relu_node");
  node1->set_op_type("Relu");
  node1->add_input("input1");
  node1->add_output("relu_out");

  // Node 2: relu_out + input2 -> Add -> output1
  auto* node2 = graph_proto.add_node();
  node2->set_name("add_node");
  node2->set_op_type("Add");
  node2->add_input("relu_out");
  node2->add_input("input2");
  node2->add_output("output1");

  // Add inputs and outputs
  auto* input1 = graph_proto.add_input();
  input1->set_name("input1");
  auto* input2 = graph_proto.add_input();
  input2->set_name("input2");
  auto* output = graph_proto.add_output();
  output->set_name("output1");

  Graph graph(graph_proto);

  // Test dependencies
  auto add_deps = graph.get_dependencies("add_node");
  EXPECT_EQ(add_deps.size(), 1);
  EXPECT_EQ(add_deps[0]->name(), "relu_node");

  auto relu_deps = graph.get_dependencies("relu_node");
  EXPECT_EQ(relu_deps.size(), 0); // No dependencies, uses graph input

  // Test dependents
  auto relu_dependents = graph.get_dependents("relu_node");
  EXPECT_EQ(relu_dependents.size(), 1);
  EXPECT_EQ(relu_dependents[0]->name(), "add_node");

  auto add_dependents = graph.get_dependents("add_node");
  EXPECT_EQ(add_dependents.size(), 0); // No dependents, produces graph output

  // Test value tracking
  EXPECT_EQ(graph.find_producer("relu_out"), "relu_node");
  EXPECT_EQ(graph.find_producer("output1"), "add_node");
  EXPECT_EQ(graph.find_producer("input1"), ""); // Graph input

  auto relu_out_consumers = graph.find_consumers("relu_out");
  EXPECT_EQ(relu_out_consumers.size(), 1);
  EXPECT_EQ(relu_out_consumers[0], "add_node");

  // Test graph input/output detection
  EXPECT_TRUE(graph.is_graph_input("input1"));
  EXPECT_TRUE(graph.is_graph_input("input2"));
  EXPECT_FALSE(graph.is_graph_input("relu_out"));
  EXPECT_TRUE(graph.is_graph_output("output1"));
  EXPECT_FALSE(graph.is_graph_output("relu_out"));
}

TEST_F(GraphClassesTest, GraphTopologicalSort) {
  // Create a graph: input1 -> Relu -> Add <- input2 -> output1
  ONNX_NAMESPACE::GraphProto graph_proto;
  graph_proto.set_name("topo_graph");

  // Node 1: input1 -> Relu -> relu_out
  auto* node1 = graph_proto.add_node();
  node1->set_name("relu_node");
  node1->set_op_type("Relu");
  node1->add_input("input1");
  node1->add_output("relu_out");

  // Node 2: relu_out + input2 -> Add -> output1
  auto* node2 = graph_proto.add_node();
  node2->set_name("add_node");
  node2->set_op_type("Add");
  node2->add_input("relu_out");
  node2->add_input("input2");
  node2->add_output("output1");

  // Add inputs and outputs
  auto* input1 = graph_proto.add_input();
  input1->set_name("input1");
  auto* input2 = graph_proto.add_input();
  input2->set_name("input2");
  auto* output = graph_proto.add_output();
  output->set_name("output1");

  Graph graph(graph_proto);

  // Test topological sort
  auto sorted_nodes = graph.topological_sort();
  EXPECT_EQ(sorted_nodes.size(), 2);

  // Relu should come before Add
  EXPECT_EQ(sorted_nodes[0]->name(), "relu_node");
  EXPECT_EQ(sorted_nodes[1]->name(), "add_node");

  // Test cycle detection
  EXPECT_FALSE(graph.has_cycles());
}

TEST_F(GraphClassesTest, GraphWithInitializers) {
  // Create a graph with initializers
  ONNX_NAMESPACE::GraphProto graph_proto;
  graph_proto.set_name("init_graph");

  // Add an initializer
  auto* init = graph_proto.add_initializer();
  init->set_name("weights");
  init->add_dims(3);
  init->add_dims(3);

  // Node: input1 + weights -> Add -> output1
  auto* node = graph_proto.add_node();
  node->set_name("add_node");
  node->set_op_type("Add");
  node->add_input("input1");
  node->add_input("weights");
  node->add_output("output1");

  // Add inputs and outputs
  auto* input1 = graph_proto.add_input();
  input1->set_name("input1");
  auto* output = graph_proto.add_output();
  output->set_name("output1");

  Graph graph(graph_proto);

  // Test initializer detection
  EXPECT_TRUE(graph.is_initializer("weights"));
  EXPECT_FALSE(graph.is_initializer("input1"));
  EXPECT_EQ(graph.initializer_count(), 1);

  // Test that initializers don't create node dependencies
  auto deps = graph.get_dependencies("add_node");
  EXPECT_EQ(deps.size(),
            0); // No node dependencies, just graph input + initializer

  // Test producer lookup for initializer
  EXPECT_EQ(graph.find_producer("weights"),
            ""); // Empty string for initializers
}

// === Model Class Tests ===

class ModelClassTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper method to create a simple ONNX ModelProto for testing
  ONNX_NAMESPACE::ModelProto create_simple_model() {
    ONNX_NAMESPACE::ModelProto model_proto;

    // Set model information
    model_proto.set_ir_version(8);
    model_proto.set_producer_name("test_producer");
    model_proto.set_producer_version("1.0.0");
    model_proto.set_domain("test.domain");
    model_proto.set_model_version(1);
    model_proto.set_doc_string("Test model documentation");

    // Add opset imports
    auto* opset = model_proto.add_opset_import();
    opset->set_domain("ai.onnx");
    opset->set_version(17);

    auto* custom_opset = model_proto.add_opset_import();
    custom_opset->set_domain("custom.domain");
    custom_opset->set_version(2);

    // Add metadata properties
    auto* prop1 = model_proto.add_metadata_props();
    prop1->set_key("author");
    prop1->set_value("test_author");

    auto* prop2 = model_proto.add_metadata_props();
    prop2->set_key("description");
    prop2->set_value("A test model");

    // Create a simple graph
    auto* graph = model_proto.mutable_graph();
    graph->set_name("test_graph");

    // Add a simple node: input -> Relu -> output
    auto* node = graph->add_node();
    node->set_name("relu_node");
    node->set_op_type("Relu");
    node->add_input("input1");
    node->add_output("output1");

    // Add graph inputs and outputs
    auto* input = graph->add_input();
    input->set_name("input1");

    auto* output = graph->add_output();
    output->set_name("output1");

    return model_proto;
  }
};

TEST_F(ModelClassTest, BasicModelInfo) {
  auto model_proto = create_simple_model();
  Model model = Model::create_model(std::move(model_proto));

  // Test basic model information
  EXPECT_EQ(model.ir_version(), 8);
  EXPECT_EQ(model.producer_name(), "test_producer");
  EXPECT_EQ(model.producer_version(), "1.0.0");
  EXPECT_EQ(model.domain(), "test.domain");
  EXPECT_EQ(model.model_version(), 1);
  EXPECT_EQ(model.doc_string(), "Test model documentation");

  // Test model proto access
  EXPECT_EQ(model.model_proto().ir_version(), 8);
  EXPECT_EQ(model.model_proto().producer_name(), "test_producer");
}

TEST_F(ModelClassTest, GraphAccess) {
  auto model_proto = create_simple_model();
  Model model = Model::create_model(std::move(model_proto));

  // Test graph access
  EXPECT_TRUE(model.has_graph());

  const auto* graph_proto = model.graph();
  ASSERT_NE(graph_proto, nullptr);
  EXPECT_EQ(graph_proto->name(), "test_graph");
  EXPECT_EQ(graph_proto->node_size(), 1);
  EXPECT_EQ(graph_proto->input_size(), 1);
  EXPECT_EQ(graph_proto->output_size(), 1);

  // Test graph count
  EXPECT_EQ(model.graph_count(), 1);

  // Test graphs vector
  const auto& graphs = model.graphs();
  EXPECT_EQ(graphs.size(), 1);
  EXPECT_EQ(graphs[0], graph_proto);
}

TEST_F(ModelClassTest, OpsetImports) {
  auto model_proto = create_simple_model();
  Model model = Model::create_model(std::move(model_proto));

  // Test opset imports
  auto opset_imports = model.get_opset_imports();
  EXPECT_EQ(opset_imports.size(), 2);
  EXPECT_EQ(opset_imports["ai.onnx"], 17);
  EXPECT_EQ(opset_imports["custom.domain"], 2);

  // Test specific opset version lookup
  EXPECT_EQ(model.get_opset_version("ai.onnx"), 17);
  EXPECT_EQ(model.get_opset_version("custom.domain"), 2);
  EXPECT_EQ(model.get_opset_version("nonexistent.domain"), -1);

  // Test opset import existence
  EXPECT_TRUE(model.has_opset_import("ai.onnx"));
  EXPECT_TRUE(model.has_opset_import("custom.domain"));
  EXPECT_FALSE(model.has_opset_import("nonexistent.domain"));
}

TEST_F(ModelClassTest, MetadataProperties) {
  auto model_proto = create_simple_model();
  Model model = Model::create_model(std::move(model_proto));

  // Test metadata properties
  auto metadata_props = model.get_metadata_props();
  EXPECT_EQ(metadata_props.size(), 2);
  EXPECT_EQ(metadata_props["author"], "test_author");
  EXPECT_EQ(metadata_props["description"], "A test model");

  // Test specific property lookup
  EXPECT_EQ(model.get_metadata_prop("author"), "test_author");
  EXPECT_EQ(model.get_metadata_prop("description"), "A test model");
  EXPECT_EQ(model.get_metadata_prop("nonexistent"), "");

  // Test property existence
  EXPECT_TRUE(model.has_metadata_prop("author"));
  EXPECT_TRUE(model.has_metadata_prop("description"));
  EXPECT_FALSE(model.has_metadata_prop("nonexistent"));
}

TEST_F(ModelClassTest, Functions) {
  auto model_proto = create_simple_model();
  // Add a function to the model
  auto* func = model_proto.add_functions();
  func->set_name("test_function");
  func->set_domain("test.domain");

  Model model = Model::create_model(std::move(model_proto));

  // Test function access
  EXPECT_EQ(model.function_count(), 1);

  auto function_names = model.get_function_names();
  EXPECT_EQ(function_names.size(), 1);
  EXPECT_EQ(function_names[0], "test_function");

  // Test function lookup
  const auto* found_func = model.find_function("test_function");
  ASSERT_NE(found_func, nullptr);
  EXPECT_EQ(found_func->name(), "test_function");
  EXPECT_EQ(found_func->domain(), "test.domain");

  // Test non-existent function
  const auto* missing_func = model.find_function("missing_function");
  EXPECT_EQ(missing_func, nullptr);
}

TEST_F(ModelClassTest, ModelValidation) {
  auto model_proto = create_simple_model();
  Model model = Model::create_model(std::move(model_proto));

  // Test valid model
  EXPECT_TRUE(model.is_valid());

  auto errors = model.get_validation_errors();
  EXPECT_EQ(errors.size(), 0);
}

TEST_F(ModelClassTest, ModelValidationErrors) { // Create an invalid model
  ONNX_NAMESPACE::ModelProto invalid_model;

  // No IR version, no graph, no opsets, no producer
  Model model = Model::create_model(std::move(invalid_model));

  // Test invalid model
  EXPECT_FALSE(model.is_valid());

  auto errors = model.get_validation_errors();
  EXPECT_GT(errors.size(), 0);

  // Check that specific errors are detected
  bool has_ir_version_error = false;
  bool has_no_graph_error = false;
  bool has_no_opset_error = false;
  bool has_no_producer_error = false;

  for (const auto& error : errors) {
    if (error.find("IR version") != std::string::npos) {
      has_ir_version_error = true;
    }
    if (error.find("no graph") != std::string::npos) {
      has_no_graph_error = true;
    }
    if (error.find("no opset") != std::string::npos) {
      has_no_opset_error = true;
    }
    if (error.find("Producer name is empty") != std::string::npos) {
      has_no_producer_error = true;
    }
  }

  EXPECT_TRUE(has_ir_version_error);
  EXPECT_TRUE(has_no_graph_error);
  EXPECT_TRUE(has_no_opset_error);
  EXPECT_TRUE(has_no_producer_error);
}

TEST_F(ModelClassTest, MoveConstructorAndAssignment) {
  auto model_proto = create_simple_model();
  Model original = Model::create_model(std::move(model_proto));

  // Store original values for comparison
  auto original_ir_version = original.ir_version();
  auto original_producer_name = original.producer_name();
  auto original_opset_version = original.get_opset_version("ai.onnx");
  auto original_metadata_prop = original.get_metadata_prop("author");

  // Test move constructor
  Model moved(std::move(original));
  EXPECT_EQ(moved.ir_version(), original_ir_version);
  EXPECT_EQ(moved.producer_name(), original_producer_name);
  EXPECT_EQ(moved.get_opset_version("ai.onnx"), original_opset_version);
  EXPECT_EQ(moved.get_metadata_prop("author"), original_metadata_prop);

  // Test move assignment operator
  auto another_model_proto = create_simple_model();
  Model target = Model::create_model(std::move(another_model_proto));
  target = std::move(moved);
  EXPECT_EQ(target.ir_version(), original_ir_version);
  EXPECT_EQ(target.producer_name(), original_producer_name);
  EXPECT_EQ(target.get_opset_version("ai.onnx"), original_opset_version);
  EXPECT_EQ(target.get_metadata_prop("author"), original_metadata_prop);
}

TEST_F(ModelClassTest, NonCopyable) {
  // Test that Model is non-copyable
  static_assert(!std::is_copy_constructible_v<Model>,
                "Model should not be copy constructible");
  static_assert(!std::is_copy_assignable_v<Model>,
                "Model should not be copy assignable");

  // Test that Model is movable
  static_assert(std::is_move_constructible_v<Model>,
                "Model should be move constructible");
  static_assert(std::is_move_assignable_v<Model>,
                "Model should be move assignable");
}

TEST_F(ModelClassTest, MultipleGraphsWithFunctions) {
  auto model_proto = create_simple_model();

  // Add a function with a graph body
  auto* func = model_proto.add_functions();
  func->set_name("test_function");
  func->set_domain("test.domain");

  // Create a graph body for the function
  auto* func_graph = func->mutable_body();
  func_graph->set_name("function_graph");

  // Add a simple node to the function graph
  auto* func_node = func_graph->add_node();
  func_node->set_name("func_node");
  func_node->set_op_type("Add");
  func_node->add_input("func_input1");
  func_node->add_input("func_input2");
  func_node->add_output("func_output");

  Model model = Model::create_model(std::move(model_proto));

  // Test that we have 2 graphs total (main + function)
  EXPECT_EQ(model.graph_count(), 2);

  const auto& graphs = model.graphs();
  EXPECT_EQ(graphs.size(), 2);

  // First graph should be the main graph
  EXPECT_EQ(graphs[0]->name(), "test_graph");
  EXPECT_EQ(graphs[0]->node_size(), 1);

  // Second graph should be the function graph
  EXPECT_EQ(graphs[1]->name(), "function_graph");
  EXPECT_EQ(graphs[1]->node_size(), 1);
}

} // namespace morphizen
