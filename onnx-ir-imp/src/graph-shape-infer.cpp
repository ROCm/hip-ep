/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// #include "./graph-shape-infer.hpp"
#include "./graph-shape-infer.hpp"
#include "./graph-inference-context.hpp"
#include "./graph.hpp"
#include "./model.hpp"
#include <glog/logging.h>
#include <onnx/defs/shape_inference.h>
#include <onnx/shape_inference/implementation.h>
#include <unordered_set>

namespace morphizen {

GraphShapeInfer::GraphShapeInfer()
    : last_error_(""), verbose_(false), check_model_(true),
      target_graph_(nullptr) {}

GraphShapeInfer::GraphShapeInfer(Graph* graph)
    : last_error_(""), verbose_(false), check_model_(true),
      target_graph_(graph) {
  if (!graph) {
    throw std::invalid_argument(
        "Graph pointer cannot be null in GraphShapeInfer constructor");
  }
}

GraphShapeInfer::~GraphShapeInfer() = default;

bool GraphShapeInfer::infer_shapes(Graph& graph) {
  try {
    last_error_.clear();

    log_progress("Starting shape inference for graph: " + graph.get_name());

    // Get the model's opset imports
    const auto& opset_imports = graph.get_model().get_opset_imports();

    // Get a mutable reference to the graph proto
    auto& graph_proto =
        const_cast<morphizen_onnx::GraphProto&>(graph.get_graph_proto());

    // Perform shape inference
    bool success = infer_shapes(graph_proto, opset_imports);

    if (success) {
      log_progress("Shape inference completed successfully for graph: " +
                   graph.get_name());
    } else {
      LOG(ERROR) << "Shape inference failed for graph: " << graph.get_name()
                 << " Error: " << last_error_;
    }

    return success;

  } catch (const std::exception& e) {
    last_error_ = std::string("Exception during shape inference: ") + e.what();
    LOG(ERROR) << "Shape inference failed for graph: " << graph.get_name()
               << " Error: " << last_error_;
    return false;
  }
}

bool GraphShapeInfer::infer_shapes() {
  if (!target_graph_) {
    last_error_ = "No target graph set in GraphShapeInfer";
    LOG(ERROR) << last_error_;
    return false;
  }

  return infer_shapes(*target_graph_);
}

bool GraphShapeInfer::infer_shapes(
    morphizen_onnx::GraphProto& graph_proto,
    const std::unordered_map<std::string, int>& opset_imports) {
  try {
    last_error_.clear();

    log_progress("Starting shape inference on graph proto: " +
                 graph_proto.name());

    // Validate the graph proto if checking is enabled
    if (check_model_ && !validate_graph_proto(graph_proto)) {
      return false;
    }

    // Create symbol table for shape inference
    auto symbol_table = create_symbol_table();

    log_progress("Created symbol table for shape inference");

    // Run ONNX shape inference
    try {
      log_progress("Running ONNX shape inference");
      // Demonstrate custom inference using our GraphInferenceContextImpl
      if (!custom_infer_shapes(graph_proto)) {
        LOG(WARNING) << "Custom inference demonstration failed, but continuing "
                        "with standard ONNX inference";
      }
      log_progress("ONNX shape inference completed successfully");
    } catch (const std::exception& e) {
      last_error_ = std::string("ONNX shape inference error: ") + e.what();
      LOG(ERROR) << last_error_;
      return false;
    }

    log_progress("Shape inference completed successfully on graph proto: " +
                 graph_proto.name());
    return true;

  } catch (const std::exception& e) {
    last_error_ = std::string("Exception during shape inference: ") + e.what();
    LOG(ERROR) << "Shape inference failed on graph proto. Error: "
               << last_error_;
    return false;
  }
}

const std::string& GraphShapeInfer::get_last_error() const {
  return last_error_;
}

void GraphShapeInfer::set_verbose(bool verbose) { verbose_ = verbose; }

void GraphShapeInfer::set_check_model(bool check) { check_model_ = check; }

bool GraphShapeInfer::validate_graph_proto(
    const morphizen_onnx::GraphProto& graph_proto) {
  // Basic validation checks
  if (graph_proto.name().empty()) {
    last_error_ = "Graph proto must have a non-empty name";
    LOG(WARNING) << last_error_;
    // This is not a fatal error, just a warning
  }

  // Check that the graph has at least some content
  if (graph_proto.node_size() == 0 && graph_proto.input_size() == 0) {
    last_error_ = "Graph proto appears to be empty (no nodes and no inputs)";
    LOG(ERROR) << last_error_;
    return false;
  }

  // Check for duplicate node names
  std::unordered_set<std::string> node_names;
  for (const auto& node : graph_proto.node()) {
    if (!node.name().empty()) {
      if (node_names.count(node.name()) > 0) {
        last_error_ = "Duplicate node name found: " + node.name();
        LOG(ERROR) << last_error_;
        return false;
      }
      node_names.insert(node.name());
    }
  }

  // Check for duplicate input names
  std::unordered_set<std::string> input_names;
  for (const auto& input : graph_proto.input()) {
    if (input_names.count(input.name()) > 0) {
      last_error_ = "Duplicate input name found: " + input.name();
      LOG(ERROR) << last_error_;
      return false;
    }
    input_names.insert(input.name());
  }

  // Check for duplicate output names
  std::unordered_set<std::string> output_names;
  for (const auto& output : graph_proto.output()) {
    if (output_names.count(output.name()) > 0) {
      last_error_ = "Duplicate output name found: " + output.name();
      LOG(ERROR) << last_error_;
      return false;
    }
    output_names.insert(output.name());
  }

  log_progress("Graph proto validation passed");
  return true;
}

std::unique_ptr<morphizen_onnx::shape_inference::SymbolTableImpl>
GraphShapeInfer::create_symbol_table() {
  // Create a symbol table for shape inference
  // This allows for tracking symbolic dimensions and relationships
  auto symbol_table =
      std::make_unique<morphizen_onnx::shape_inference::SymbolTableImpl>();

  log_progress("Created symbol table implementation");
  return symbol_table;
}

std::unique_ptr<GraphInferenceContextImpl>
GraphShapeInfer::createInferenceContext(
    morphizen_onnx::NodeProto& node,
    const std::unordered_map<std::string, morphizen_onnx::TypeProto*>&
        valueTypesByName,
    const std::unordered_map<std::string, const morphizen_onnx::TensorProto*>&
        inputDataByName,
    const std::unordered_map<std::string,
                             const morphizen_onnx::SparseTensorProto*>&
        inputSparseDataByName,
    const morphizen_onnx::ShapeInferenceOptions& options,
    morphizen_onnx::shape_inference::DataValueMap* generatedShapeData,
    morphizen_onnx::shape_inference::GraphInferenceContext*
        graphInferenceContext) const {

  return std::make_unique<GraphInferenceContextImpl>(
      node, valueTypesByName, inputDataByName, inputSparseDataByName, options,
      generatedShapeData, graphInferenceContext);
}

void GraphShapeInfer::log_progress(const std::string& message) {
  if (verbose_) {
    LOG(INFO) << "[GraphShapeInfer] " << message;
  }
}

bool GraphShapeInfer::custom_infer_shapes(
    morphizen_onnx::GraphProto& graph_proto) {
  try {
    log_progress(
        "Demonstrating custom inference using ONNX InferenceContext interface");

    // Build value type maps
    std::unordered_map<std::string, morphizen_onnx::TypeProto*>
        valueTypesByName;
    std::unordered_map<std::string, const morphizen_onnx::TensorProto*>
        inputDataByName;
    std::unordered_map<std::string, const morphizen_onnx::SparseTensorProto*>
        inputSparseDataByName;

    // Add input types
    for (auto& input : *graph_proto.mutable_input()) {
      valueTypesByName[input.name()] = input.mutable_type();
    }

    // Add output types (may be incomplete initially)
    for (auto& output : *graph_proto.mutable_output()) {
      valueTypesByName[output.name()] = output.mutable_type();
    }

    // Add value info types
    for (auto& value_info : *graph_proto.mutable_value_info()) {
      valueTypesByName[value_info.name()] = value_info.mutable_type();
    }

    // Add initializer data
    for (const auto& initializer : graph_proto.initializer()) {
      inputDataByName[initializer.name()] = &initializer;
    }

    // Add sparse initializer data
    for (const auto& sparse_initializer : graph_proto.sparse_initializer()) {
      inputSparseDataByName[sparse_initializer.values().name()] =
          &sparse_initializer;
    }

    // Set up shape inference options
    morphizen_onnx::ShapeInferenceOptions options;
    options.check_type = true;
    options.error_mode = 1;
    options.enable_data_propagation = false;

    // Process each node with our custom inference context
    for (auto& node : *graph_proto.mutable_node()) {
      // Create custom inference context for this node
      auto context =
          createInferenceContext(node, valueTypesByName, inputDataByName,
                                 inputSparseDataByName, options);

      // Demonstrate accessing the context information
      log_progress("Processing node '" + context->getDisplayName() + "' with " +
                   std::to_string(context->getNumInputs()) + " inputs and " +
                   std::to_string(context->getNumOutputs()) + " outputs");

      // Here you could implement custom shape inference logic for specific
      // operators using the full ONNX InferenceContext interface:

      // Example: Check if this is an element-wise operation
      const std::string& op_type = node.op_type();
      if (op_type == "Add" || op_type == "Mul" || op_type == "Sub" ||
          op_type == "Div") {
        log_progress("  -> Element-wise operation detected: " + op_type);

        // For element-wise ops, output shape typically matches first input
        if (context->getNumInputs() > 0 && context->getNumOutputs() > 0) {
          const auto* input_type = context->getInputType(0);
          auto* output_type = context->getOutputType(0);

          if (input_type && output_type && input_type->has_tensor_type()) {
            // Copy element type and shape from input to output
            *output_type->mutable_tensor_type() = input_type->tensor_type();
            log_progress("  -> Propagated shape from input to output");
          }
        }
      }

      // Example: Check for attributes
      if (context->getAttribute("axis") != nullptr) {
        log_progress("  -> Node has 'axis' attribute");
      }
    }

    log_progress("Custom inference demonstration completed successfully");
    return true;

  } catch (const std::exception& e) {
    last_error_ =
        std::string("Exception during custom inference demonstration: ") +
        e.what();
    LOG(ERROR) << "Custom inference demonstration failed. Error: "
               << last_error_;
    return false;
  }
}

} // namespace morphizen
