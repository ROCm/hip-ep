<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Model Class Usage Examples

This document provides comprehensive examples of how to use the `Model` class for in-memory ONNX ModelProto operations.

## Overview

The `Model` class provides efficient operations on ONNX model structure, including:
- Access to model metadata (IR version, producer info, domain, version)
- Opset import information and version checking
- Metadata properties access
- Function definitions access
- Direct access to GraphProto objects (main graph and function graphs)
- Model validation

## Basic Usage

### Creating a Model Instance

```cpp
#include "model.hpp"
#include <onnx/onnx_pb.h>

// Load or create an ONNX ModelProto
ONNX_NAMESPACE::ModelProto model_proto;
// ... load from file or create programmatically ...

// Create Model instance using the factory method (recommended)
morphizen::Model model = morphizen::Model::create_model(std::move(model_proto));

// Note: Model is non-copyable but movable
// auto copy = model;  // ❌ This will not compile
// auto moved = std::move(model);  // ✅ This works
```

### Accessing Model Information

```cpp
// Basic model information
std::cout << "IR Version: " << model.ir_version() << std::endl;
std::cout << "Producer: " << model.producer_name()
          << " v" << model.producer_version() << std::endl;
std::cout << "Domain: " << model.domain() << std::endl;
std::cout << "Model Version: " << model.model_version() << std::endl;
std::cout << "Documentation: " << model.doc_string() << std::endl;
```

## Move Semantics and Ownership

The `Model` class uses move semantics for efficient memory management and is designed to be non-copyable but movable.

### Why Move Semantics?

```cpp
// ✅ Efficient - transfers ownership without copying large ModelProto
ONNX_NAMESPACE::ModelProto model_proto = load_model_from_file("model.onnx");
morphizen::Model model(std::move(model_proto));

// ❌ This would require copying the entire ModelProto (not supported)
// morphizen::Model model_copy(model);  // Compilation error

// ✅ Moving is allowed and efficient
morphizen::Model moved_model = std::move(model);
```

### Working with Move Semantics

```cpp
// Function that returns a Model (using factory method)
std::unique_ptr<morphizen::Model> create_model() {
    ONNX_NAMESPACE::ModelProto model_proto;
    // ... setup model_proto ...

    auto model = morphizen::Model::create_model(std::move(model_proto));
    return std::make_unique<morphizen::Model>(std::move(model));
}

// Using the returned Model
auto model_ptr = create_model();
// model_ptr->ir_version(), etc.

// Moving Models in containers
std::vector<morphizen::Model> models;
ONNX_NAMESPACE::ModelProto proto1, proto2;
// ... setup protos ...

models.emplace_back(morphizen::Model::create_model(std::move(proto1)));  // ✅ Construct in-place
models.push_back(morphizen::Model::create_model(std::move(proto2)));  // ✅ Move construct
```

### Benefits of Move-Only Design

- **Memory Efficiency**: Avoids expensive copying of large ModelProto objects
- **Clear Ownership**: Makes ownership transfer explicit
- **Performance**: Move operations are typically much faster than copies
- **Resource Safety**: Prevents accidental expensive copies

## Opset Import Management

### Checking Opset Versions

```cpp
// Get all opset imports
auto opsets = model.get_opset_imports();
for (const auto& [domain, version] : opsets) {
    std::cout << "Domain: " << domain << ", Version: " << version << std::endl;
}

// Check specific opset version
int64_t onnx_version = model.get_opset_version("ai.onnx");
if (onnx_version != -1) {
    std::cout << "ONNX opset version: " << onnx_version << std::endl;
} else {
    std::cout << "ONNX opset not found" << std::endl;
}

// Check if domain exists
if (model.has_opset_import("ai.onnx.ml")) {
    std::cout << "Model uses ONNX ML operators" << std::endl;
}
```

### Working with Custom Opsets

```cpp
// Check for custom domain opsets
if (model.has_opset_import("com.microsoft")) {
    int64_t ms_version = model.get_opset_version("com.microsoft");
    std::cout << "Microsoft opset version: " << ms_version << std::endl;
}

// Iterate through all domains
auto opsets = model.get_opset_imports();
for (const auto& [domain, version] : opsets) {
    if (domain != "ai.onnx" && !domain.empty()) {
        std::cout << "Custom domain: " << domain
                  << " (version " << version << ")" << std::endl;
    }
}
```

## Metadata Properties

### Accessing Model Metadata

```cpp
// Get all metadata properties
auto metadata = model.get_metadata_props();
for (const auto& [key, value] : metadata) {
    std::cout << key << ": " << value << std::endl;
}

// Get specific properties
std::string author = model.get_metadata_prop("author");
std::string description = model.get_metadata_prop("description");
std::string license = model.get_metadata_prop("license");

if (!author.empty()) {
    std::cout << "Model author: " << author << std::endl;
}
```

### Checking for Standard Metadata

```cpp
// Common metadata properties to check
std::vector<std::string> common_props = {
    "author", "description", "license", "version",
    "source", "converted_from", "accuracy"
};

for (const auto& prop : common_props) {
    if (model.has_metadata_prop(prop)) {
        std::cout << prop << ": " << model.get_metadata_prop(prop) << std::endl;
    }
}
```

## Function Definitions

### Working with Model Functions

```cpp
// Check if model has functions
if (model.function_count() > 0) {
    std::cout << "Model has " << model.function_count() << " functions:" << std::endl;

    // Get all function names
    auto function_names = model.get_function_names();
    for (const auto& name : function_names) {
        std::cout << "  - " << name << std::endl;
    }
}

// Find specific function
const auto* custom_op = model.find_function("CustomOp");
if (custom_op != nullptr) {
    std::cout << "Found CustomOp function in domain: "
              << custom_op->domain() << std::endl;

    // Access function details
    std::cout << "Function inputs: " << custom_op->input_size() << std::endl;
    std::cout << "Function outputs: " << custom_op->output_size() << std::endl;
    std::cout << "Function nodes: " << custom_op->node_size() << std::endl;
}
```

## Graph Access

### Working with the Main Graph

```cpp
if (model.has_graph()) {
    // Get direct access to the main GraphProto
    const auto* graph_proto = model.graph();

    std::cout << "Graph: " << graph_proto->name() << std::endl;
    std::cout << "Nodes: " << graph_proto->node_size() << std::endl;
    std::cout << "Inputs: " << graph_proto->input_size() << std::endl;
    std::cout << "Outputs: " << graph_proto->output_size() << std::endl;
    std::cout << "Initializers: " << graph_proto->initializer_size() << std::endl;

    // Iterate through nodes
    for (int i = 0; i < graph_proto->node_size(); ++i) {
        const auto& node = graph_proto->node(i);
        std::cout << "Node: " << node.name()
                  << " (" << node.op_type() << ")" << std::endl;
    }
}
```

### Working with All Graphs (Including Function Graphs)

```cpp
// Get all graphs in the model (main + function graphs)
const auto& all_graphs = model.graphs();
std::cout << "Total graphs in model: " << model.graph_count() << std::endl;

for (size_t i = 0; i < all_graphs.size(); ++i) {
    const auto* graph = all_graphs[i];
    std::cout << "Graph " << i << ": " << graph->name()
              << " (" << graph->node_size() << " nodes)" << std::endl;
}

// Mutable access for modifications
if (auto* mutable_graph = model.mutable_graph()) {
    // Can modify the main graph
    mutable_graph->set_name("modified_graph_name");
}
```

## Model Validation

### Basic Validation

```cpp
// Quick validation check
if (model.is_valid()) {
    std::cout << "Model appears to be valid" << std::endl;
} else {
    std::cout << "Model has validation issues" << std::endl;

    // Get detailed error list
    auto errors = model.get_validation_errors();
    for (const auto& error : errors) {
        std::cout << "ERROR: " << error << std::endl;
    }
}
```

### Comprehensive Model Analysis

```cpp
void analyze_model(const morphizen::Model& model) {
    std::cout << "=== Model Analysis ===" << std::endl;

    // Basic info
    std::cout << "Producer: " << model.producer_name()
              << " v" << model.producer_version() << std::endl;
    std::cout << "IR Version: " << model.ir_version() << std::endl;

    // Opsets
    std::cout << "\nOpset Imports:" << std::endl;
    auto opsets = model.get_opset_imports();
    for (const auto& [domain, version] : opsets) {
        std::cout << "  " << (domain.empty() ? "ai.onnx" : domain)
                  << ": v" << version << std::endl;
    }
      // Graph structure
    if (model.has_graph()) {
        const auto* graph = model.graph();
        std::cout << "\nGraph Structure:" << std::endl;
        std::cout << "  Name: " << graph->name() << std::endl;
        std::cout << "  Nodes: " << graph->node_size() << std::endl;
        std::cout << "  Inputs: " << graph->input_size() << std::endl;
        std::cout << "  Outputs: " << graph->output_size() << std::endl;
        std::cout << "  Initializers: " << graph->initializer_size() << std::endl;
    }

    // All graphs (including function graphs)
    std::cout << "\nTotal graphs: " << model.graph_count() << std::endl;

    // Functions
    if (model.function_count() > 0) {
        std::cout << "\nFunctions: " << model.function_count() << std::endl;
        auto function_names = model.get_function_names();
        for (const auto& name : function_names) {
            std::cout << "  - " << name << std::endl;
        }
    }

    // Validation
    std::cout << "\nValidation: ";
    if (model.is_valid()) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        auto errors = model.get_validation_errors();
        for (const auto& error : errors) {
            std::cout << "  ERROR: " << error << std::endl;
        }
    }

    std::cout << "=====================" << std::endl;
}
```

## Advanced Usage Patterns

### Model Comparison

```cpp
bool compare_models(const morphizen::Model& model1, const morphizen::Model& model2) {
    // Compare basic properties
    if (model1.ir_version() != model2.ir_version()) {
        std::cout << "Different IR versions" << std::endl;
        return false;
    }

    // Compare opsets
    auto opsets1 = model1.get_opset_imports();
    auto opsets2 = model2.get_opset_imports();
    if (opsets1 != opsets2) {
        std::cout << "Different opset imports" << std::endl;
        return false;
    }

    // Compare graph structure
    if (model1.has_graph() != model2.has_graph()) {
        return false;
    }

    if (model1.has_graph()) {
        const auto& g1 = model1.graph();
        const auto& g2 = model2.graph();

        if (g1.node_count() != g2.node_count() ||
            g1.input_count() != g2.input_count() ||
            g1.output_count() != g2.output_count()) {
            std::cout << "Different graph structure" << std::endl;
            return false;
        }
    }

    return true;
}
```

### Model Compatibility Checking

```cpp
bool is_model_compatible(const morphizen::Model& model, int64_t min_ir_version) {
    // Check IR version
    if (model.ir_version() < min_ir_version) {
        std::cout << "IR version too old: " << model.ir_version()
                  << " < " << min_ir_version << std::endl;
        return false;
    }

    // Check for required opsets
    if (!model.has_opset_import("ai.onnx")) {
        std::cout << "Missing required ONNX opset" << std::endl;
        return false;
    }

    int64_t onnx_version = model.get_opset_version("ai.onnx");
    if (onnx_version < 11) {  // Minimum supported version
        std::cout << "ONNX opset version too old: " << onnx_version << std::endl;
        return false;
    }

    // Check model validity
    if (!model.is_valid()) {
        std::cout << "Model failed validation" << std::endl;
        return false;
    }

    return true;
}
```

## Error Handling

### Robust Model Loading

```cpp
#include <exception>
#include <iostream>

morphizen::Model load_model_safely(const ONNX_NAMESPACE::ModelProto& model_proto) {
    try {
        morphizen::Model model(model_proto);

        // Validate the model
        if (!model.is_valid()) {
            auto errors = model.get_validation_errors();
            std::cerr << "Model validation failed:" << std::endl;
            for (const auto& error : errors) {
                std::cerr << "  - " << error << std::endl;
            }
            throw std::runtime_error("Invalid model");
        }

        return model;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        throw;
    }
}
```

## Integration with Graph Operations

### Combined Model and Graph Analysis

```cpp
void analyze_model_execution_flow(const morphizen::Model& model) {
    if (!model.has_graph()) {
        std::cout << "Model has no graph" << std::endl;
        return;
    }

    const auto& graph = model.graph();

    std::cout << "=== Execution Flow Analysis ===" << std::endl;
    std::cout << "Model: " << model.producer_name() << std::endl;
    std::cout << "ONNX Version: " << model.get_opset_version("ai.onnx") << std::endl;

    // Get execution order
    auto sorted_nodes = graph.topological_sort();

    std::cout << "\nExecution Order:" << std::endl;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        const auto* node = sorted_nodes[i];
        std::cout << (i + 1) << ". " << node->name()
                  << " (" << node->op_type() << ")" << std::endl;

        // Show inputs
        std::cout << "   Inputs: ";
        for (int j = 0; j < node->input_size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << node->input(j);

            if (graph.is_graph_input(node->input(j))) {
                std::cout << " [graph input]";
            } else if (graph.is_initializer(node->input(j))) {
                std::cout << " [initializer]";
            }
        }
        std::cout << std::endl;

        // Show outputs
        std::cout << "   Outputs: ";
        for (int j = 0; j < node->output_size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << node->output(j);

            if (graph.is_graph_output(node->output(j))) {
                std::cout << " [graph output]";
            }
        }
        std::cout << std::endl;
    }
}
```

This comprehensive documentation shows how to effectively use the `Model` class for various ONNX model operations. The class provides efficient access to model metadata, opset information, functions, and the underlying graph structure while maintaining performance through caching and smart data access patterns.
