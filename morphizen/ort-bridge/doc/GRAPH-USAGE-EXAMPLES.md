<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Graph Class Usage Examples

This document demonstrates how to use the new `Graph` class for efficient in-memory operations on ONNX GraphProto structures.

## Basic Usage

```cpp
#include "ort-bridge/src/graph.hpp"
#include <onnx/onnx_pb.h>

// Load or create an ONNX GraphProto
ONNX_NAMESPACE::GraphProto graph_proto;
// ... populate graph_proto ...

// Create Graph instance for efficient operations
morphizen::Graph graph(graph_proto);

// Basic information
std::cout << "Graph name: " << graph.name() << std::endl;
std::cout << "Nodes: " << graph.node_count() << std::endl;
std::cout << "Inputs: " << graph.input_count() << std::endl;
std::cout << "Outputs: " << graph.output_count() << std::endl;
```

## Node Operations

```cpp
// Find a specific node
const auto* node = graph.find_node("my_conv_node");
if (node) {
    std::cout << "Found node: " << node->name() << " (type: " << node->op_type() << ")" << std::endl;
}

// Get all nodes
auto all_nodes = graph.nodes();
for (const auto* node : all_nodes) {
    std::cout << "Node: " << node->name() << " -> " << node->op_type() << std::endl;
}

// Get node by index
try {
    const auto& first_node = graph.node(0);
    std::cout << "First node: " << first_node.name() << std::endl;
} catch (const std::out_of_range& e) {
    std::cout << "No nodes in graph" << std::endl;
}
```

## Dependency Analysis

```cpp
// Get direct dependencies of a node
auto deps = graph.get_dependencies("my_add_node");
std::cout << "Direct dependencies of my_add_node:" << std::endl;
for (const auto* dep : deps) {
    std::cout << "  - " << dep->name() << std::endl;
}

// Get direct dependents of a node
auto dependents = graph.get_dependents("my_conv_node");
std::cout << "Direct dependents of my_conv_node:" << std::endl;
for (const auto* dep : dependents) {
    std::cout << "  - " << dep->name() << std::endl;
}

// Get transitive dependencies (all nodes this node depends on)
auto transitive_deps = graph.get_transitive_dependencies("output_node");
std::cout << "All dependencies of output_node:" << std::endl;
for (const std::string& dep_name : transitive_deps) {
    std::cout << "  - " << dep_name << std::endl;
}
```

## Topological Operations

```cpp
// Check for cycles
if (graph.has_cycles()) {
    std::cout << "Graph has cycles - cannot execute sequentially!" << std::endl;
} else {
    std::cout << "Graph is acyclic - safe for sequential execution" << std::endl;

    // Get nodes in topological order
    try {
        auto sorted_nodes = graph.topological_sort();
        std::cout << "Execution order:" << std::endl;
        for (size_t i = 0; i < sorted_nodes.size(); ++i) {
            std::cout << i+1 << ". " << sorted_nodes[i]->name() << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
```

## Value Tracking

```cpp
// Find who produces a value
std::string producer = graph.find_producer("intermediate_tensor");
if (!producer.empty()) {
    std::cout << "intermediate_tensor is produced by: " << producer << std::endl;
} else {
    std::cout << "intermediate_tensor is a graph input or initializer" << std::endl;
}

// Find who consumes a value
auto consumers = graph.find_consumers("input_tensor");
std::cout << "input_tensor is consumed by:" << std::endl;
for (const std::string& consumer : consumers) {
    std::cout << "  - " << consumer << std::endl;
}

// Check value types
if (graph.is_graph_input("data")) {
    std::cout << "data is a graph input" << std::endl;
}
if (graph.is_graph_output("result")) {
    std::cout << "result is a graph output" << std::endl;
}
if (graph.is_initializer("weights")) {
    std::cout << "weights is an initializer" << std::endl;
}

// Get node inputs and outputs
auto node_inputs = graph.get_node_inputs("my_conv_node");
auto node_outputs = graph.get_node_outputs("my_conv_node");
std::cout << "my_conv_node inputs: ";
for (const auto& input : node_inputs) {
    std::cout << input << " ";
}
std::cout << std::endl;
```

## Graph Analysis Example

```cpp
void analyze_graph(const morphizen::Graph& graph) {
    std::cout << "=== Graph Analysis ===" << std::endl;
    std::cout << "Name: " << graph.name() << std::endl;
    std::cout << "Nodes: " << graph.node_count() << std::endl;
    std::cout << "Inputs: " << graph.input_count() << std::endl;
    std::cout << "Outputs: " << graph.output_count() << std::endl;
    std::cout << "Initializers: " << graph.initializer_count() << std::endl;

    // Check for cycles
    if (graph.has_cycles()) {
        std::cout << "WARNING: Graph contains cycles!" << std::endl;
        return;
    }

    // Show execution order
    auto sorted_nodes = graph.topological_sort();
    std::cout << "\\nExecution order:" << std::endl;
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        const auto* node = sorted_nodes[i];
        std::cout << i+1 << ". " << node->name() << " (" << node->op_type() << ")" << std::endl;

        // Show dependencies
        auto deps = graph.get_dependencies(node->name());
        if (!deps.empty()) {
            std::cout << "   Dependencies: ";
            for (const auto* dep : deps) {
                std::cout << dep->name() << " ";
            }
            std::cout << std::endl;
        }
    }

    // Show graph inputs and outputs
    std::cout << "\\nGraph inputs:" << std::endl;
    auto inputs = graph.inputs();
    for (const auto* input : inputs) {
        std::cout << "  - " << input->name() << std::endl;
    }

    std::cout << "\\nGraph outputs:" << std::endl;
    auto outputs = graph.outputs();
    for (const auto* output : outputs) {
        std::cout << "  - " << output->name() << std::endl;
    }

    if (graph.initializer_count() > 0) {
        std::cout << "\\nInitializers:" << std::endl;
        auto initializers = graph.initializers();
        for (const auto* init : initializers) {
            std::cout << "  - " << init->name() << std::endl;
        }
    }
}
```

## Performance Considerations

The `Graph` class builds dependency maps during construction, so:

1. **Construction Cost**: O(V + E) where V = nodes, E = edges (value dependencies)
2. **Lookup Operations**: O(1) for node lookup, dependency/dependent queries
3. **Topological Sort**: O(V + E) using DFS
4. **Memory Usage**: Additional O(V + E) for dependency maps

For large graphs with frequent queries, this upfront cost pays off with fast subsequent operations.

## Comparison with OrtGraphWrapper

| Operation | OrtGraphWrapper | Graph |
|-----------|----------------|--------|
| Purpose | Bridge ORT's incomplete API | In-memory graph operations |
| Data Source | ORT runtime graph | ONNX GraphProto |
| Dependency Analysis | Not available | Full dependency tracking |
| Topological Sort | Not available | Available |
| Performance | Direct ORT calls | Optimized for queries |
| Use Case | Runtime integration | Graph analysis/transformation |

Use `OrtGraphWrapper` when working with ORT runtime graphs, and `Graph` when you need efficient analysis of ONNX GraphProto structures.
