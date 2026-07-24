<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# morphizen-graph

C++ wrapper utilities for graph operations over MORPHIZEN_ORT_API interface.

## Overview

`morphizen-graph` provides clean, type-safe C++ wrappers over the MORPHIZEN_ORT_API function pointer interface. This component encapsulates low-level graph operations and provides a modern C++ object model for working with ONNX computational graphs.

## Features

- **Graph Operations**: `Graph`, `GraphRef`, `GraphConstRef` classes for graph manipulation
- **Node Operations**: `Node`, `NodeRef`, `NodeConstRef` classes for working with operators
- **NodeArg Operations**: `NodeArg`, `NodeArgRef` classes for tensor values
- **NodeInput**: Abstraction combining NodeArg with optional producer Node
- **NodeAttr**: Node attribute handling
- **NodeBuilder**: High-level API for constructing nodes (requires IPass from morphizen-core)

## Architecture

```
┌──────────────────────────────────────┐
│     morphizen-graph (~4,000 LOC)    │
│                                      │
│  C++ Wrappers:                       │
│  - Graph, GraphRef, GraphConstRef    │
│  - Node, NodeRef, NodeConstRef       │
│  - NodeArg, NodeArgRef               │
│  - NodeInput                         │
│  - NodeAttr, NodeAttributesBuilder   │
│  - NodeBuilder (high-level)          │
│                                      │
│  Calls MORPHIZEN_ORT_API internally       │
│  ↓                                   │
└──────────────────────────────────────┘
         │
         ↓
┌──────────────────────────────────────┐
│      morphizen-ort-api-ext                │
│      MORPHIZEN_ORT_API interface          │
│      (111 function pointers)         │
└──────────────────────────────────────┘
         │
         ↓
    Implemented by backends:
    - onnx-ir-imp
    - mlir-imp
    - ORT native
```

## Dependencies

- **PUBLIC**: `morphizen-ort-api-ext` - MORPHIZEN_ORT_API interface
- **PRIVATE**: `glog::glog` - Logging

## Usage

### Basic Graph Operations

```cpp
#include <morphizen/graph.hpp>
#include <morphizen/node.hpp>
#include <morphizen/node_arg.hpp>

// Get graph nodes
const Graph& graph = ...;
auto nodes = graph_get_nodes(graph);

// Query node information
const Node& node = ...;
std::string op_type = node_op_type(node);
auto inputs = node_get_inputs(node);
auto outputs = node_get_output_node_args(node);

// Work with node arguments
const NodeArg& arg = ...;
std::string name = node_arg_get_name(arg);
int elem_type = node_arg_get_element_type(arg);
auto shape = node_arg_get_shape_i64(arg);
```

### Building Nodes (requires morphizen-core for IPass)

```cpp
#include <morphizen/graph.hpp>

Graph& graph = ...;
IPass& pass = ...;

// Build a new node
NodeBuilder(graph, pass)
    .set_op_type("Conv", "com.xilinx")
    .set_input_node_args({input_arg, weight_arg})
    .set_anchor_point(...) // Optional: for quantization
    .build();
```

## Backend Independence

All operations in `morphizen-graph` work with any backend through the MORPHIZEN_ORT_API abstraction layer. The active backend is selected at runtime via the `MORPHIZEN_ORT_BRIDGE_BACKEND` environment variable.

## Size

- ~4,000 lines of code
- Single point of MORPHIZEN_ORT_API calls
- No external dependencies beyond morphizen-ort-api-ext

## Building

```bash
cmake -B build
cmake --build build --target morphizen-graph
```

## License

Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
