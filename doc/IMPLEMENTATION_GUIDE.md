# HipDNN Implementation Guide

**Version:** 2.0  
**Date:** January 10, 2026  
**Author:** Morphizen HipDNN Team

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Protocol Buffer Changes](#3-protocol-buffer-changes)
4. [Level-1 Pass Implementation](#4-level-1-pass-implementation)
5. [Custom Op Implementation](#5-custom-op-implementation)
6. [File I/O Helpers](#6-file-io-helpers)
7. [UID Extraction and Management](#7-uid-extraction-and-management)
8. [Complete Code Examples](#8-complete-code-examples)
9. [Testing Strategy](#9-testing-strategy)
10. [Migration Path](#10-migration-path)

---

## 1. Overview

### 1.1 Motivation

The initial implementation passed Conv operation parameters through the protocol buffer and rebuilt the hipDNN graph in the custom op. This approach had limitations:

- **Complexity**: Need to maintain parameter passing and graph reconstruction logic
- **Pattern Dependency**: Requires pattern matching for each operation type
- **Duplication**: Graph construction logic exists in both pass and custom op
- **Scalability**: Adding new operations requires modifying both components

### 1.2 Solution

**Serialize the hipDNN graph in the pass, deserialize and compile in the custom op.**

This approach:
- ✅ Simplifies the architecture - only filename is passed
- ✅ Centralizes graph creation logic in the pass
- ✅ Enables debugging - serialized graphs can be inspected
- ✅ Scales easily - same mechanism for all operations

### 1.3 Key Insight from hipDNN Frontend

The `Graph` class internally uses **backend descriptors** for all operations:

```cpp
// Graph.hpp internal implementation
std::unique_ptr<ScopedHipdnnBackendDescriptor> _graphDesc;
std::unique_ptr<ScopedHipdnnBackendDescriptor> _executionPlanDesc;

Error build_operation_graph(hipdnnHandle_t handle) {
    auto serializedGraph = buildFlatbufferOperationGraph();
    _graphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
        serializedGraph.data(), serializedGraph.size());
    // ... set handle, finalize ...
}
```

**We can use the same approach** - load serialized data into descriptors and compile directly!

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         ONNX Graph                              │
│                  (with supported operations)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Level-1 Pass (Compiler Time)                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Check operation support (IsSupportedOp)                     │
│  2. Build hipDNN Graph using symbol table pattern               │
│  3. Serialize graph using buildFlatbufferOperationGraph()       │
│  4. Save to file: hipdnn_graph_{output_name}.bin               │
│  5. Pass filename via proto parameter                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Fused Node (HIPDNN Custom Op)                   │
│                   graph_file_name = "hipdnn_graph_X.bin"        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Custom Op Constructor (Load Time)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Read graph_file_name from proto                             │
│  2. Load serialized graph from file                             │
│  3. Create graphDesc from buffer (ScopedHipdnnBackendDescriptor)│
│  4. Set handle and finalize descriptor                          │
│  5. Run compilation pipeline:                                   │
│     - InitializeHeuristicDescriptor()                          │
│     - InitializeEngineConfig()                                 │
│     - create execution plan                                    │
│     - get workspace size                                       │
│  6. Extract UIDs using GraphWrapper for variant pack           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Custom Op Compute (Inference Time)               │
├─────────────────────────────────────────────────────────────────┤
│  1. Build variant pack (UID → tensor pointer mapping)          │
│  2. Create variant pack descriptor                              │
│  3. Execute: backendExecute(handle, plan, variantPack)         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Key APIs Used |
|-----------|---------------|---------------|
| **Proto** | Define interface (filename only) | google::protobuf |
| **Level-1 Pass** | Check support, build & serialize graphs | hipdnn_frontend::graph::Graph |
| **Custom Op** | Load, compile, & execute | hipdnnBackend() APIs, GraphWrapper |

---

## 3. Protocol Buffer Changes

### 3.1 Proto Definition

**File:** `proto/hipdnn.proto`

```protobuf
// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
syntax = "proto3";
package hipdnn;

message HipdnnParamProto {
  // File path to the serialized hipDNN graph
  string graph_file_name = 1;
}
```

### 3.2 Rationale

- **Single Parameter**: Only the filename is needed
- **Simple**: No need to pass operation-specific parameters
- **Extensible**: Same mechanism works for all operation types
- **Debuggable**: Graph files can be inspected offline

---

## 4. Level-1 Pass Implementation

### 4.1 Overview

The Level-1 pass (`level-1-pass-hipdnn/src/pass_main.cpp`) follows the hip DNNEP architecture pattern with:

**Key Features:**
- ✅ Operation support checking (similar to `ep.cc::IsSupportedOp()`)
- ✅ Generic graph building with symbol table pattern (similar to `kernel.cc::BuildAndCompile()`)
- ✅ Operation dispatching through `AddNode()` function
- ✅ Extensible design for adding new operations

### 4.2 Operation Support Checking

```cpp
// Check if a Conv node is supported (similar to ep.cc)
static bool IsSupportedConv(const Node& node) {
  try {
    auto inputs = node_get_inputs(node);
    auto outputs = node_get_output_node_args(node);

    // Conv requires at least 2 inputs (X, W) and optionally bias
    if (inputs.size() < 2 || outputs.size() != 1) {
      return false;
    }

    // Check data types - we support float and float16
    auto x_dtype = node_arg_get_element_type(*inputs[0].node_arg);
    auto w_dtype = node_arg_get_element_type(*inputs[1].node_arg);
    auto y_dtype = node_arg_get_element_type(*outputs[0]);

    bool supported_type =
        (x_dtype == 1 || x_dtype == 10) &&  // FLOAT or FLOAT16
        x_dtype == w_dtype && x_dtype == y_dtype;

    if (!supported_type) return false;

    // Check if it's a 2D convolution (4D tensors: NCHW)
    auto x_shape = node_arg_get_shape_i64(*inputs[0].node_arg);
    auto w_shape = node_arg_get_shape_i64(*inputs[1].node_arg);

    if (!x_shape.has_value() || !w_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (x_shape->size() != 4 || w_shape->size() != 4) {
      return false;  // Only 2D conv supported
    }

    // Check auto_pad - only NOTSET supported (explicit padding)
    auto auto_pad_attr = node_try_get_attr_string(node, "auto_pad");
    if (auto_pad_attr.has_value() && auto_pad_attr.value() != "NOTSET") {
      return false;
    }

    // Check group - only 1 supported (no grouped/depthwise convolutions)
    auto group = node_get_attr_int_with_default(node, "group", 1);
    if (group != 1) return false;

    // Check dilations - only [1,1] supported
    auto dilations = node_get_attr_ints(node, "dilations");
    if (!dilations.empty()) {
      if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
        return false;
      }
    }

    return true;
  } catch (...) {
    return false;
  }
}

// Check if an op is supported (dispatcher)
static bool IsSupportedOp(const Node& node) {
  std::string op_type = node_op_type(node);

  if (op_type == "Conv") {
    return IsSupportedConv(node);
  }

  // Add more operations here as we implement them
  return false;
}
```

### 4.3 Tensor Attribute Creation

```cpp
// Create TensorAttributes from NodeArg (similar to kernel.cc)
bool CreateTensorAttr(
    const Graph& ort_graph,
    const NodeArg& node_arg,
    int64_t uid,
    TensorAttrPtr& out_attr) {
  using hipdnn_frontend::graph::TensorAttributes;

  auto node_arg_ref = NodeArgConstRef::from_node_arg(ort_graph, node_arg);
  std::string name = node_arg_get_name(node_arg_ref);

  auto shape = node_arg_get_shape_i64(node_arg_ref);
  if (!shape.has_value()) {
    MY_LOG(1) << "Value must have static shape: " << name;
    return false;
  }

  auto dtype = ToHipDNNDataType(node_arg_get_element_type(node_arg_ref));
  if (!dtype.has_value()) {
    MY_LOG(1) << "Unsupported data type for value: " << name;
    return false;
  }

  out_attr = std::make_shared<TensorAttributes>();
  out_attr->set_uid(uid)
      .set_name(name)
      .set_data_type(dtype.value())
      .set_dim(shape.value())
      .set_stride(ComputeStrides(shape.value()));

  return true;
}
```

### 4.4 Operation Node Addition

```cpp
// Add Conv operation to hipDNN graph
bool AddConvNode(
    const Graph& ort_graph,
    hipdnn_frontend::graph::Graph& graph,
    const Node& node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;

  if (input_attrs.size() < 2) {
    MY_LOG(1) << "Conv requires at least 2 input tensor attributes";
    return false;
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

  // Extract Conv attributes
  auto pads = node_get_attr_ints(node, "pads");
  auto strides = node_get_attr_ints(node, "strides");
  auto dilations = node_get_attr_ints(node, "dilations");

  // Normalize to vectors
  std::vector<int64_t> pads_vec(pads.begin(), pads.end());
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  std::vector<int64_t> dilations_vec(dilations.begin(), dilations.end());

  // Normalize padding format
  if (pads_vec.empty()) {
    pads_vec = {0, 0, 0, 0};
  } else if (pads_vec.size() == 2) {
    pads_vec = {pads_vec[0], pads_vec[1], pads_vec[0], pads_vec[1]};
  } else if (pads_vec.size() != 4) {
    MY_LOG(1) << "Conv pads must have 2 or 4 elements";
    return false;
  }

  if (strides_vec.empty()) strides_vec = {1, 1};
  if (dilations_vec.empty()) dilations_vec = {1, 1};

  // Determine compute data type
  auto compute_dtype = GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    MY_LOG(1) << "Unsupported data type combination for Conv compute";
    return false;
  }

  // Create convolution attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads_vec[0], pads_vec[1]})
      .set_stride({strides_vec[0], strides_vec[1]})
      .set_dilation({dilations_vec[0], dilations_vec[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());

  // Add convolution to graph - returns output tensor attributes
  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);

  return true;
}

// Dispatcher for operation-specific graph building
bool AddNode(
    const Graph& ort_graph,
    hipdnn_frontend::graph::Graph& graph,
    const Node& node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs) {
  std::string op_type = node_op_type(node);

  if (op_type == "Conv") {
    TensorAttrPtr y_attr;
    if (!AddConvNode(ort_graph, graph, node, input_attrs, y_attr)) {
      return false;
    }
    output_attrs.push_back(y_attr);
    return true;
  }

  MY_LOG(1) << "Unsupported op type: " << op_type;
  return false;
}
```

### 4.5 Generic Graph Building with Symbol Table Pattern

```cpp
// Build hipDNN graph generically using symbol table pattern (similar to kernel.cc)
bool BuildAndSerializeGraph(
    const Graph& ort_graph,
    const Node& fused_node,
    const std::vector<NodeInput>& graph_inputs,
    const std::vector<const NodeArg*>& graph_outputs,
    std::string& out_filename) {
  using hipdnn_frontend::graph::Graph;

  try {
    // Create hipDNN graph
    auto graph = std::make_unique<Graph>();
    
    auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *graph_outputs[0]);
    graph->set_name("Graph_" + node_arg_get_name(output_ref));
    
    int64_t next_uid = 1;
    std::unordered_map<std::string, TensorAttrPtr> symbol_table;
    std::vector<int64_t> input_uids;
    std::vector<int64_t> output_uids;

    // STEP 1: Create TensorAttributes for all graph inputs
    for (const auto& input : graph_inputs) {
      TensorAttrPtr attr;
      if (!CreateTensorAttr(ort_graph, *input.node_arg, next_uid++, attr)) {
        return false;
      }
      attr->set_is_virtual(false);
      auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *input.node_arg);
      symbol_table[node_arg_get_name(input_ref)] = attr;
      input_uids.push_back(attr->get_uid());
    }

    // STEP 2: Process each node in the fused subgraph
    auto subgraph_nodes = node_get_inputs(fused_node);
    for (const auto& input : subgraph_nodes) {
      if (!input.node) continue;
      
      const Node& node = *input.node;
      
      // Look up input TensorAttributes from symbol table
      auto node_inputs = node_get_inputs(node);
      std::vector<TensorAttrPtr> input_attrs;
      for (const auto& node_input : node_inputs) {
        auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *node_input.node_arg);
        std::string name = node_arg_get_name(input_ref);
        auto it = symbol_table.find(name);
        if (it == symbol_table.end()) {
          MY_LOG(1) << "Input not found in symbol table: " << name;
          return false;
        }
        input_attrs.push_back(it->second);
      }

      // Add the node to hipDNN graph
      std::vector<TensorAttrPtr> output_attrs;
      if (!AddNode(ort_graph, *graph, node, input_attrs, output_attrs)) {
        return false;
      }

      // Set UID, name on output TensorAttributes and add to symbol table
      auto node_outputs = node_get_output_node_args(node);
      for (size_t i = 0; i < output_attrs.size(); ++i) {
        auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *node_outputs[i]);
        std::string name = node_arg_get_name(output_ref);

        auto dtype = ToHipDNNDataType(node_arg_get_element_type(output_ref));
        auto shape = node_arg_get_shape_i64(output_ref);
        if (!dtype.has_value() || !shape.has_value()) return false;

        output_attrs[i]->set_uid(next_uid++)
            .set_name(name)
            .set_data_type(dtype.value())
            .set_dim(shape.value())
            .set_stride(ComputeStrides(shape.value()));
        symbol_table[name] = output_attrs[i];
      }
    }

    // STEP 3: Mark graph outputs as non-virtual
    for (const auto* output_arg : graph_outputs) {
      auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *output_arg);
      std::string name = node_arg_get_name(output_ref);
      auto it = symbol_table.find(name);
      if (it == symbol_table.end()) return false;
      
      it->second->set_is_virtual(false);
      output_uids.push_back(it->second->get_uid());
    }

    // STEP 4: Serialize and save
    flatbuffers::DetachedBuffer buffer = graph->buildFlatbufferOperationGraph();
    out_filename = "hipdnn_graph_" + node_arg_get_name(output_ref) + ".bin";
    SaveGraphToFile(buffer, out_filename);
    
    return true;
  } catch (const std::exception& ex) {
    MY_LOG(1) << "Exception building/serializing hipDNN graph: " << ex.what();
    return false;
  }
}
```

### 4.6 Main Processing Loop

```cpp
struct Level1HipDnn {
  void process(IPass& self, Graph& ort_graph) {
    auto node_indices = graph_get_node_in_topoligical_order(ort_graph);
    
    for (auto node_idx : node_indices) {
      auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
      
      // Check if operation is supported
      if (!IsSupportedOp(*node)) {
        continue;
      }
      
      std::string op_type = node_op_type(*node);
      MY_LOG(1) << "Found supported " << op_type << " node";
      
      // Get inputs and outputs
      auto node_inputs = node_get_inputs(*node);
      auto node_outputs = node_get_output_node_args(*node);
      
      // Build names for fusion
      std::vector<std::string> input_names, output_names;
      for (const auto& input : node_inputs) {
        auto input_ref = NodeArgConstRef::from_node_arg(ort_graph, *input.node_arg);
        input_names.push_back(node_arg_get_name(input_ref));
      }
      for (const auto* output : node_outputs) {
        auto output_ref = NodeArgConstRef::from_node_arg(ort_graph, *output);
        output_names.push_back(node_arg_get_name(output_ref));
      }
      
      // Build and serialize graph
      std::string graph_filename;
      if (!BuildAndSerializeGraph(ort_graph, *node, node_inputs, node_outputs, graph_filename)) {
        continue;
      }
      
      // Fuse and attach proto
      auto [meta_def, fuse_error] =
          self_.try_fuse(ort_graph, output_names[0], input_names, output_names, {}, "HIPDNN");
      
      if (meta_def == nullptr) continue;
      
      auto hipdnn_param = hipdnn::HipdnnParamProto();
      hipdnn_param.set_graph_file_name(graph_filename);
      
      auto hipdnn_json_str = std::string();
      google::protobuf::util::MessageToJsonString(hipdnn_param, &hipdnn_json_str);
      self_.attach_meta_def_param(*meta_def, hipdnn_json_str.c_str());
      self_.fuse(ort_graph, std::move(*meta_def));
    }
  }
};
```

### 4.7 Key Design Points

1. **Operation Support Checking**: `IsSupportedOp()` → `IsSupportedConv()` pattern
2. **Symbol Table Pattern**: Maintains `name → TensorAttr` mapping during graph building
3. **Generic Dispatching**: `AddNode()` dispatches to operation-specific implementations
4. **UID Management**: Sequential UID assignment for all tensors
5. **Non-Virtual Tensors**: Graph inputs/outputs marked as `is_virtual(false)`
6. **Extensibility**: Easy to add new operations by implementing `IsSupportedXXX()` and `AddXXXNode()`

---

## 5. Custom Op Implementation

### 5.1 Overview

The custom op (`custom-op-hipdnn/src/custom_op.cpp`) loads the serialized graph, compiles it using backend descriptors, and executes it.

**Architecture:**
- Constructor: Load graph → Compile → Extract UIDs
- Compute: Build variant pack → Execute

### 5.2 Class Definition

```cpp
class HipdnnCustomOp : public CustomOpImp {
public:
  HipdnnCustomOp(std::shared_ptr<const PassContext> context,
                 const std::shared_ptr<MetaDefProto>& meta_def,
                 onnxruntime::Model* model);
  virtual ~HipdnnCustomOp();

private:
  void Compute(const OrtApi* api, OrtKernelContext* context) const override;
  
  void LoadAndCompileGraph();
  void InitializeHeuristicDescriptor();
  void InitializeEngineConfig();
  void ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer);

private:
  HipdnnParamProto hipdnn_proto_;
  hipdnnHandle_t handle_;
  
  // Backend descriptors (mimics Graph class internals)
  std::unique_ptr<ScopedHipdnnBackendDescriptor> graphDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> engineHeuristicDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> engineConfigDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> executionPlanDesc_;
  
  std::vector<char> workspace_;
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  std::vector<std::vector<int64_t>> output_shapes_;
};
```

### 5.3 Constructor

```cpp
HipdnnCustomOp::HipdnnCustomOp(...) : CustomOpImp(context, meta_def, model), handle_(nullptr) {
  // Parse proto to get graph filename
  auto hipdnn_json_str = get_meta_def_param();
  google::protobuf::util::JsonStringToMessage(hipdnn_json_str, &hipdnn_proto_);
  
  // Create hipDNN handle
  hipdnnCreate(&handle_);
  
  // Load and compile the graph
  LoadAndCompileGraph();
}
```

### 5.4 Graph Loading and Compilation

```cpp
void HipdnnCustomOp::LoadAndCompileGraph() {
  // Load serialized graph
  std::vector<uint8_t> buffer = LoadGraphFromFile(hipdnn_proto_.graph_file_name());
  
  // Create graph descriptor from buffer
  graphDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(buffer.data(), buffer.size());
  
  hipdnnBackend()->backendSetAttribute(graphDesc_->get(), 
      HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle_);
  hipdnnBackend()->backendFinalize(graphDesc_->get());
  
  // Create execution plans
  InitializeHeuristicDescriptor();
  InitializeEngineConfig();
  
  // Build execution plan
  executionPlanDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
  hipdnnBackend()->backendSetAttribute(executionPlanDesc_->get(),
      HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 
      1, &engineConfigDesc_->get());
  hipdnnBackend()->backendFinalize(executionPlanDesc_->get());
  
  // Get workspace size
  int64_t workspace_size = 0;
  hipdnnBackend()->backendGetAttribute(executionPlanDesc_->get(),
      HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, HIPDNN_TYPE_INT64, 
      1, nullptr, &workspace_size);
  if (workspace_size > 0) workspace_.resize(workspace_size);
  
  // Extract UIDs
  ExtractUIDsFromSerializedGraph(buffer);
}
```

---

## 6. File I/O Helpers

### 6.1 Loading Graph from File

```cpp
static std::vector<uint8_t> LoadGraphFromFile(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open file for reading: " + filepath);
  }
  
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read graph from file: " + filepath);
  }
  
  return buffer;
}
```

### 6.2 Saving Graph to File

```cpp
void SaveGraphToFile(const flatbuffers::DetachedBuffer& buffer, 
                     const std::string& filepath) {
  std::ofstream file(filepath, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }
  
  file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  file.close();
  
  if (!file.good()) {
    throw std::runtime_error("Failed to write graph to file: " + filepath);
  }
}
```

---

## 7. UID Extraction and Management

### 7.1 Overview

UIDs (Unique Identifiers) map ONNX Runtime tensors to hipDNN graph tensors during execution.

### 7.2 UID Extraction Using GraphWrapper

```cpp
void HipdnnCustomOp::ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer) {
  using namespace hipdnn_plugin_sdk;
  
  GraphWrapper graphWrapper(buffer.data(), buffer.size());
  if (!graphWrapper.isValid()) {
    throw std::runtime_error("Invalid serialized graph");
  }
  
  // Get tensor map: UID → TensorAttributes
  auto tensorMap = graphWrapper.getTensorMap();
  
  // Build set of output UIDs by examining node attributes
  std::unordered_set<int64_t> outputUids;
  for (uint32_t i = 0; i < graphWrapper.nodeCount(); ++i) {
    auto& node = graphWrapper.getNode(i);
    
    if (node.attributes_type() == hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes) {
      auto* conv_attrs = node.attributes_as_ConvolutionFwdAttributes();
      if (conv_attrs) {
        outputUids.insert(conv_attrs->y_tensor_uid());
      }
    }
  }
  
  // Classify non-virtual tensors as inputs or outputs
  std::vector<std::pair<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>> sortedTensors;
  for (const auto& [uid, tensor] : tensorMap) {
    if (!tensor->virtual_()) {
      sortedTensors.push_back({uid, tensor});
    }
  }
  
  // Sort by UID
  std::sort(sortedTensors.begin(), sortedTensors.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  
  // Classify as inputs or outputs
  for (const auto& [uid, tensor] : sortedTensors) {
    if (outputUids.count(uid) > 0) {
      output_uids_.push_back(uid);
      auto dims = tensor->dims();
      if (dims) {
        std::vector<int64_t> shape(dims->begin(), dims->end());
        output_shapes_.push_back(shape);
      }
    } else {
      input_uids_.push_back(uid);
    }
  }
}
```

### 7.3 Variant Pack Construction

```cpp
void HipdnnCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  Ort::KernelContext ctx(context);
  
  // Build variant pack: UID → tensor pointer mapping
  std::unordered_map<int64_t, void*> variant_pack;
  
  for (size_t i = 0; i < input_uids_.size(); ++i) {
    Ort::ConstValue input = ctx.GetInput(i);
    variant_pack[input_uids_[i]] = const_cast<void*>(input.GetTensorRawData());
  }
  
  for (size_t i = 0; i < output_uids_.size(); ++i) {
    Ort::UnownedValue output = ctx.GetOutput(i, output_shapes_[i]);
    variant_pack[output_uids_[i]] = output.GetTensorMutableRawData();
  }
  
  // Convert to arrays
  std::vector<int64_t> keys;
  std::vector<void*> values;
  for (const auto& [key, value] : variant_pack) {
    keys.push_back(key);
    values.push_back(value);
  }
  
  // Create and execute variant pack
  auto variantPackDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  
  hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS, HIPDNN_TYPE_VOID_PTR,
      static_cast<int64_t>(values.size()), values.data());
  
  hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64,
      static_cast<int64_t>(keys.size()), keys.data());
  
  void* workspace_ptr = workspace_.empty() ? nullptr : 
                        const_cast<void*>(static_cast<const void*>(workspace_.data()));
  hipdnnBackend()->backendSetAttribute(variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_WORKSPACE, HIPDNN_TYPE_VOID_PTR, 1, &workspace_ptr);
  
  hipdnnBackend()->backendFinalize(variantPackDesc->get());
  hipdnnBackend()->backendExecute(handle_, executionPlanDesc_->get(), variantPackDesc->get());
}
```

---

## 8. Complete Code Examples

### 8.1 Adding a New Operation

To add support for a new operation (e.g., BatchNorm):

**Step 1: Add support checking in pass:**
```cpp
static bool IsSupportedBatchNorm(const Node& node) {
  // Check inputs, outputs, data types, shapes, etc.
  return true;  // if supported
}

static bool IsSupportedOp(const Node& node) {
  std::string op_type = node_op_type(node);
  if (op_type == "Conv") return IsSupportedConv(node);
  if (op_type == "BatchNormalization") return IsSupportedBatchNorm(node);
  return false;
}
```

**Step 2: Add node building function:**
```cpp
bool AddBatchNormNode(
    const Graph& ort_graph,
    hipdnn_frontend::graph::Graph& graph,
    const Node& node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  
  // Extract attributes and call graph.batch_norm()
  output_attr = graph.batch_norm(input_attrs[0], input_attrs[1], 
                                  input_attrs[2], bn_attrs);
  return true;
}

bool AddNode(...) {
  if (op_type == "Conv") return AddConvNode(...);
  if (op_type == "BatchNormalization") return AddBatchNormNode(...);
  return false;
}
```

**Step 3: Add UID extraction in custom op:**
```cpp
if (node.attributes_type() == hipdnn_data_sdk::data_objects::NodeAttributes::BatchNormAttributes) {
  auto* bn_attrs = node.attributes_as_BatchNormAttributes();
  if (bn_attrs) {
    outputUids.insert(bn_attrs->y_tensor_uid());
  }
}
```

That's it! The same serialization/deserialization mechanism handles the new operation.

---

## 9. Testing Strategy

### 9.1 Unit Testing

```cpp
TEST(HipDNNPass, SupportedOpsCheck) {
  // Test operation support checking
  EXPECT_TRUE(IsSupportedConv(valid_conv_node));
  EXPECT_FALSE(IsSupportedConv(unsupported_conv_node));
}

TEST(HipDNNPass, GraphSerialization) {
  std::string filename;
  bool success = BuildAndSerializeGraph(ort_graph, node, inputs, outputs, filename);
  ASSERT_TRUE(success);
  ASSERT_TRUE(std::filesystem::exists(filename));
}
```

### 9.2 Integration Testing

```cpp
TEST(HipDNNIntegration, EndToEnd) {
  auto model = LoadONNXModel("test_model.onnx");
  RunLevel1Pass(model);
  
  auto session = CreateSession(model);
  auto output = session.Run(input_data);
  
  CompareWithReference(output, expected_output);
}
```

---

## 10. Migration Path

### 10.1 Comparison

| Aspect | Before | After |
|--------|--------|-------|
| **Architecture** | Parameter passing | Graph serialization |
| **Support Checking** | Pattern matching | `IsSupportedOp()` function |
| **Graph Building** | Conv-specific code | Generic symbol table pattern |
| **Extensibility** | Modify both pass and custom op | Add `IsSupportedXXX()` and `AddXXXNode()` |
| **Debugging** | Hard (parameters in proto) | Easy (inspect .bin files) |

### 10.2 Benefits

1. **Better Separation**: Support checking, graph building, serialization are separate
2. **Easier Extension**: Adding operations only requires new `IsSupportedXXX()` and `AddXXXNode()`

---

## Appendix A: Key References

- **hipDNN Frontend**: `c:/Develop/TheRock/include/hipdnn/frontend/hipdnn_frontend/Graph.hpp`
- **GraphWrapper**: `c:/Develop/TheRock/include/hipdnn/data_sdk/hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp`
- **Level-1 Pass**: `level-1-pass-hipdnn/src/pass_main.cpp`
- **Custom Op**: `custom-op-hipdnn/src/custom_op.cpp`

## Appendix B: Glossary

- **UID**: Unique Identifier for tensors in hipDNN graph
- **Variant Pack**: Mapping of UIDs to tensor data pointers for execution
- **Backend Descriptor**: Low-level hipDNN descriptor objects
- **ScopedHipdnnBackendDescriptor**: RAII wrapper for backend descriptors
- **GraphWrapper**: Utility to read FlatBuffer serialized graphs
- **Symbol Table**: Map of tensor names to TensorAttributes during graph building

---

**End of Document**
