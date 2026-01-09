# HipDNN Graph Serialization Implementation Guide

**Version:** 1.0  
**Date:** January 9, 2026  
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

The current implementation passes Conv operation parameters (pads, strides, dilations, etc.) through the protocol buffer and rebuilds the hipDNN graph in the custom op. This approach has limitations:

- **Complexity**: Need to maintain parameter passing and graph reconstruction logic
- **Pattern Dependency**: Requires pattern matching for each operation type
- **Duplication**: Graph construction logic exists in both pass and custom op
- **Scalability**: Adding new operations requires modifying both components

### 1.2 Proposed Solution

**Serialize the hipDNN graph in the pass, deserialize and compile in the custom op.**

This approach:
- ✅ Simplifies the architecture - only filename is passed
- ✅ Eliminates pattern matching - just check op type
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
│                    (with Conv operations)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Level-1 Pass (Compiler Time)                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Identify Conv nodes (no pattern matching needed)            │
│  2. Build hipDNN Graph from Conv node attributes                │
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
│     - initializeHeuristicDescriptor()                          │
│     - initializeEngineConfig()                                 │
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
| **Level-1 Pass** | Build & serialize graphs | hipdnn_frontend::graph::Graph |
| **Custom Op** | Load, compile, & execute | hipdnnBackend() APIs, GraphWrapper |

---

## 3. Protocol Buffer Changes

### 3.1 New Proto Definition

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

The Level-1 pass (`level-1-pass-hipdnn/src/pass_main.cpp`) identifies Conv operations in the ONNX graph, builds corresponding hipDNN graphs, serializes them to files, and creates fused nodes.

**Key Features:**
- ✅ No pattern matching required - simply checks op type
- ✅ Direct node iteration in topological order
- ✅ Uses hipDNN Frontend Graph API for graph construction
- ✅ Serializes using `buildFlatbufferOperationGraph()`
- ✅ Minimal proto - only passes filename

### 4.2 Main Processing Loop

```cpp
struct Level1HipDnn {
  Level1HipDnn(IPass& self) : self_{self} {}
  
  void process(IPass& self, Graph& ort_graph) {
    // Iterate through all nodes in topological order
    auto node_indices = graph_get_node_in_topoligical_order(ort_graph);
    
    for (auto node_idx : node_indices) {
      auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
      auto node_ref = NodeConstRef::from_node(ort_graph, *node);
      
      // Simple type check - no pattern matching!
      if (node_op_type(*node) != "Conv") {
        continue;
      }
      
      // Extract inputs/outputs
      auto conv_inputs = node_get_inputs(*node);
      auto conv_output_node_args = node_get_output_node_args(*node);
      
      // Build and serialize graph
      std::string graph_filename;
      bool success = BuildAndSerializeGraph(
          *node, input_data, weight_data, output_data, graph_filename);
      
      if (!success) continue;
      
      // Create fused node with graph filename
      auto [meta_def, fuse_error] = self_.try_fuse(
          ort_graph, unique_id, 
          {input_data.name(), weight_data.name()}, 
          {output_data.name()},
          {}, "HIPDNN");
      
      // Create proto with only the graph filename
      auto hipdnn_param = hipdnn::HipdnnParamProto();
      hipdnn_param.set_graph_file_name(graph_filename);
      
      // Serialize to JSON and attach
      auto hipdnn_json_str = std::string();
      google::protobuf::util::MessageToJsonString(hipdnn_param, &hipdnn_json_str);
      self_.attach_meta_def_param(*meta_def, hipdnn_json_str.c_str());
      self_.fuse(ort_graph, std::move(*meta_def));
    }
  }
  
  IPass& self_;
};
```

### 4.3 Graph Building and Serialization

```cpp
bool BuildAndSerializeGraph(
    const Node& conv_node,
    const NodeArgConstRef& input_ref,
    const NodeArgConstRef& weight_ref,
    const NodeArgConstRef& output_ref,
    std::string& out_filename) {
  
  using HipDNNGraph = hipdnn_frontend::graph::Graph;
  using hipdnn_frontend::graph::TensorAttributes;
  using hipdnn_frontend::graph::ConvFpropAttributes;
  
  // Step 1: Create graph
  auto graph = std::make_unique<HipDNNGraph>();
  graph->set_name("Conv_" + node_arg_get_name(output_ref));
  
  int64_t next_uid = 1;
  
  // Step 2: Get shapes and data types from ONNX
  auto input_shape = node_arg_get_shape_i64(input_ref);
  auto weight_shape = node_arg_get_shape_i64(weight_ref);
  auto output_shape = node_arg_get_shape_i64(output_ref);
  
  auto input_dtype = ToHipDNNDataType(node_arg_get_element_type(input_ref));
  auto weight_dtype = ToHipDNNDataType(node_arg_get_element_type(weight_ref));
  auto output_dtype = ToHipDNNDataType(node_arg_get_element_type(output_ref));
  
  // Step 3: Create tensor attributes with UIDs
  auto x_attr = std::make_shared<TensorAttributes>();
  x_attr->set_uid(next_uid++)
      .set_name(node_arg_get_name(input_ref))
      .set_data_type(input_dtype.value())
      .set_dim(*input_shape)
      .set_stride(ComputeStrides(*input_shape))
      .set_is_virtual(false);  // Graph input
  
  auto w_attr = std::make_shared<TensorAttributes>();
  w_attr->set_uid(next_uid++)
      .set_name(node_arg_get_name(weight_ref))
      .set_data_type(weight_dtype.value())
      .set_dim(*weight_shape)
      .set_stride(ComputeStrides(*weight_shape))
      .set_is_virtual(false);  // Graph input
  
  // Step 4: Extract Conv attributes from ONNX
  auto pads = node_get_attr_ints(conv_node, "pads");
  auto strides = node_get_attr_ints(conv_node, "strides");
  auto dilations = node_get_attr_ints(conv_node, "dilations");
  
  // Normalize to vectors
  std::vector<int64_t> pads_vec(pads.begin(), pads.end());
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  std::vector<int64_t> dilations_vec(dilations.begin(), dilations.end());
  
  // Step 5: Create convolution attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads_vec[0], pads_vec[1]})
      .set_stride({strides_vec[0], strides_vec[1]})
      .set_dilation({dilations_vec[0], dilations_vec[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());
  
  // Step 6: Add operation to graph
  auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);
  
  // Set output properties
  y_attr->set_uid(next_uid++)
      .set_name(node_arg_get_name(output_ref))
      .set_data_type(output_dtype.value())
      .set_dim(*output_shape)
      .set_stride(ComputeStrides(*output_shape))
      .set_is_virtual(false);  // Graph output
  
  // Step 7: Serialize the graph
  flatbuffers::DetachedBuffer buffer = graph->buildFlatbufferOperationGraph();
  
  // Step 8: Generate unique filename
  out_filename = "hipdnn_graph_" + node_arg_get_name(output_ref) + ".bin";
  
  // Step 9: Save to file
  SaveGraphToFile(buffer, out_filename);
  
  return true;
}
```

### 4.4 Helper Functions

```cpp
// Compute strides from shape (NCHW layout)
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Convert ONNX data type to hipDNN data type
std::optional<hipdnn_frontend::DataType> ToHipDNNDataType(int32_t onnx_dtype) {
  using hipdnn_frontend::DataType;
  switch (onnx_dtype) {
    case 1:  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
      return DataType::FLOAT;
    case 10: // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
      return DataType::HALF;
    default:
      return std::nullopt;
  }
}

// Save FlatBuffer to file
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

### 4.5 Key Design Points

1. **No Pattern Matching**: Simply checks `node_op_type(*node) != "Conv"`
2. **Direct Attribute Access**: Uses ONNX node attributes directly
3. **UID Assignment**: Assigns sequential UIDs (1, 2, 3, ...) to tensors
4. **Non-Virtual Tensors**: All inputs/outputs marked as `is_virtual(false)` for proper UID extraction
5. **File Naming**: Uses output tensor name for unique filenames

---

## 5. Custom Op Implementation

### 5.1 Overview

The custom op (`custom-op-hipdnn/src/custom_op.cpp`) loads the serialized graph, compiles it using backend descriptors (mimicking the Graph class internals), and executes it.

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
  
  // Graph loading and compilation helpers
  void LoadAndCompileGraph();
  void InitializeHeuristicDescriptor();
  void InitializeEngineConfig();
  void ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer);

private:
  HipdnnParamProto hipdnn_proto_;           // Proto parameter
  hipdnnHandle_t handle_;                    // hipDNN handle
  
  // Backend descriptors (mimics Graph class internals)
  std::unique_ptr<ScopedHipdnnBackendDescriptor> graphDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> engineHeuristicDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> engineConfigDesc_;
  std::unique_ptr<ScopedHipdnnBackendDescriptor> executionPlanDesc_;
  
  // Execution resources
  std::vector<char> workspace_;
  
  // UID mappings for variant pack construction
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  std::vector<std::vector<int64_t>> output_shapes_;
};
```

### 5.3 Constructor - Load and Compile

```cpp
HipdnnCustomOp::HipdnnCustomOp(
    std::shared_ptr<const PassContext> context,
    const std::shared_ptr<MetaDefProto>& meta_def,
    onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model), handle_(nullptr) {
  
  // Step 1: Parse proto to get graph filename
  auto hipdnn_json_str = get_meta_def_param();
  google::protobuf::util::JsonStringToMessage(hipdnn_json_str, &hipdnn_proto_);
  
  LOG(INFO) << "Graph file: " << hipdnn_proto_.graph_file_name();
  
  // Step 2: Create hipDNN handle
  hipdnnCreate(&handle_);
  
  // Step 3: Load and compile the graph
  LoadAndCompileGraph();
}
```

### 5.4 Graph Loading and Compilation Pipeline

```cpp
void HipdnnCustomOp::LoadAndCompileGraph() {
  using namespace hipdnn_frontend;
  
  // STEP 1: Load serialized graph from file
  std::string filename = hipdnn_proto_.graph_file_name();
  std::vector<uint8_t> buffer = LoadGraphFromFile(filename);
  
  // STEP 2: Create graph descriptor from buffer
  // (Same as Graph::build_operation_graph does internally)
  graphDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
      buffer.data(), buffer.size());
  
  // Set handle on graph descriptor
  hipdnnBackend()->backendSetAttribute(
      graphDesc_->get(),
      HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
      HIPDNN_TYPE_HANDLE,
      1,
      &handle_);
  
  // Finalize graph descriptor
  hipdnnBackend()->backendFinalize(graphDesc_->get());
  
  // STEP 3: Create execution plans (same as Graph::create_execution_plans)
  InitializeHeuristicDescriptor();
  InitializeEngineConfig();
  
  // STEP 4: Build execution plan
  executionPlanDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
  
  hipdnnBackend()->backendSetAttribute(
      executionPlanDesc_->get(),
      HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &engineConfigDesc_->get());
  
  hipdnnBackend()->backendFinalize(executionPlanDesc_->get());
  
  // STEP 5: Get workspace size
  int64_t workspace_size = 0;
  hipdnnBackend()->backendGetAttribute(
      executionPlanDesc_->get(),
      HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
      HIPDNN_TYPE_INT64,
      1,
      nullptr,
      &workspace_size);
  
  if (workspace_size > 0) {
    workspace_.resize(workspace_size);
  }
  
  // STEP 6: Extract UIDs for variant pack mapping
  ExtractUIDsFromSerializedGraph(buffer);
}
```

### 5.5 Heuristic Descriptor Initialization

```cpp
void HipdnnCustomOp::InitializeHeuristicDescriptor() {
  using namespace hipdnn_frontend;
  
  // Create heuristic descriptor
  engineHeuristicDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  
  // Set operation graph
  hipdnnBackend()->backendSetAttribute(
      engineHeuristicDesc_->get(),
      HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &graphDesc_->get());
  
  // Set heuristic mode to FALLBACK
  hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
  hipdnnBackend()->backendSetAttribute(
      engineHeuristicDesc_->get(),
      HIPDNN_ATTR_ENGINEHEUR_MODE,
      HIPDNN_TYPE_HEUR_MODE,
      1,
      &mode);
  
  // Finalize
  hipdnnBackend()->backendFinalize(engineHeuristicDesc_->get());
}
```

### 5.6 Engine Configuration Selection

```cpp
void HipdnnCustomOp::InitializeEngineConfig() {
  using namespace hipdnn_frontend;
  
  // Get number of available engine configurations
  int64_t availableEngineCount = 0;
  hipdnnBackend()->backendGetAttribute(
      engineHeuristicDesc_->get(),
      HIPDNN_ATTR_ENGINEHEUR_RESULTS,
      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
      0,
      &availableEngineCount,
      nullptr);
  
  if (availableEngineCount == 0) {
    throw std::runtime_error("No engine configurations available");
  }
  
  // Get the first (best) engine configuration
  auto engineCfgDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
  
  hipdnnBackendDescriptor_t engineCfgPtr = engineCfgDesc->get();
  int64_t count = 0;
  hipdnnBackend()->backendGetAttribute(
      engineHeuristicDesc_->get(),
      HIPDNN_ATTR_ENGINEHEUR_RESULTS,
      HIPDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &count,
      &engineCfgPtr);
  
  // Finalize engine config
  hipdnnBackend()->backendFinalize(engineCfgPtr);
  
  engineConfigDesc_ = std::move(engineCfgDesc);
}
```

---

## 6. File I/O Helpers

### 6.1 Loading Graph from File

**File:** `custom-op-hipdnn/src/custom_op.cpp`

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

**Key Points:**
- Opens file in binary mode
- Uses `ios::ate` to position at end for size determination
- Reads entire file into `std::vector<uint8_t>` buffer
- Exception-based error handling

### 6.2 Saving Graph to File

**File:** `level-1-pass-hipdnn/src/pass_main.cpp`

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

**Key Points:**
- Opens file in binary write mode
- Writes FlatBuffer data directly
- Checks file state after closing
- Exception-based error handling

---

## 7. UID Extraction and Management

### 7.1 Overview

UIDs (Unique Identifiers) are critical for mapping ONNX Runtime tensors to hipDNN graph tensors during execution. The custom op must extract UIDs from the serialized graph and maintain mappings for both inputs and outputs.

### 7.2 UID Extraction Implementation

```cpp
void HipdnnCustomOp::ExtractUIDsFromSerializedGraph(const std::vector<uint8_t>& buffer) {
  using namespace hipdnn_plugin_sdk;
  
  // Use GraphWrapper to read the serialized graph structure
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
    
    // Check node type and extract output UID
    if (node.attributes_type() == hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes) {
      auto* conv_attrs = node.attributes_as_ConvolutionFwdAttributes();
      if (conv_attrs) {
        outputUids.insert(conv_attrs->y_tensor_uid());
      }
    }
    // Add other node types as needed
  }
  
  // Classify non-virtual tensors as inputs or outputs
  std::vector<std::pair<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>> sortedTensors;
  for (const auto& [uid, tensor] : tensorMap) {
    if (!tensor->virtual_()) {
      sortedTensors.push_back({uid, tensor});
    }
  }
  
  // Sort by UID to maintain consistent ordering
  std::sort(sortedTensors.begin(), sortedTensors.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  
  // Now classify as inputs or outputs
  for (const auto& [uid, tensor] : sortedTensors) {
    if (outputUids.count(uid) > 0) {
      // This is a graph output
      output_uids_.push_back(uid);
      
      // Extract shape for output allocation
      auto dims = tensor->dims();
      if (dims) {
        std::vector<int64_t> shape(dims->begin(), dims->end());
        output_shapes_.push_back(shape);
      } else {
        throw std::runtime_error("Output tensor missing dimensions");
      }
    } else {
      // This is a graph input
      input_uids_.push_back(uid);
    }
  }
  
  MY_LOG(1) << "Extracted UIDs: " 
            << input_uids_.size() << " inputs, "
            << output_uids_.size() << " outputs";
}
```

### 7.3 Key Concepts

**GraphWrapper:**
- Utility class from `hipdnn_data_sdk`
- Reads FlatBuffer serialized graphs
- Provides access to tensor and node information
- API: `getTensorMap()`, `getNode()`, `nodeCount()`

**UID Classification:**
1. **Virtual vs Non-Virtual**: Only non-virtual tensors need UID mapping
2. **Input vs Output**: Determined by checking node output UIDs
3. **Ordering**: UIDs sorted to ensure consistent tensor ordering

**Why This Matters:**
- ONNX Runtime provides tensors in a specific order (inputs first, then outputs)
- We must map these ordered tensors to their corresponding UIDs
- Variant pack requires UID → pointer mapping for execution

### 7.4 Variant Pack Construction

During `Compute()`, UIDs are used to build the variant pack:

```cpp
void HipdnnCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  Ort::KernelContext ctx(context);
  
  // Build variant pack: UID → tensor pointer mapping
  std::unordered_map<int64_t, void*> variant_pack;
  
  // Map inputs (using extracted input_uids_)
  for (size_t i = 0; i < input_uids_.size(); ++i) {
    Ort::ConstValue input = ctx.GetInput(i);
    variant_pack[input_uids_[i]] = const_cast<void*>(input.GetTensorRawData());
  }
  
  // Allocate and map outputs (using extracted output_uids_ and output_shapes_)
  for (size_t i = 0; i < output_uids_.size(); ++i) {
    Ort::UnownedValue output = ctx.GetOutput(i, output_shapes_[i]);
    variant_pack[output_uids_[i]] = output.GetTensorMutableRawData();
  }
  
  // Convert to arrays for backend API
  std::vector<int64_t> keys;
  std::vector<void*> values;
  for (const auto& [key, value] : variant_pack) {
    keys.push_back(key);
    values.push_back(value);
  }
  
  // Create variant pack descriptor
  auto variantPackDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
      HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  
  // Set data pointers
  hipdnnBackend()->backendSetAttribute(
      variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
      HIPDNN_TYPE_VOID_PTR,
      static_cast<int64_t>(values.size()),
      values.data());
  
  // Set UIDs
  hipdnnBackend()->backendSetAttribute(
      variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
      HIPDNN_TYPE_INT64,
      static_cast<int64_t>(keys.size()),
      keys.data());
  
  // Set workspace
  void* workspace_ptr = workspace_.empty() ? nullptr : 
                        const_cast<void*>(static_cast<const void*>(workspace_.data()));
  hipdnnBackend()->backendSetAttribute(
      variantPackDesc->get(),
      HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
      HIPDNN_TYPE_VOID_PTR,
      1,
      &workspace_ptr);
  
  // Finalize and execute
  hipdnnBackend()->backendFinalize(variantPackDesc->get());
  hipdnnBackend()->backendExecute(handle_, executionPlanDesc_->get(), variantPackDesc->get());
}
```

---

## 8. Complete Code Examples

### 8.1 End-to-End Conv Example

**ONNX Model:**
```
Input [1, 3, 224, 224] (float32)
  ↓
Conv (kernel=3x3, stride=1, pad=1)
  ↓
Output [1, 64, 224, 224] (float32)
```

**Pass Processing:**

```cpp
// 1. Identify Conv node
if (node_op_type(*node) == "Conv") {
  
  // 2. Build hipDNN graph
  auto graph = std::make_unique<HipDNNGraph>();
  graph->set_name("Conv_output");
  
  // Input tensor (UID=1)
  auto x_attr = std::make_shared<TensorAttributes>();
  x_attr->set_uid(1)
      .set_data_type(DataType::FLOAT)
      .set_dim({1, 3, 224, 224})
      .set_stride({150528, 50176, 224, 1})
      .set_is_virtual(false);
  
  // Weight tensor (UID=2)
  auto w_attr = std::make_shared<TensorAttributes>();
  w_attr->set_uid(2)
      .set_data_type(DataType::FLOAT)
      .set_dim({64, 3, 3, 3})
      .set_stride({27, 9, 3, 1})
      .set_is_virtual(false);
  
  // Conv attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({1, 1})
      .set_stride({1, 1})
      .set_dilation({1, 1})
      .set_compute_data_type(DataType::FLOAT);
  
  // Add conv operation
  auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);
  y_attr->set_uid(3).set_is_virtual(false);
  
  // 3. Serialize
  auto buffer = graph->buildFlatbufferOperationGraph();
  SaveGraphToFile(buffer, "hipdnn_graph_output.bin");
  
  // 4. Create fused node
  hipdnn::HipdnnParamProto proto;
  proto.set_graph_file_name("hipdnn_graph_output.bin");
  // ... attach to meta_def and fuse
}
```

**Custom Op Execution:**

```cpp
// Constructor: Load and compile
HipdnnCustomOp::HipdnnCustomOp(...) {
  // Load graph file
  auto buffer = LoadGraphFromFile("hipdnn_graph_output.bin");
  
  // Create descriptors
  graphDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
      buffer.data(), buffer.size());
  // ... compile pipeline ...
  
  // Extract UIDs: input_uids_ = {1, 2}, output_uids_ = {3}
  ExtractUIDsFromSerializedGraph(buffer);
}

// Compute: Execute
void HipdnnCustomOp::Compute(...) const {
  // Build variant pack
  std::unordered_map<int64_t, void*> variant_pack;
  variant_pack[1] = input_0_ptr;   // Input data
  variant_pack[2] = input_1_ptr;   // Weight data
  variant_pack[3] = output_0_ptr;  // Output data
  
  // Execute
  hipdnnBackend()->backendExecute(handle_, executionPlanDesc_->get(), variantPackDesc->get());
}
```

### 8.2 Multi-Operation Graph Example

For more complex graphs with multiple operations:

```cpp
// Build graph with multiple operations
auto graph = std::make_unique<HipDNNGraph>();

// Conv → BatchNorm → ReLU
auto conv_out = graph->conv_fprop(input, weight, conv_attrs);
auto bn_out = graph->batch_norm(conv_out, scale, bias, bn_attrs);
auto relu_out = graph->relu(bn_out, relu_attrs);

// Mark final output as non-virtual
relu_out->set_is_virtual(false);

// Serialize entire graph
auto buffer = graph->buildFlatbufferOperationGraph();
```

The same serialization mechanism handles complex graphs seamlessly!

---

## 9. Testing Strategy

### 9.1 Unit Testing

**Test Graph Serialization:**
```cpp
TEST(HipDNNPass, SerializeSimpleConv) {
  // Create simple Conv node
  auto conv_node = CreateConvNode(/* params */);
  
  // Build and serialize
  std::string filename;
  bool success = BuildAndSerializeGraph(conv_node, input, weight, output, filename);
  
  ASSERT_TRUE(success);
  ASSERT_TRUE(std::filesystem::exists(filename));
  
  // Verify file size > 0
  auto size = std::filesystem::file_size(filename);
  ASSERT_GT(size, 0);
}
```

**Test Graph Loading:**
```cpp
TEST(HipDNNCustomOp, LoadSerializedGraph) {
  // Create test graph file
  CreateTestGraphFile("test_graph.bin");
  
  // Load in custom op
  auto buffer = LoadGraphFromFile("test_graph.bin");
  
  ASSERT_GT(buffer.size(), 0);
  
  // Verify can create descriptor
  auto graphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
      buffer.data(), buffer.size());
  
  ASSERT_TRUE(graphDesc->valid());
}
```

**Test UID Extraction:**
```cpp
TEST(HipDNNCustomOp, ExtractUIDs) {
  // Create graph with known UIDs
  auto buffer = CreateGraphWithUIDs({1, 2}, {3});  // 2 inputs, 1 output
  
  // Extract UIDs
  std::vector<int64_t> input_uids, output_uids;
  ExtractUIDs(buffer, input_uids, output_uids);
  
  ASSERT_EQ(input_uids.size(), 2);
  ASSERT_EQ(output_uids.size(), 1);
  EXPECT_EQ(input_uids[0], 1);
  EXPECT_EQ(input_uids[1], 2);
  EXPECT_EQ(output_uids[0], 3);
}
```

### 9.2 Integration Testing

**Test End-to-End Execution:**
```cpp
TEST(HipDNNIntegration, ExecuteConv) {
  // 1. Load ONNX model with Conv
  auto model = LoadONNXModel("conv_model.onnx");
  
  // 2. Run pass to fuse Conv
  RunLevel1Pass(model);
  
  // 3. Verify graph file created
  ASSERT_TRUE(std::filesystem::exists("hipdnn_graph_*.bin"));
  
  // 4. Create session and run inference
  auto session = CreateSession(model);
  auto output = session.Run(input_data);
  
  // 5. Verify output correctness
  CompareWithReference(output, expected_output, tolerance);
}
```

### 9.3 Performance Testing

```cpp
BENCHMARK(HipDNNExecution, Conv2D_224x224) {
  // Measure execution time
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 1000; ++i) {
    session.Run(input_data);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  LOG(INFO) << "Average time: " << duration.count() / 1000.0 << " μs";
}
```

---

## 10. Migration Path

### 10.1 From Parameter Passing to Graph Serialization

**Old Approach (Parameter Passing):**

```cpp
// Proto with all parameters
message HipdnnParamProto {
  repeated int64 pads = 1;
  repeated int64 strides = 2;
  repeated int64 dilations = 3;
  // ... many more fields
}

// Pass: Extract and pass parameters
hipdnn_param.set_pads(...);
hipdnn_param.set_strides(...);
// ...

// Custom Op: Rebuild graph from parameters
auto graph = BuildGraphFromParams(hipdnn_param);
```

**New Approach (Graph Serialization):**

```cpp
// Proto with only filename
message HipdnnParamProto {
  string graph_file_name = 1;
}

// Pass: Build and serialize graph
auto graph = BuildGraphFromONNX(node);
auto buffer = graph->buildFlatbufferOperationGraph();
SaveGraphToFile(buffer, filename);
hipdnn_param.set_graph_file_name(filename);

// Custom Op: Load pre-built graph
auto buffer = LoadGraphFromFile(hipdnn_param.graph_file_name());
graphDesc_ = std::make_unique<ScopedHipdnnBackendDescriptor>(
    buffer.data(), buffer.size());
```

### 10.2 Migration Steps

1. **Update Proto Definition**
   - Replace operation-specific parameters with `graph_file_name`
   - Regenerate proto bindings

2. **Modify Level-1 Pass**
   - Remove pattern matching logic
   - Add graph building logic using Frontend API
   - Add serialization and file saving
   - Pass filename instead of parameters

3. **Modify Custom Op**
   - Remove graph reconstruction from parameters
   - Add graph loading from file
   - Add backend descriptor compilation pipeline
   - Add UID extraction logic

4. **Update CMake**
   - Add dependencies: `hipdnn_frontend`, `hipdnn_data_sdk`
   - Link against hipDNN libraries

5. **Testing**
   - Verify graph files are created correctly
   - Verify execution produces correct results
   - Verify performance is maintained or improved

### 10.3 Benefits of Migration

| Aspect | Before | After |
|--------|--------|-------|
| **Complexity** | High (parameter passing + reconstruction) | Low (just load and compile) |
| **Scalability** | Add parameters for each op type | Same mechanism for all ops |
| **Debugging** | Hard (parameters in proto) | Easy (inspect graph files) |
| **Maintenance** | Two places (pass + custom op) | One place (pass only) |
| **Performance** | Overhead from reconstruction | Direct compilation |

### 10.4 Backward Compatibility

If you need to support both approaches temporarily:

```cpp
// Proto with version field
message HipdnnParamProto {
  int32 version = 1;
  
  // New approach (version 2)
  string graph_file_name = 2;
  
  // Old approach (version 1)
  repeated int64 pads = 10;
  repeated int64 strides = 11;
  // ...
}

// Custom Op: Check version
if (hipdnn_proto_.version() == 2) {
  // Use graph serialization
  LoadAndCompileGraph();
} else {
  // Use parameter passing (legacy)
  BuildGraphFromParams();
}
```

---

## Appendix A: Key References

- **hipDNN Frontend**: `c:/Develop/TheRock/include/hipdnn/frontend/hipdnn_frontend/Graph.hpp`
- **GraphWrapper**: `c:/Develop/TheRock/include/hipdnn/data_sdk/hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp`
- **hipDNNEP Kernel**: `external/hipDNNEP/src/kernel.cc`
- **hipDNNEP EP**: `external/hipDNNEP/src/ep.cc`

## Appendix B: Glossary

- **UID**: Unique Identifier for tensors in hipDNN graph
- **Variant Pack**: Mapping of UIDs to tensor data pointers for execution
- **Backend Descriptor**: Low-level hipDNN descriptor objects
- **ScopedHipdnnBackendDescriptor**: RAII wrapper for backend descriptors
- **GraphWrapper**: Utility to read FlatBuffer serialized graphs

---

**End of Document**
