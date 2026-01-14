# hipDNN Constant Data Handling Guide

## Overview

This document explains how hipDNN handles constant data (such as convolution weights and biases) in the morphizen-hipdnn integration. The implementation separates constant tensors from runtime inputs, enabling efficient pre-loading of weights and avoiding redundant data processing during inference.

## Table of Contents

1. [hipDNN Graph Basics](#1-hipdnn-graph-basics)
2. [Tensor Types in hipDNN](#2-tensor-types-in-hipdnn)
3. [Constant Data Workflow](#3-constant-data-workflow)
4. [Implementation Details](#4-implementation-details)
5. [UID and Variant Pack](#5-uid-and-variant-pack)
6. [Key Concepts Summary](#6-key-concepts-summary)

---

## 1. hipDNN Graph Basics

### What is a hipDNN Graph?

A hipDNN graph is a computational graph that represents neural network operations. Unlike immediate execution, hipDNN uses a **graph-based API** where:

1. **Build Phase**: You construct a graph describing all operations and their data flow
2. **Compile Phase**: hipDNN optimizes and compiles the graph for execution
3. **Execute Phase**: You run the compiled graph with actual data

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         hipDNN Graph Lifecycle                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Build Graph         Compile Graph           Execute Graph              │
│   ┌─────────┐         ┌─────────────┐         ┌───────────────────┐     │
│   │ Define  │         │ Validate    │         │ Provide Data      │     │
│   │ Tensors │  ───►   │ Optimize    │  ───►   │ (via UID mapping) │     │
│   │ & Ops   │         │ Build Plans │         │ Run Kernels       │     │
│   └─────────┘         └─────────────┘         └───────────────────┘     │
│                                                                          │
│   One-time setup      One-time setup          Repeated many times        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Difference from cuDNN-style APIs

Traditional DNN libraries often have explicit function calls like:
```cpp
// Traditional style (per-operation call)
cudnnConvolutionForward(handle, alpha, inputDesc, inputData, 
                        filterDesc, filterData, convDesc, ...);
```

hipDNN Graph API instead:
```cpp
// hipDNN Graph style
// 1. Build graph with tensor descriptors (no data yet!)
auto y = graph->conv_fprop(x_tensor, w_tensor, conv_attrs);

// 2. Compile once
graph->build_operation_graph(handle);
graph->create_execution_plans(...);
graph->build_plans();

// 3. Execute with data pointers mapped via UIDs
std::unordered_map<int64_t, void*> variant_pack;
variant_pack[x_tensor_uid] = input_data_ptr;
variant_pack[w_tensor_uid] = weight_data_ptr;
variant_pack[y_tensor_uid] = output_data_ptr;
graph->execute(handle, variant_pack, workspace);
```

---

## 2. Tensor Types in hipDNN

### 2.1 Graph Inputs vs Outputs

In a hipDNN graph, tensors are classified as:

| Type | Description | `is_virtual` |
|------|-------------|--------------|
| **Graph Input** | Tensor not produced by any operation | `false` |
| **Graph Output** | Tensor consumed by the external world | `false` |
| **Intermediate** | Tensor passed between operations | `true` (optional) |

### 2.2 Runtime Inputs vs Constants

From **ONNX Runtime's perspective**, graph inputs can be:

| Type | Description | When Data is Available |
|------|-------------|------------------------|
| **Runtime Input** | Changes every inference (e.g., input image) | At `Compute()` call |
| **Constant Initializer** | Fixed throughout model lifetime (e.g., weights) | At model load time |

**Key Insight**: hipDNN's graph API doesn't distinguish between runtime inputs and constants—they're all just "graph inputs" with UIDs. The distinction is handled at the **integration layer**.

### 2.3 Visual Example: Convolution

```
                      hipDNN Graph View
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Input Tensor      Weight Tensor                           │
│   UID=1             UID=2                                   │
│   is_virtual=false  is_virtual=false                        │
│        │                  │                                 │
│        └──────────┬───────┘                                 │
│                   │                                         │
│                   ▼                                         │
│           ┌──────────────┐                                  │
│           │  Conv2D Op   │                                  │
│           └──────────────┘                                  │
│                   │                                         │
│                   ▼                                         │
│            Output Tensor                                    │
│            UID=3                                            │
│            is_virtual=false                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                    Integration View
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Runtime Input      Constant (Weight)                      │
│   UID=1              UID=2                                  │
│   Source: ORT ctx    Source: Pre-loaded file                │
│        │                  │                                 │
│        └──────────────────┴─────────────────────┐           │
│                                                 │           │
│                                                 ▼           │
│                              variant_pack[UID] = data_ptr   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Constant Data Workflow

### 3.1 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Constant Data Flow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMPILE TIME (Level-1 Pass)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  1. Pattern Match finds Conv node                                   │    │
│  │           ↓                                                         │    │
│  │  2. Check: Is weight a constant initializer?                        │    │
│  │           ↓ Yes                                                     │    │
│  │  3. Build hipDNN graph with both tensors (UID=1, UID=2)            │    │
│  │           ↓                                                         │    │
│  │  4. Serialize graph to .bin file                                    │    │
│  │           ↓                                                         │    │
│  │  5. Extract weight data, save to .weight0.bin file                  │    │
│  │           ↓                                                         │    │
│  │  6. Create meta_def with:                                           │    │
│  │      - inputs: [input_tensor_name]          (runtime only)          │    │
│  │      - constant_initializers: [weight_name] (constants separately)  │    │
│  │           ↓                                                         │    │
│  │  7. Create proto with:                                              │    │
│  │      - graph_file_name: "xxx.bin"                                   │    │
│  │      - constant_names: ["weight_name"]                              │    │
│  │      - constant_data_files: ["xxx.weight0.bin"]                     │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  LOAD TIME (Custom Op Constructor)                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  1. Parse proto, get graph_file and constant_data_files             │    │
│  │           ↓                                                         │    │
│  │  2. Load graph from .bin file                                       │    │
│  │           ↓                                                         │    │
│  │  3. Compile: create descriptors, execution plans                    │    │
│  │           ↓                                                         │    │
│  │  4. Extract UIDs from serialized graph                              │    │
│  │      - input_uids_ = [1, 2]   (all graph inputs)                    │    │
│  │      - Identify last N as constant UIDs                             │    │
│  │           ↓                                                         │    │
│  │  5. Load constant data from .weight0.bin into memory                │    │
│  │      - constant_data_[0] = file contents                            │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  RUN TIME (Compute)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │  1. Get runtime inputs from ORT context                             │    │
│  │      - num_inputs = 1 (just the input image)                        │    │
│  │           ↓                                                         │    │
│  │  2. Build variant_pack:                                             │    │
│  │      - variant_pack[1] = ctx.GetInput(0) (runtime input)            │    │
│  │      - variant_pack[2] = constant_data_[0].data() (pre-loaded)      │    │
│  │           ↓                                                         │    │
│  │  3. Execute graph with variant_pack                                 │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Files Generated

For a single convolution operation, the compilation generates:

```
cache_dir/
├── hipdnn_graph_xxx.bin         # Serialized hipDNN graph structure
└── hipdnn_graph_xxx.weight0.bin # Raw weight data bytes
```

---

## 4. Implementation Details

### 4.1 Level-1 Pass: Saving Constants

```cpp
// level-1-pass-hipdnn/src/pass_main.cpp

// Check if weight is a constant initializer
bool weight_is_constant = weight_data.is_constant();

// Create meta_def with separate inputs and constants
std::vector<std::string> inputs_list = {input_data.name()};      // Runtime only
std::vector<std::string> constants_list = {weight_data.name()};  // Constants

auto [meta_def, fuse_error] = self_.try_fuse(
    ort_graph, unique_id,
    inputs_list,           // Runtime inputs
    {output_data.name()},  // Outputs
    constants_list,        // Constant initializers
    "HIPDNN"
);

// Save weight data to file
auto& weight_tensor = node_arg_get_const_data_as_tensor(ort_graph, weight_data);
auto weight_raw = vaip_core::api()->tensor_proto_as_raw(ort_graph, weight_tensor);

std::string weight_filename = graph_filename + ".weight0.bin";
std::ofstream weight_file(weight_filename, std::ios::binary);
weight_file.write(weight_raw.data(), weight_raw.size());

// Record in proto
hipdnn_param.add_constant_names(weight_data.name());
hipdnn_param.add_constant_data_files(weight_filename);
```

### 4.2 Custom Op: Loading Constants

```cpp
// custom-op-hipdnn/src/custom_op.cpp

// In constructor: get constant names from meta_def
auto constant_names = meta_def_->constant_initializers();
for (const auto& name : constant_names) {
    constant_initializer_names_.push_back(name);
}

// After graph loading: load constant data from files
void HipdnnCustomOp::LoadConstantData(onnxruntime::Model* model) {
    for (int i = 0; i < hipdnn_proto_.constant_data_files_size(); ++i) {
        const auto& data_file = hipdnn_proto_.constant_data_files(i);
        
        // Read binary file into memory
        std::ifstream file(data_file, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> data(size);
        file.read(data.data(), size);
        
        constant_data_.push_back(std::move(data));
    }
}
```

### 4.3 Custom Op: Execution with Constants

```cpp
// custom-op-hipdnn/src/custom_op.cpp

void HipdnnCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
    Ort::KernelContext ctx(context);
    auto num_inputs = ctx.GetInputCount();    // Runtime inputs only
    auto num_constants = constant_initializer_names_.size();
    
    // Validate: runtime + constants = graph inputs
    if (num_inputs + num_constants != input_uids_.size()) {
        // Error: mismatch
    }
    
    std::unordered_map<int64_t, void*> variant_pack;
    
    // Map runtime inputs (first N UIDs)
    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::ConstValue input = ctx.GetInput(i);
        variant_pack[input_uids_[i]] = const_cast<void*>(input.GetTensorRawData());
    }
    
    // Map constants (remaining UIDs, from pre-loaded data)
    for (size_t i = 0; i < num_constants; ++i) {
        size_t uid_idx = num_inputs + i;
        int64_t uid = input_uids_[uid_idx];
        void* data_ptr = const_cast<void*>(static_cast<const void*>(constant_data_[i].data()));
        variant_pack[uid] = data_ptr;
    }
    
    // Execute with variant_pack
    graph->execute(handle_, variant_pack, workspace);
}
```

---

## 5. UID and Variant Pack

### 5.1 What is a UID?

A **UID (Unique Identifier)** is an integer assigned to each tensor in a hipDNN graph. It serves as the key for mapping actual data pointers during execution.

```cpp
// When building the graph
auto x_attr = std::make_shared<TensorAttributes>();
x_attr->set_uid(1);  // UID for input tensor

auto w_attr = std::make_shared<TensorAttributes>();
w_attr->set_uid(2);  // UID for weight tensor
```

### 5.2 What is a Variant Pack?

A **Variant Pack** is hipDNN's mechanism for binding data to tensors at execution time. It's essentially a `map<UID, void*>`.

```cpp
// At execution time
std::unordered_map<int64_t, void*> variant_pack;
variant_pack[1] = input_data_ptr;   // UID 1 → actual input data
variant_pack[2] = weight_data_ptr;  // UID 2 → actual weight data
variant_pack[3] = output_data_ptr;  // UID 3 → output buffer

// Pass to hipDNN
auto variantPackDesc = create_variant_pack_descriptor(variant_pack);
hipdnnBackend()->backendExecute(handle, executionPlanDesc, variantPackDesc);
```

### 5.3 UID Ordering Convention

The implementation assumes UIDs are ordered as:

```
input_uids_ = [runtime_input_UIDs..., constant_UIDs...]

Example for Conv with 1 input + 1 weight:
input_uids_ = [1, 2]
                ↑  ↑
                │  └── Constant (weight), index = num_inputs + 0
                └───── Runtime input, index = 0
```

This ordering is established during graph construction and must be consistent between build and execution.

---

## 6. Key Concepts Summary

### 6.1 Why Separate Constants from Runtime Inputs?

| Aspect | Without Separation | With Separation |
|--------|-------------------|-----------------|
| **ONNX Runtime contract** | All inputs via GetInput() | Only dynamic inputs via GetInput() |
| **Weight loading** | ORT provides each call | Pre-loaded once at init |
| **Memory management** | ORT controlled | Custom op controlled |
| **Performance** | Re-process metadata each call | Zero overhead at runtime |

### 6.2 Key Data Structures

```cpp
class HipdnnCustomOp {
    // Proto metadata
    HipdnnParamProto hipdnn_proto_;  // Contains constant_names[], constant_data_files[]
    
    // UID mappings
    std::vector<int64_t> input_uids_;           // All graph inputs [runtime + constants]
    std::vector<int64_t> constant_input_uids_;  // Just constant UIDs
    
    // Pre-loaded constant data
    std::vector<std::string> constant_initializer_names_;  // Names from meta_def
    std::vector<std::vector<char>> constant_data_;         // Raw bytes from files
};
```

### 6.3 Execution Flow Summary

```
                          ┌──────────────────────────────┐
                          │  Compute() called            │
                          └──────────────────────────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
                   Get runtime inputs          Use pre-loaded constants
                   from ORT context            from constant_data_[]
                          │                              │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  Build variant_pack:         │
                          │  UID → data pointer map      │
                          └──────────────────────────────┘
                                         │
                          ┌──────────────▼───────────────┐
                          │  hipDNN execute()            │
                          └──────────────────────────────┘
```

---

## References

- Git Commit: `cba7c33 "Add support for loading constant initializers in HipdnnCustomOp"`
- hipDNN Graph API Guide: `doc/04_Graph_API_Guide.md`
- Custom Op Implementation: `custom-op-hipdnn/src/custom_op.cpp`
- Level-1 Pass Implementation: `level-1-pass-hipdnn/src/pass_main.cpp`
- Proto Definition: `proto/hipdnn.proto`

---

**Document Version**: 1.0  
**Last Updated**: January 13, 2026  
**Author**: morphizen-hipdnn Development Team
