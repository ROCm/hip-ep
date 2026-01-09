# hipDNN Graph API Guide

## Overview

This guide explains how to work with the hipDNN Frontend Graph API, covering graph construction, serialization, deserialization, and kernel execution. The guide is based on the current implementation in the hipDNNEP project.

## Table of Contents

1. [Building a hipDNN Graph](#1-building-a-hipdnn-graph)
2. [Serializing the Graph](#2-serializing-the-graph)
3. [Loading the Graph](#3-loading-the-graph)
4. [Building a Kernel from Graph](#4-building-a-kernel-from-graph)
5. [Complete Example](#5-complete-example)
6. [Limitations and Notes](#6-limitations-and-notes)

---

## 1. Building a hipDNN Graph

### 1.1 Graph Creation Basics

A hipDNN graph is constructed using the `hipdnn_frontend::graph::Graph` class. The graph represents a computational workflow with operations (nodes) and data flow (tensors).

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend::graph;

// Create a new graph
auto graph = std::make_unique<Graph>();
graph->set_name("MyConvGraph");
```

### 1.2 Creating Tensor Attributes

Tensors are the data that flows through the graph. Each tensor needs attributes describing its shape, data type, and layout.

```cpp
using hipdnn_frontend::DataType;

// Helper function to compute NCHW strides
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Create input tensor (e.g., 1x3x224x224 image in NCHW format)
auto x_attr = std::make_shared<TensorAttributes>();
std::vector<int64_t> x_shape = {1, 3, 224, 224};  // [N, C, H, W]
x_attr->set_uid(1)
       .set_name("input_image")
       .set_data_type(DataType::FLOAT)
       .set_dim(x_shape)
       .set_stride(ComputeStrides(x_shape))
       .set_is_virtual(false);  // Graph inputs must be non-virtual

// Create weight tensor (e.g., 64 output channels, 3 input channels, 3x3 kernel)
auto w_attr = std::make_shared<TensorAttributes>();
std::vector<int64_t> w_shape = {64, 3, 3, 3};  // [K, C, R, S]
w_attr->set_uid(2)
       .set_name("conv_weights")
       .set_data_type(DataType::FLOAT)
       .set_dim(w_shape)
       .set_stride(ComputeStrides(w_shape))
       .set_is_virtual(false);  // Weights are graph inputs
```

**Important Tensor Attributes:**
- `uid`: Unique identifier (must be unique across all tensors in the graph)
- `name`: Human-readable name for debugging
- `data_type`: FLOAT, HALF, etc.
- `dim`: Shape in NCHW format
- `stride`: Memory layout strides
- `is_virtual`: `false` for graph inputs/outputs, `true` for intermediate tensors

### 1.3 Adding Operations to the Graph

Operations are added to the graph by calling methods on the `Graph` object. Each operation takes input tensors and returns output tensors.

#### Convolution Forward Propagation

```cpp
using hipdnn_frontend::ConvolutionMode;

// Create convolution attributes
ConvFpropAttributes conv_attrs;
conv_attrs.set_padding({1, 1})           // [pad_h, pad_w]
          .set_stride({1, 1})            // [stride_h, stride_w]
          .set_dilation({1, 1})          // [dilation_h, dilation_w]
          .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
          .set_compute_data_type(DataType::FLOAT);

// Add convolution to graph
auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);

// The output tensor is automatically created as virtual
// Mark it as non-virtual if it's a graph output
y_attr->set_is_virtual(false);
```

#### Pointwise Operations

```cpp
// Add ReLU activation (pointwise operation)
PointwiseAttributes relu_attrs;
relu_attrs.set_mode(PointwiseMode::RELU_FWD);

auto relu_out = graph->pointwise(y_attr, relu_attrs);
relu_out->set_is_virtual(false);  // Mark as graph output
```

### 1.4 Graph Input/Output Management

**Graph Inputs** (non-virtual tensors that aren't outputs of any operation):
- Must have `is_virtual(false)`
- Typically: input data, weights, biases
- UIDs will be used to map actual data pointers during execution

**Graph Outputs** (tensors that should be materialized):
- Must have `is_virtual(false)`
- Intermediate tensors can remain `is_virtual(true)` for optimization

### 1.5 UID Management

Each tensor must have a unique UID. The current implementation in `kernel.cc` uses:

```cpp
int64_t next_uid_ = 1;  // Start from 1

// Assign UIDs to tensors
tensor->set_uid(next_uid_++);
```

**Best Practice:** Track UIDs to ensure uniqueness across the graph.

---

## 2. Serializing the Graph

### 2.1 What Gets Serialized

The `buildFlatbufferOperationGraph()` method serializes:
- ✅ All tensor attributes (shape, dtype, strides, names, UIDs)
- ✅ All operation nodes (type, attributes, connections)
- ✅ Graph attributes (name, compute dtype, etc.)
- ❌ **NOT** the compiled execution plan
- ❌ **NOT** the engine configuration

### 2.2 Serialization Process

```cpp
// Build the FlatBuffer representation
flatbuffers::DetachedBuffer serialized = graph->buildFlatbufferOperationGraph();

// Save to file
#include <fstream>

void SaveGraphToFile(const flatbuffers::DetachedBuffer& buffer, 
                     const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing");
    }
    
    file.write(reinterpret_cast<const char*>(buffer.data()), 
               buffer.size());
    file.close();
    
    if (!file.good()) {
        throw std::runtime_error("Failed to write graph to file");
    }
}

// Usage
SaveGraphToFile(serialized, "conv_graph.bin");
```

### 2.3 When to Serialize

Serialize the graph **after** building the graph structure but **before** compilation:

```cpp
// 1. Build graph structure
auto graph = std::make_unique<Graph>();
// ... add operations ...

// 2. Serialize (OPTIONAL)
auto serialized = graph->buildFlatbufferOperationGraph();
SaveGraphToFile(serialized, "graph.bin");

// 3. Continue with compilation
graph->validate();
graph->build_operation_graph(handle);
// ... etc.
```

---

## 3. Loading the Graph

### 3.1 Deserialization Process

Loading a serialized graph requires reconstructing the backend descriptor:

```cpp
#include <fstream>
#include <vector>

std::vector<uint8_t> LoadGraphFromFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading");
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read graph from file");
    }
    
    return buffer;
}
```

### 3.2 Creating Graph Descriptor from Serialized Data

```cpp
#include <hipdnn_frontend/backend/ScopedHipdnnBackendDescriptor.hpp>

// Load serialized data
auto buffer = LoadGraphFromFile("graph.bin");

// Create backend descriptor from serialized data
auto graphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
    buffer.data(), 
    buffer.size()
);

if (!graphDesc->valid()) {
    throw std::runtime_error("Failed to create graph descriptor from data");
}
```

### 3.3 Required Post-Load Setup

After loading, you must:

1. **Set the hipDNN handle**
2. **Finalize the descriptor**
3. **Continue with compilation pipeline**

```cpp
// Set handle
hipdnnHandle_t handle;
hipdnnCreate(&handle);

hipdnnStatus_t status = hipdnnBackend()->backendSetAttribute(
    graphDesc->get(),
    HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
    HIPDNN_TYPE_HANDLE,
    1,
    &handle
);

if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to set handle on graph");
}

// Finalize
status = hipdnnBackend()->backendFinalize(graphDesc->get());
if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to finalize graph descriptor");
}
```

### 3.4 Important Limitation

**Loaded graphs still require full compilation!** The serialization only saves the graph structure, not the compiled execution plan.

---

## 4. Building a Kernel from Graph

### 4.1 Compilation Pipeline Overview

The kernel building process follows these stages:

```
Graph Structure
    ↓
validate() ─────────────→ Validate topology, check for cycles
    ↓
build_operation_graph() → Create backend descriptor
    ↓
create_execution_plans() → Generate engine configurations
    ↓
check_support() ─────────→ Verify device support
    ↓
build_plans() ───────────→ Finalize execution plan
    ↓
execute() ───────────────→ Run the compiled kernel
```

### 4.2 Step-by-Step Kernel Building

Based on the implementation in `kernel.cc`:

```cpp
class Kernel {
public:
    Kernel(const OrtApi& ort_api, const OrtLogger& logger, hipdnnHandle_t handle)
        : ort_api_(ort_api), logger_(logger), handle_(handle) {}
    
    OrtStatus* BuildAndCompile(Ort::ConstGraph ort_graph) {
        try {
            // Create hipDNN graph
            graph_ = std::make_unique<hipdnn_frontend::graph::Graph>();
            
            // Build graph structure from ORT graph
            RETURN_IF_ERROR(BuildGraphStructure(ort_graph));
            
            // Compile the graph
            RETURN_IF_ERROR(CompileGraph());
            
        } catch (const std::exception& ex) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "Exception building hipDNN graph: " << ex.what());
        }
        return nullptr;
    }

private:
    OrtStatus* CompileGraph() {
        using hipdnn_frontend::HeuristicMode;
        
        // Step 1: Validate graph
        auto error = graph_->validate();
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "Graph validation failed: " << error.get_message());
        }
        
        // Step 2: Build operation graph
        error = graph_->build_operation_graph(handle_);
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "build_operation_graph failed: " << error.get_message());
        }
        
        // Step 3: Create execution plans
        error = graph_->create_execution_plans({HeuristicMode::FALLBACK});
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "create_execution_plans failed: " << error.get_message());
        }
        
        // Step 4: Check device support
        error = graph_->check_support();
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "check_support failed: " << error.get_message());
        }
        
        // Step 5: Build execution plans
        error = graph_->build_plans();
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "build_plans failed: " << error.get_message());
        }
        
        // Step 6: Get workspace size
        int64_t workspace_size = 0;
        error = graph_->get_workspace_size(workspace_size);
        if (error.is_bad()) {
            RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                        "get_workspace_size failed: " << error.get_message());
        }
        
        // Allocate workspace if needed
        if (workspace_size > 0) {
            workspace_.resize(workspace_size);
        }
        
        return nullptr;
    }
    
    hipdnnHandle_t handle_;
    std::unique_ptr<hipdnn_frontend::graph::Graph> graph_;
    std::vector<char> workspace_;
    std::vector<int64_t> input_uids_;
    std::vector<int64_t> output_uids_;
};
```

### 4.3 Heuristic Modes

The `create_execution_plans()` method accepts heuristic modes:

```cpp
using hipdnn_frontend::HeuristicMode;

// Available modes:
HeuristicMode::FALLBACK    // Default fallback implementation
// More modes may be added in future hipDNN versions
```

### 4.4 Workspace Management

Some operations require temporary workspace memory:

```cpp
// Query workspace size after building plans
int64_t workspace_size = 0;
auto error = graph_->get_workspace_size(workspace_size);

// Allocate workspace
std::vector<char> workspace;
if (workspace_size > 0) {
    workspace.resize(workspace_size);
}
```

### 4.5 Execution

Execute the compiled kernel with a variant pack mapping UIDs to data pointers:

```cpp
OrtStatus* Execute(OrtKernelContext* kernel_ctx) {
    Ort::KernelContext context(kernel_ctx);
    
    // Build variant pack: UID → data pointer
    std::unordered_map<int64_t, void*> variant_pack;
    
    // Map inputs
    for (size_t i = 0; i < input_uids_.size(); ++i) {
        Ort::ConstValue input = context.GetInput(i);
        variant_pack[input_uids_[i]] = 
            const_cast<void*>(input.GetTensorRawData());
    }
    
    // Allocate and map outputs
    for (size_t i = 0; i < output_uids_.size(); ++i) {
        Ort::UnownedValue output = context.GetOutput(i, output_shapes_[i]);
        variant_pack[output_uids_[i]] = 
            output.GetTensorMutableRawData();
    }
    
    // Execute
    void* workspace_ptr = workspace_.empty() ? nullptr : workspace_.data();
    auto error = graph_->execute(handle_, variant_pack, workspace_ptr);
    
    if (error.is_bad()) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, 
                    "Execute failed: " << error.get_message());
    }
    
    return nullptr;
}
```

---

## 5. Complete Example

### 5.1 End-to-End Workflow

```cpp
#include <hipdnn_frontend.hpp>
#include <hipdnn_backend.h>

using namespace hipdnn_frontend::graph;
using hipdnn_frontend::DataType;
using hipdnn_frontend::ConvolutionMode;

class ConvKernel {
public:
    ConvKernel() {
        hipdnnCreate(&handle_);
    }
    
    ~ConvKernel() {
        if (handle_) {
            hipdnnDestroy(handle_);
        }
    }
    
    void BuildGraph() {
        graph_ = std::make_unique<Graph>();
        graph_->set_name("SimpleConv2D");
        
        // Input: 1x3x224x224
        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1)
         .set_name("input")
         .set_data_type(DataType::FLOAT)
         .set_dim({1, 3, 224, 224})
         .set_stride({150528, 50176, 224, 1})
         .set_is_virtual(false);
        input_uids_.push_back(1);
        
        // Weights: 64x3x3x3
        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(2)
         .set_name("weights")
         .set_data_type(DataType::FLOAT)
         .set_dim({64, 3, 3, 3})
         .set_stride({27, 9, 3, 1})
         .set_is_virtual(false);
        input_uids_.push_back(2);
        
        // Convolution
        ConvFpropAttributes conv_attrs;
        conv_attrs.set_padding({1, 1})
                  .set_stride({1, 1})
                  .set_dilation({1, 1})
                  .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
                  .set_compute_data_type(DataType::FLOAT);
        
        auto y = graph_->conv_fprop(x, w, conv_attrs);
        y->set_is_virtual(false)
         .set_name("output")
         .set_data_type(DataType::FLOAT)
         .set_dim({1, 64, 224, 224})
         .set_stride({3211264, 50176, 224, 1});
        output_uids_.push_back(y->get_uid());
        output_shapes_.push_back({1, 64, 224, 224});
    }
    
    void Compile() {
        using hipdnn_frontend::HeuristicMode;
        
        auto error = graph_->validate();
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        error = graph_->build_operation_graph(handle_);
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        error = graph_->create_execution_plans({HeuristicMode::FALLBACK});
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        error = graph_->check_support();
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        error = graph_->build_plans();
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        int64_t workspace_size = 0;
        error = graph_->get_workspace_size(workspace_size);
        if (error.is_bad()) throw std::runtime_error(error.get_message());
        
        if (workspace_size > 0) {
            workspace_.resize(workspace_size);
        }
    }
    
    void Execute(void* input_data, void* weight_data, void* output_data) {
        std::unordered_map<int64_t, void*> variant_pack;
        variant_pack[input_uids_[0]] = input_data;   // Input
        variant_pack[input_uids_[1]] = weight_data;  // Weights
        variant_pack[output_uids_[0]] = output_data; // Output
        
        void* workspace_ptr = workspace_.empty() ? nullptr : workspace_.data();
        auto error = graph_->execute(handle_, variant_pack, workspace_ptr);
        if (error.is_bad()) {
            throw std::runtime_error(error.get_message());
        }
    }

private:
    hipdnnHandle_t handle_ = nullptr;
    std::unique_ptr<Graph> graph_;
    std::vector<char> workspace_;
    std::vector<int64_t> input_uids_;
    std::vector<int64_t> output_uids_;
    std::vector<std::vector<int64_t>> output_shapes_;
};

// Usage
int main() {
    ConvKernel kernel;
    
    // Build and compile
    kernel.BuildGraph();
    kernel.Compile();
    
    // Allocate device memory (pseudo-code)
    void* d_input = allocate_device_memory(1 * 3 * 224 * 224 * sizeof(float));
    void* d_weights = allocate_device_memory(64 * 3 * 3 * 3 * sizeof(float));
    void* d_output = allocate_device_memory(1 * 64 * 224 * 224 * sizeof(float));
    
    // Execute
    kernel.Execute(d_input, d_weights, d_output);
    
    return 0;
}
```

---

## 6. Limitations and Notes

### 6.1 Serialization Limitations

| Feature | Supported | Notes |
|---------|-----------|-------|
| Graph structure | ✅ Yes | Via FlatBuffers |
| Tensor attributes | ✅ Yes | Shapes, dtypes, UIDs |
| Operation attributes | ✅ Yes | Conv params, etc. |
| Compiled execution plan | ❌ No | Must recompile after loading |
| Engine configuration | ❌ No | Selected during compilation |
| Workspace size | ❌ No | Computed during compilation |

### 6.2 Key Differences vs. Full Artifact Caching

**Current Serialization (Graph Structure Only):**
- Saves: Graph topology and attributes
- Benefits: Avoid rebuilding graph programmatically
- Limitation: Still requires full compilation

**True Artifact Caching (Not Available):**
- Would save: Compiled execution plan
- Would benefit: Skip compilation entirely
- Would require: Backend API extensions

### 6.3 Best Practices

1. **UID Management**: Ensure all tensors have unique UIDs
2. **Virtual Tensors**: Mark only inputs/outputs as non-virtual
3. **Topology**: Validate graph before compilation
4. **Error Handling**: Check all `Error` return values
5. **Workspace**: Allocate workspace after `build_plans()`
6. **Handle Lifetime**: Keep `hipdnnHandle_t` alive during execution

### 6.4 Performance Considerations

- **Compilation Time**: Can be significant; consider caching at application level
- **Workspace Size**: May be large; allocate once and reuse
- **Graph Complexity**: More operations = longer compilation time

### 6.5 Future Enhancements

Potential improvements to consider:

1. **Execution Plan Serialization**: Add backend support for plan caching
2. **AOT Compilation**: Pre-compile graphs during build time
3. **ORT EP Context Integration**: Use ORT's built-in artifact management
4. **Cache Validation**: Hash-based cache invalidation

---

## References

- hipDNN Frontend API: `C:\Develop\TheRock\include\hipdnn\frontend\hipdnn_frontend.hpp`
- Graph Implementation: `C:\Develop\TheRock\include\hipdnn\frontend\hipdnn_frontend\Graph.hpp`
- Current Kernel Implementation: `src/kernel.cc`
- Current EP Implementation: `src/ep.cc`

---

**Document Version**: 1.0  
**Last Updated**: January 9, 2026  
**Author**: hipDNNEP Development Team
