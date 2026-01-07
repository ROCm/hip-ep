# Conv Node Support Implementation

This document describes the implementation of Conv (Convolution) node support in the morphizen-hipdnn project, following the reference implementation from [hipDNNEP](https://github.com/MaheshRavishankar/hipDNNEP).

## Overview

The implementation adds Conv operation support across three main components:
1. **Pattern Definition** - Recognizes Conv nodes in ONNX graphs
2. **Pass Implementation** - Extracts Conv attributes and creates fusion metadata
3. **Custom Op Execution** - Executes Conv operations using hipDNN

## Files Modified

### 1. `proto/hipdnn.proto`
Cleaned protocol buffer with only essential parameters:

```protobuf
message HipdnnParamProto {
  // HIP DNN specific parameters
  string device_id = 1;
  string kernel_type = 2;
  
  // Conv operation parameters
  string op_type = 3;
  repeated int64 pads = 4;
  repeated int64 strides = 5;
  repeated int64 dilations = 6;
  int64 group = 7;
}
```

### 2. `patterns/hipdnn.json`
Pattern already configured to match Conv operations:

```json
{
  "pattern": {
    "version": "1.0",
    "name": "hipdnn",
    "description": "HIP DNN operation pattern",
    "nodes": [
      {"id": "input", "op_type": "Input"},
      {"id": "hipdnn_op", "op_type": "Conv", "inputs": ["input"]},
      {"id": "output", "op_type": "Output", "inputs": ["hipdnn_op"]}
    ]
  }
}
```

### 3. `level-1-pass-hipdnn/src/pass_main.cpp`
Enhanced to extract Conv attributes during pattern matching:

**Key Changes:**
- Captures the Conv node from the binder: `auto conv_node = binder["hipdnn_op"].node;`
- Sets kernel type to "conv": `hipdnn_param.set_kernel_type("conv");`
- Extracts Conv-specific ONNX attributes:
  - `pads` - Padding for height and width dimensions
  - `strides` - Convolution stride
  - `dilations` - Dilation rate
  - `group` - Number of groups for grouped convolutions

```cpp
// Extract Conv attributes from the node
hipdnn_param.set_op_type(node_get_op_type(*conv_node));
auto pads_attr = node_get_attr_ints(*conv_node, "pads");
if (pads_attr.has_value()) {
  for (auto pad : pads_attr.value()) {
    hipdnn_param.add_pads(pad);
  }
}
// ... similar for strides, dilations, group
```

### 4. `custom-op-hipdnn/src/custom_op.cpp`
Updated Compute method to handle Conv operations:

**Key Changes:**
- Logs Conv operation parameters for debugging
- Differentiates between Conv and other operations
- Provides structure for hipDNN Conv execution (placeholder implementation)

```cpp
if (hipdnn_proto_.op_type() == "Conv") {
  MY_LOG(1) << "Conv operation detected";
  // Log pads, strides, dilations, group
  
  // TODO: Implement actual hipDNN Conv execution
  // 1. Create hipDNN graph with Conv operation using captured parameters
  // 2. Build and compile the graph
  // 3. Execute with variant pack mapping tensors to device memory
  
  // Placeholder implementation
  auto in_base = ctx.GetInput(idx).GetTensorData<float>();
  auto out_base = output_tensor.GetTensorMutableData<float>();
  for(auto i = 0; i < element_num; ++i) {
    out_base[i] = in_base[i];
  }
}
```

## Reference Implementation

The implementation follows the pattern from the hipDNNEP reference project:

### From `kernel.cc`:
```cpp
OrtStatus* AddConvNode(
    const OrtApi& ort_api,
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  // Extract Conv attributes
  std::vector<int64_t> pads = GetIntsAttrOrDefault(node, "pads", {0, 0, 0, 0});
  std::vector<int64_t> strides = GetIntsAttrOrDefault(node, "strides", {1, 1});
  std::vector<int64_t> dilations = GetIntsAttrOrDefault(node, "dilations", {1, 1});
  
  // Create convolution attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads[0], pads[1]})
      .set_stride({strides[0], strides[1]})
      .set_dilation({dilations[0], dilations[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());
  
  // Add convolution to graph
  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);
}
```

## Implementation Status

### ✅ Completed
- Pattern recognition for Conv operations
- Attribute extraction (pads, strides, dilations, group)
- Protocol buffer schema for Conv parameters
- Custom op structure for Conv execution
- Logging and debugging support

### 🚧 TODO (Future Work)
The current implementation provides the infrastructure for Conv support. To complete the implementation:

1. **Integrate hipDNN Library:**
   - Add hipDNN frontend API calls in `custom_op.cpp`
   - Create `hipdnn_frontend::graph::Graph` instance
   - Build Conv operation with extracted attributes

2. **Implement Actual Execution:**
   ```cpp
   // In HipdnnCustomOp::Compute()
   if (hipdnn_proto_.op_type() == "Conv") {
     // Create hipDNN graph
     auto graph = std::make_unique<hipdnn_frontend::graph::Graph>();
     
     // Create tensor attributes for inputs
     auto x_attr = CreateTensorAttr(input_tensor_0);
     auto w_attr = CreateTensorAttr(input_tensor_1);
     
     // Create Conv operation
     ConvFpropAttributes conv_attrs;
     conv_attrs.set_padding({pads[0], pads[1]})
               .set_stride({strides[0], strides[1]})
               .set_dilation({dilations[0], dilations[1]});
     
     auto y_attr = graph->conv_fprop(x_attr, w_attr, conv_attrs);
     
     // Build, compile, and execute
     graph->validate();
     graph->build_operation_graph(handle);
     graph->create_execution_plans({HeuristicMode::FALLBACK});
     graph->build_plans();
     graph->execute(handle, variant_pack, workspace);
   }
   ```

3. **Memory Management:**
   - Handle HIP device memory allocation
   - Implement CPU to GPU data transfer
   - Manage workspace memory for Conv operation

4. **Testing:**
   - Create test cases with various Conv configurations
   - Validate against reference implementations
   - Performance benchmarking

## Conv Attributes

### Pads
- Format: `[pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]` or `[pad_h, pad_w]`
- Default: `[0, 0, 0, 0]`
- Controls padding added to input tensor

### Strides
- Format: `[stride_h, stride_w]`
- Default: `[1, 1]`
- Controls the stride of the convolution

### Dilations
- Format: `[dilation_h, dilation_w]`
- Default: `[1, 1]`
- Controls the spacing between kernel elements

### Group
- Format: Single integer value
- Default: 1
- Number of groups for grouped convolution

## Building the Project

After these changes, rebuild the project:

```bash
# Configure
cmake --preset RelWithDebInfo

# Build
cmake --build --preset RelWithDebInfo

# Test
ctest --preset RelWithDebInfo
```

## Debugging

Enable debug logging to see Conv parameters:

```bash
export MORPHIZEN_DEBUG_HIPDNN=1
```

This will log:
- Op type (Conv)
- Kernel type (conv)
- Pads, strides, dilations values
- Group parameter
- Input/output tensor shapes and types

## References

- [hipDNNEP Reference Implementation](https://github.com/MaheshRavishankar/hipDNNEP)
- [ONNX Conv Operator Spec](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)
- hipDNN Frontend API Documentation
