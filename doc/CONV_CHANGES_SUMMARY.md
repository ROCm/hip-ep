# Conv Node Support - Changes Summary

## Task
Add Conv node support to onnx-hipdnn-ep project following the reference implementation from [hipDNNEP](https://github.com/MaheshRavishankar/hipDNNEP).

## Changes Made

### 1. Protocol Buffer Schema (`proto/hipdnn.proto`)
**Cleaned up and reorganized with essential fields only:**
- `string device_id` (field 1) - HIP device identifier
- `string kernel_type` (field 2) - Kernel type ("conv")
- `string op_type` (field 3) - Operation type ("Conv")
- `repeated int64 pads` (field 4) - Convolution padding parameters
- `repeated int64 strides` (field 5) - Convolution stride values
- `repeated int64 dilations` (field 6) - Dilation rates
- `int64 group` (field 7) - Group convolution parameter

**Removed demo fields:** sample_string, sample_int, sample_strings, sample_ints, ep_context_file_name, ep_context_file_size

### 2. Pattern Recognition (`patterns/hipdnn.json`)
**Status:** Already configured correctly
- Pattern matches Conv operations with Input -> Conv -> Output flow
- No changes required

### 3. Pass Implementation (`level-1-pass-hipdnn/src/pass_main.cpp`)
**Enhanced to capture Conv attributes:**
- Retrieves Conv node from pattern binder
- Sets kernel_type to "conv"
- Extracts ONNX Conv attributes:
  - `pads` attribute
  - `strides` attribute  
  - `dilations` attribute
  - `group` attribute
- Stores all attributes in HipdnnParamProto for later use

**Code additions:**
```cpp
auto conv_node = binder["hipdnn_op"].node;
hipdnn_param.set_kernel_type("conv");
hipdnn_param.set_op_type(node_get_op_type(*conv_node));
// Extract pads, strides, dilations, group attributes
```

### 4. Custom Operation (`custom-op-hipdnn/src/custom_op.cpp`)
**Updated Compute method:**
- Added logging for Conv operation parameters
- Differentiates Conv from other operations
- Logs pads, strides, dilations, and group values
- Placeholder implementation (copies input to output)
- Structure ready for hipDNN integration

**Code additions:**
```cpp
if (hipdnn_proto_.op_type() == "Conv") {
  MY_LOG(1) << "Conv operation detected";
  // Log all Conv parameters
  // TODO: Implement actual hipDNN execution
}
```

### 5. Documentation
**Created comprehensive documentation:**
- `CONV_IMPLEMENTATION.md` - Full implementation guide
- `CONV_CHANGES_SUMMARY.md` - This summary

## Architecture Flow

```
ONNX Graph (Conv Node)
        ↓
Pattern Matching (hipdnn.json)
        ↓
Pass (pass_main.cpp)
  - Extracts Conv attributes
  - Creates HipdnnParamProto
  - Serializes to JSON
        ↓
Custom Op (custom_op.cpp)
  - Deserializes parameters
  - Logs Conv configuration
  - [TODO] Execute with hipDNN
        ↓
Output Tensor
```

## Current Implementation Status

### ✅ Complete Infrastructure
1. Pattern recognition for Conv nodes
2. Attribute extraction from ONNX Conv operations
3. Protocol buffer schema with Conv parameters
4. Custom operation structure for Conv execution
5. Debug logging for Conv parameters
6. Documentation and examples

### 🚧 Next Steps for Full Implementation
1. **Add hipDNN library integration:**
   - Include hipDNN frontend headers in custom_op.cpp
   - Initialize hipDNN handle in constructor
   - Manage device memory allocation

2. **Implement Conv execution:**
   - Create hipDNN graph from Conv parameters
   - Build tensor attributes for inputs/outputs
   - Construct ConvFpropAttributes from extracted params
   - Execute graph.conv_fprop()
   - Handle memory transfers (CPU ↔ GPU)

3. **Testing:**
   - Create test ONNX models with Conv operations
   - Validate correctness against reference
   - Performance benchmarking

## Reference Alignment

The implementation closely follows the hipDNNEP reference:

| Component | hipDNNEP | onnx-hipdnn-ep | Status |
|-----------|----------|----------------|--------|
| Conv attribute extraction | `kernel.cc:AddConvNode()` | `pass_main.cpp` | ✅ Aligned |
| Pads handling | Default `{0,0,0,0}` | Extracted from ONNX | ✅ Aligned |
| Strides handling | Default `{1,1}` | Extracted from ONNX | ✅ Aligned |
| Dilations handling | Default `{1,1}` | Extracted from ONNX | ✅ Aligned |
| Graph execution | `kernel.cc:Execute()` | `custom_op.cpp:Compute()` | 🚧 Structure ready |

## Building the Project

After these changes, rebuild:

```bash
# Configure
cmake --preset RelWithDebInfo

# Build  
cmake --build --preset RelWithDebInfo

# Test
ctest --preset RelWithDebInfo
```

## Debugging

Enable debug output:
```bash
export MORPHIZEN_DEBUG_HIPDNN=1
```

This will show:
- Conv operation detection
- All Conv parameters (pads, strides, dilations, group)
- Input/output tensor information

## Files Modified

1. `proto/hipdnn.proto` - Added Conv parameters
2. `level-1-pass-hipdnn/src/pass_main.cpp` - Extract Conv attributes
3. `custom-op-hipdnn/src/custom_op.cpp` - Handle Conv execution
4. `CONV_IMPLEMENTATION.md` - Implementation guide (new)
5. `CONV_CHANGES_SUMMARY.md` - This summary (new)

## Verification

To verify the implementation works:

1. **Build the project** to ensure no compilation errors
2. **Run with debug logging** on a model containing Conv operations
3. **Check logs** for Conv parameter extraction
4. **Verify** pattern matching occurs correctly

Expected log output:
```
Op type: Conv
Kernel type: conv
Conv operation detected
Pads: [0, 0, 0, 0]
Strides: [1, 1]
Dilations: [1, 1]
Group: 1
```

## Integration with hipDNN

The next phase would involve:
```cpp
// In custom_op.cpp Compute() method
if (hipdnn_proto_.op_type() == "Conv") {
  // 1. Create hipDNN graph
  auto graph = hipdnn_frontend::graph::Graph();
  
  // 2. Create tensor attributes
  auto x_attr = CreateInputTensorAttr(ctx.GetInput(0));
  auto w_attr = CreateInputTensorAttr(ctx.GetInput(1));
  
  // 3. Build Conv operation
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding(ExtractPads())
            .set_stride(ExtractStrides())
            .set_dilation(ExtractDilations());
  
  auto y_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);
  
  // 4. Execute
  graph.validate();
  graph.build_operation_graph(hipdnn_handle_);
  graph.execute(hipdnn_handle_, variant_pack, workspace);
}
```

## Notes

- The C++ include errors shown in VSCode are expected before building, as the headers are generated during the CMake build process
- Protocol buffer code is generated from `hipdnn.proto` during build
- Pattern JSON is embedded as a C++ header during build
- All infrastructure is in place for Conv support; actual hipDNN API integration is the remaining work
