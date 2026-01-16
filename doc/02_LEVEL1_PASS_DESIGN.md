# Level-1 ROCm Pass Design

## Overview

The Level-1 ROCm pass (`vaip-pass_level1_rocm`) serves as the orchestrator for ROCm-based graph optimizations. It coordinates Level-2 sub-passes that perform pattern matching for specific operations (Conv, Gemm), then merges consecutive fused nodes into larger subgraphs for efficient execution on AMD GPUs.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Original Graph (Read-Only)                │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     Level-1 Pass: ROCm Orchestrator           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   1. Clone Model                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   2. Run Sub-Passes                      │ │
│  │  ┌─────────────────┐    ┌─────────────────┐             │ │
│  │  │  L2 Conv Pass   │    │  L2 Gemm Pass   │             │ │
│  │  │  (MIOpen)       │    │  (hipBLASLt)    │             │ │
│  │  └─────────────────┘    └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        3. Find & Group ROCm Fused Nodes                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        4. Merge Groups into Original Graph               │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     Modified Original Graph                   │
│           (Contains merged ROCm fused nodes)                  │
└──────────────────────────────────────────────────────────────┘
```

## Processing Steps

### Step 1: Model Cloning

The original graph passed to the Level-1 pass is **read-only**. To allow Level-2 sub-passes to modify the graph (via `fuse()`), we first clone the entire model:

```cpp
auto& model = VAIP_ORT_API(graph_get_model)(graph);
auto cloned_model = vaip_core::model_clone(model, 64);
auto& cloned_graph = VAIP_ORT_API(model_main_graph)(*cloned_model);
```

### Step 2: Running Sub-Passes

Sub-passes are configured via `pass_generic_param` in `vaip_config.json`:

```json
{
  "sub_pass_names": ["vaip-pass_level2_rocm_conv", "vaip-pass_level2_rocm_gemm"]
}
```

Each sub-pass is created and run on the cloned graph:

```cpp
PassProto sub_pass_proto;
sub_pass_proto.set_plugin(sub_pass_name);
sub_pass_proto.set_name(sub_pass_name);

auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
IPass::run_passes({sub_pass}, cloned_graph);
```

### Step 3: Finding ROCm Fused Nodes

After sub-passes complete, the cloned graph contains fused nodes created by Level-2 passes. These nodes are identified by the presence of a `rocm_meta_def` node attribute.

#### Node Attribute Structure

```
Fused Node
├── domain: "com.xilinx"
├── op_type: "super_layer"  (hardcoded by morphizen framework)
└── attributes:
    └── rocm_meta_def: <serialized MetaDefProto>
          ├── id, inputs[], outputs[], nodes[]
          └── generic_param: <serialized RocmParamProto>
                ├── op_type: "conv" | "gemm"
                └── conv_params | gemm_params
```

#### Detection Logic

```cpp
bool is_rocm_fused_node(const Node& node) {
  // ROCm fused nodes have the rocm_meta_def attribute
  // set by Level-2 passes
  return node_get_attr(node, "rocm_meta_def") != nullptr;
}
```

**Note**: Level-2 passes set the `rocm_meta_def` attribute on each fused node they create. This attribute contains a serialized `MetaDefProto` which includes the operation parameters via `generic_param`.

### Step 4: Grouping Consecutive Nodes

ROCm fused nodes are grouped into **connected components** - nodes that can share a single HIP stream for implicit fusion. Two nodes are connected if one directly consumes the output of the other (no intermediate non-ROCm node between them).

```
Example: X → [Conv1] → T1 → [Conv2] → T2 → [ReLU_CPU] → T3 → [Gemm] → Y

ROCm nodes: Conv1, Conv2, Gemm
Non-ROCm: ReLU_CPU

Groups formed:
- Group 1: {Conv1, Conv2} - directly connected
- Group 2: {Gemm}         - separated by ReLU_CPU
```

For detailed algorithm description, see [03_GROUPING_ALGORITHM.md](03_GROUPING_ALGORITHM.md).

### Step 5: Collecting Inputs/Outputs

For each group, identify:

- **External Inputs**: Inputs not produced by nodes in the group
- **External Outputs**: Outputs consumed outside the group or are graph outputs

```
Group: [Conv1] → [Conv2]
         ↓
       [ReLU] (not in group)

External Inputs: Conv1's input (X, W, B)
External Outputs: Conv2's output (consumed by ReLU)
```

### Step 6: Merging into Original Graph

Create a single merged fused node in the **original** graph:

```cpp
auto [meta_def, error] = self_.try_fuse(
    graph,                  // Original graph
    merged_name,            // e.g., "rocm_merged_0"
    external_inputs,        // All external inputs
    external_outputs,       // All external outputs
    {},                     // No constant initializers
    "ROCm_EP"               // EP name
);

if (meta_def) {
    // Attach merged parameters
    rocm::RocmMergedParamProto merged_params;
    merged_params.set_op_count(group.size());
    merged_params.set_implicit_fusion(true);
    
    self_.attach_meta_def_param(*meta_def, merged_params_str);
    self_.fuse(graph, std::move(*meta_def));
}
```

## Data Structures

### RocmMergedParamProto

Stores parameters for merged nodes:

```protobuf
message RocmMergedParamProto {
  repeated RocmParamProto rocm_params = 1;  // Individual op params
  int32 op_count = 2;                        // Number of operations
  bool implicit_fusion = 3;                  // Share HIP stream
}
```

### PassRocmConfigProto

Configuration for Level-1 pass:

```protobuf
message PassRocmConfigProto {
  repeated string sub_pass_names = 1;  // Sub-pass names
  bool debug = 2;                       // Enable debug logging
}
```

## Benefits of Merging

1. **Reduced Kernel Launch Overhead**: One fused kernel instead of multiple
2. **Implicit Operation Fusion**: All ops share a single HIP stream
3. **Memory Optimization**: Intermediate tensors can be optimized away
4. **Better GPU Utilization**: Larger subgraphs allow better scheduling

## Configuration

### vaip_config.json

```json
{
  "name": "vaip-pass_level1_rocm",
  "plugin": "vaip-pass_level1_rocm",
  "pass_generic_param": "{\"sub_pass_names\": [\"vaip-pass_level2_rocm_conv\", \"vaip-pass_level2_rocm_gemm\"]}"
}
```

### Environment Variables

- `MORPHIZEN_DEBUG_ROCM=1`: Enable Level-1 pass logging
- `MORPHIZEN_DEBUG_ROCM=2`: Enable verbose logging (node details)

## Debugging

Enable debug output:

```bash
set MORPHIZEN_DEBUG_ROCM=1
```

Example output:

```
[HIP EP Level-1] Starting ROCm pass
[HIP EP Level-1] Cloned model for sub-pass processing
[HIP EP Level-1] Running sub-pass: vaip-pass_level2_rocm_conv
[HIP EP Level-1] Running sub-pass: vaip-pass_level2_rocm_gemm
[HIP EP Level-1] Found 3 ROCm fused nodes
[HIP EP Level-1] Found 2 mergeable groups
[HIP EP Level-1] Processing group 0 with 2 nodes
[HIP EP Level-1] Created merged fused node: rocm_merged_0
[HIP EP Level-1] Processing group 1 with 1 nodes
[HIP EP Level-1] Created merged fused node: rocm_merged_1
[HIP EP Level-1] Completed
```

## See Also

- [01_DESIGN.md](01_DESIGN.md) - Overall project design
- [Level-2 Conv Pass](../level-2-pass-rocm-conv/) - Conv pattern matching
- [Level-2 Gemm Pass](../level-2-pass-rocm-gemm/) - Gemm pattern matching
- [Custom Op](../custom-op-rocm/) - Runtime execution
