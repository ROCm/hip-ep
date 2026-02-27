<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Level-1 ROCm Pass Design

## Overview

The Level-1 ROCm pass (`morphizen-pass_level1_rocm`) serves as the orchestrator for ROCm-based graph optimizations. It coordinates Level-2 sub-passes that perform pattern matching for specific operations (Conv, Gemm), then groups consecutive fused nodes and builds a `RocmSubgraphProto` that represents the complete topology for efficient execution on AMD GPUs.

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
│  │                   1. Run Sub-Passes                      │ │
│  │  ┌─────────────────┐    ┌─────────────────┐             │ │
│  │  │  L2 Conv Pass   │    │  L2 Gemm Pass   │             │ │
│  │  │  (MIOpen)       │    │  (hipBLASLt)    │             │ │
│  │  └─────────────────┘    └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        2. Find & Group ROCm Fused Nodes (Union-Find)     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        3. Build RocmSubgraphProto for Each Group         │ │
│  │           - Create RocmNodeProto for each node           │ │
│  │           - Build TensorRefProto for each input          │ │
│  │           - Map ExternalOutputProto for outputs          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        4. Create Merged Fused Node in Graph              │ │
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

### Step 1: Running Sub-Passes

Sub-passes are configured via `pass_generic_param` in `morphizen_config.json`:

```json
{
  "sub_pass_names": ["morphizen-pass_level2_rocm_conv", "morphizen-pass_level2_rocm_gemm"]
}
```

Each sub-pass is created and run on the graph:

```cpp
for (const auto& sub_pass_name : config.sub_pass_names()) {
  PassProto sub_pass_proto;
  sub_pass_proto.set_plugin(sub_pass_name);
  sub_pass_proto.set_name(sub_pass_name);

  auto sub_pass = IPass::create_pass(self_.get_context(), sub_pass_proto);
  IPass::run_passes({sub_pass}, graph);
}
```

**Important:** Level-2 passes use `level_2_fuse()` instead of `fuse()`. The key difference:
- `fuse()`: Creates fused node AND updates `context.json`
- `level_2_fuse()`: Creates fused node but does NOT update `context.json`

Level-1 calls `fuse()` after merging subgraphs, ensuring only merged nodes are written to `context.json`.

#### Level-2 to Level-1 Parameter Passing

Level-2 passes store operation parameters using a hybrid approach:

1. **Write params to cache file** (for troubleshooting):
   ```cpp
   std::string param_filename = "rocm_param_" + output_name + ".json";
   auto writer = pass_context->open_file_for_write(param_filename);
   writer->fwrite(rocm_param_json.data(), rocm_param_json.size());
   ```

2. **Add attribute to fused node** (for Level-1 to find):
   ```cpp
   const auto& fused_node = self->level_2_fuse(*graph, *meta_def);

   // const_cast is needed because level_2_fuse returns const Node&
   Node& mutable_node = const_cast<Node&>(fused_node);
   NodeAttributesBuilder attr_builder;
   attr_builder.add("rocm_param_file", param_filename);
   attr_builder.merge_into(mutable_node);
   ```

Level-1 retrieves params by:
1. Finding nodes with `rocm_param_file` attribute using `NodeConstRef.has_attr()`
2. Getting the param filename using `NodeConstRef.get_attr_string()`
3. Reading the JSON file from cache and parsing into `RocmParamProto`

This design enables:
- **Debuggability**: Param JSON files can be inspected in the cache directory
- **Traceability**: Each node's params are clearly linked via ONNX node attribute
- **Consistency**: Same pattern as weight files (`weight_file_path` in params)

### Step 2: Finding ROCm Fused Nodes

After sub-passes complete, the graph contains fused nodes created by Level-2 passes. These nodes are identified by:
- Domain `com.xilinx` (morphizen framework's domain for fused nodes)
- Having the `rocm_param_file` attribute (set by Level-2 ROCm passes)

```cpp
bool is_rocm_fused_node(Graph& graph, const Node& node) {
  auto node_ref = morphizen_cxx::NodeConstRef::from_node(graph, node);

  if (node_ref.op_domain() != "com.xilinx") {
    return false;
  }

  // Check for ROCm-specific attribute set by Level-2 passes
  if (!node_ref.has_attr("rocm_param_file")) {
    return false;
  }

  return true;
}
```

This two-criteria approach ensures we only match nodes from ROCm Level-2 passes, not nodes from other Level-1 passes (e.g., NPU, DPU) that also use the `com.xilinx` domain.

### Step 3: Grouping Consecutive Nodes (Union-Find)

ROCm fused nodes are grouped into **connected components** using Union-Find algorithm. Two nodes are connected if one directly consumes the output of the other:

```
Example: X → [Conv1] → T1 → [Conv2] → T2 → [ReLU_CPU] → T3 → [Gemm] → Y

ROCm nodes: Conv1, Conv2, Gemm
Non-ROCm: ReLU_CPU

Groups formed:
- Group 1: {Conv1, Conv2} - directly connected
- Group 2: {Gemm}         - separated by ReLU_CPU
```

The Union-Find algorithm efficiently groups connected ROCm nodes.

### Step 4: Building RocmSubgraphProto

For each group, build a `RocmSubgraphProto` with complete topology:

```cpp
RocmSubgraphProto build_subgraph(
    const std::vector<const Node*>& group,
    Graph& graph) {

  RocmSubgraphProto subgraph;

  // Map from original node to new node_id
  std::unordered_map<const Node*, int32_t> node_id_map;
  std::unordered_map<std::string, std::pair<int32_t, int32_t>> output_producer_map;

  // Build nodes in topological order
  for (int32_t i = 0; i < group.size(); ++i) {
    const Node* node = group[i];
    node_id_map[node] = i;

    RocmNodeProto* node_proto = subgraph.add_nodes();
    node_proto->set_node_id(i);

    // Copy operation parameters
    *node_proto->mutable_params() = get_rocm_params(*node);

    // Build input references
    for (auto* input : node_get_input_node_args(*node)) {
      TensorRefProto* input_ref = node_proto->add_inputs();
      auto input_name = node_arg_get_name(*input);

      auto it = output_producer_map.find(input_name);
      if (it != output_producer_map.end()) {
        // Internal reference - from another node in subgraph
        auto* internal_ref = input_ref->mutable_internal();
        internal_ref->set_producer_node_id(it->second.first);
        internal_ref->set_output_index(it->second.second);
      } else {
        // External reference - from outside subgraph
        input_ref->set_external_name(input_name);
      }
    }

    // Register outputs for dependency tracking
    auto outputs = node_get_output_node_args(*node);
    for (int32_t j = 0; j < outputs.size(); ++j) {
      auto output_name = node_arg_get_name(*outputs[j]);
      output_producer_map[output_name] = {i, j};
      node_proto->add_output_names(output_name);
    }
  }

  // Identify external outputs
  auto external_output_names = collect_external_outputs(group, graph);
  for (const auto& name : external_output_names) {
    auto it = output_producer_map.find(name);
    if (it != output_producer_map.end()) {
      ExternalOutputProto* ext_output = subgraph.add_outputs();
      ext_output->set_name(name);
      ext_output->set_producer_node_id(it->second.first);
      ext_output->set_output_index(it->second.second);
    }
  }

  return subgraph;
}
```

### Step 5: Creating Merged Fused Node

Create a single merged fused node in the graph:

```cpp
auto [meta_def, error] = self_.try_fuse(
    graph,
    merged_name,           // e.g., "rocm_subgraph_0"
    external_inputs,       // All external inputs
    external_outputs,      // All external outputs
    constant_initializers, // Weights/biases
    "ROCm_EP"
);

if (meta_def) {
    std::string subgraph_json;
    google::protobuf::util::MessageToJsonString(subgraph_proto, &subgraph_json);
    self_.attach_meta_def_param(*meta_def, subgraph_json.c_str());
    self_.fuse(graph, std::move(*meta_def));
}
```

## Data Structures

> **Note:** For detailed protobuf message definitions and design rationale, see [01_DESIGN.md §6.2 - Core Messages](01_DESIGN.md#62-core-messages).

The key data structures used by the Level-1 pass are:

| Message | Purpose |
|---------|---------|
| `RocmSubgraphProto` | Container for the complete fused subgraph |
| `RocmNodeProto` | Single operation node with parameters and input references |
| `TensorRefProto` | References inputs (external from ORT or internal from another node) |
| `ExternalOutputProto` | Maps outputs back to ORT tensors for async D2H scheduling |

For a complete subgraph example showing how these structures work together, see [01_DESIGN.md §6.3 - Subgraph Example](01_DESIGN.md#63-subgraph-example).

## Configuration

### morphizen_config.json

```json
{
  "name": "morphizen-pass_level1_rocm",
  "plugin": "morphizen-pass_level1_rocm",
  "pass_generic_param": "{\"sub_pass_names\": [\"morphizen-pass_level2_rocm_conv\", \"morphizen-pass_level2_rocm_gemm\"]}"
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
[ROCm EP Level-1] Starting ROCm pass
[ROCm EP Level-1] Running sub-pass: morphizen-pass_level2_rocm_conv
[ROCm EP Level-1] Running sub-pass: morphizen-pass_level2_rocm_gemm
[ROCm EP Level-1] Found 3 ROCm fused nodes
[ROCm EP Level-1] Found 2 mergeable groups
[ROCm EP Level-1] Group 0: 2 nodes (Conv1, Conv2)
[ROCm EP Level-1] Building RocmSubgraphProto for group 0
[ROCm EP Level-1]   Node 0: conv, inputs: [external:X], outputs: [conv1_out]
[ROCm EP Level-1]   Node 1: conv, inputs: [internal:0.0], outputs: [conv2_out]
[ROCm EP Level-1]   External outputs: [Y -> node 1, output 0]
[ROCm EP Level-1] Created merged fused node: rocm_subgraph_0
[ROCm EP Level-1] Completed
```

## See Also

- [01_DESIGN.md](01_DESIGN.md) - Overall project design
- [03_GROUPING_ALGORITHM.md](03_GROUPING_ALGORITHM.md) - Node grouping algorithm
- [Level-2 Conv Pass](../level-2-pass-rocm-conv/) - Conv pattern matching
- [Level-2 Gemm Pass](../level-2-pass-rocm-gemm/) - Gemm pattern matching
- [Custom Op](../custom-op-rocm/) - Subgraph execution
