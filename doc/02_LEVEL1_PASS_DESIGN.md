# Level-1 ROCm Pass Design

## Overview

The Level-1 ROCm pass (`vaip-pass_level1_rocm`) serves as the orchestrator for ROCm-based graph optimizations. It coordinates Level-2 sub-passes that perform pattern matching for specific operations (Conv, Gemm), then groups consecutive fused nodes and builds a `RocmSubgraphProto` that represents the complete topology for efficient execution on AMD GPUs.

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

Sub-passes are configured via `pass_generic_param` in `vaip_config.json`:

```json
{
  "sub_pass_names": ["vaip-pass_level2_rocm_conv", "vaip-pass_level2_rocm_gemm"]
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

### Step 2: Finding ROCm Fused Nodes

After sub-passes complete, the graph contains fused nodes created by Level-2 passes. These nodes are identified by domain `com.xilinx`:

```cpp
bool is_rocm_fused_node(const Node& node) {
  auto domain = node_op_domain(node);
  return domain == "com.xilinx";
}
```

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

For detailed algorithm description, see [03_GROUPING_ALGORITHM.md](03_GROUPING_ALGORITHM.md).

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

### RocmSubgraphProto

The main data structure representing a fused subgraph:

```protobuf
message RocmSubgraphProto {
  // Nodes in topological (execution) order
  repeated RocmNodeProto nodes = 1;
  
  // External outputs with source node mappings
  repeated ExternalOutputProto outputs = 2;
}
```

### RocmNodeProto

A single node in the subgraph:

```protobuf
message RocmNodeProto {
  int32 node_id = 1;                      // Unique ID (0-indexed)
  RocmParamProto params = 2;              // Operation parameters
  repeated TensorRefProto inputs = 3;     // Input references
  repeated string output_names = 4;       // Output tensor names
}
```

### TensorRefProto

Reference to a tensor input - either external or internal:

```protobuf
message TensorRefProto {
  oneof source {
    string external_name = 1;          // From ORT context
    InternalTensorRefProto internal = 2; // From another node
  }
}

message InternalTensorRefProto {
  int32 producer_node_id = 1;  // Which node produces this
  int32 output_index = 2;      // Which output (usually 0)
}
```

### ExternalOutputProto

Maps external outputs to their producing nodes:

```protobuf
message ExternalOutputProto {
  string name = 1;                 // ORT output tensor name
  int32 producer_node_id = 2;      // Which node produces it
  int32 output_index = 3;          // Which output
}
```

## Example: Conv → Conv Subgraph

Consider this pattern: `X → Conv1 → Conv2 → Y`

The Level-1 pass builds:

```protobuf
RocmSubgraphProto {
  nodes: [
    {
      node_id: 0
      params: { op_type: "conv", conv_params: {...weight_file, shapes...} }
      inputs: [
        { external_name: "X" }  // From ORT
      ]
      output_names: ["conv1_out"]
    },
    {
      node_id: 1
      params: { op_type: "conv", conv_params: {...weight_file, shapes...} }
      inputs: [
        { internal: { producer_node_id: 0, output_index: 0 } }  // From Conv1
      ]
      output_names: ["conv2_out"]
    }
  ]
  outputs: [
    { name: "Y", producer_node_id: 1, output_index: 0 }
  ]
}
```

## Benefits of RocmSubgraphProto Design

1. **Explicit Topology**: The subgraph structure is explicitly represented with `TensorRefProto`
2. **Type-Safe**: Uses `oneof` instead of sentinel values like `-1`
3. **Async-Ready**: `ExternalOutputProto` mapping enables overlapped D2H transfers
4. **Memory Optimization**: Intermediate tensors (internal references) stay on GPU
5. **Extensible**: Easy to add new operation types

## Async Execution Enabled by ExternalOutputProto

The `ExternalOutputProto` mapping enables this optimization:

```
Time →
┌───────────────────────────────────────────────────────────────────────────┐
│ H2D(X) │ Conv1 │ Conv2 │ D2H(conv2_out) │ Gemm │ D2H(gemm_out) │ sync    │
└───────────────────────────────────────────────────────────────────────────┘
                         ↑                        ↑
                         └── can overlap with Gemm execution!
```

When the custom op finishes executing node N, it checks if any `ExternalOutputProto` maps to that node and immediately issues `hipMemcpyAsync` for D2H transfer.

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
[HIP EP Level-1] Running sub-pass: vaip-pass_level2_rocm_conv
[HIP EP Level-1] Running sub-pass: vaip-pass_level2_rocm_gemm
[HIP EP Level-1] Found 3 ROCm fused nodes
[HIP EP Level-1] Found 2 mergeable groups
[HIP EP Level-1] Group 0: 2 nodes (Conv1, Conv2)
[HIP EP Level-1] Building RocmSubgraphProto for group 0
[HIP EP Level-1]   Node 0: conv, inputs: [external:X], outputs: [conv1_out]
[HIP EP Level-1]   Node 1: conv, inputs: [internal:0.0], outputs: [conv2_out]
[HIP EP Level-1]   External outputs: [Y -> node 1, output 0]
[HIP EP Level-1] Created merged fused node: rocm_subgraph_0
[HIP EP Level-1] Completed
```

## See Also

- [01_DESIGN.md](01_DESIGN.md) - Overall project design
- [03_GROUPING_ALGORITHM.md](03_GROUPING_ALGORITHM.md) - Union-Find algorithm details
- [Level-2 Conv Pass](../level-2-pass-rocm-conv/) - Conv pattern matching
- [Level-2 Gemm Pass](../level-2-pass-rocm-gemm/) - Gemm pattern matching
- [Custom Op](../custom-op-rocm/) - Subgraph execution
