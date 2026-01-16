# Node Grouping Algorithm

This document describes the algorithm used by the Level-1 ROCm pass to group consecutive fused nodes for merging.

## Problem Definition

Given a graph with ROCm fused nodes (identified by `rocm_meta_def` attribute), find **maximal connected components** among these nodes. Connected nodes can share a single HIP stream for implicit operation fusion.

## Definitions

- **ROCm Node**: A fused node with the `rocm_meta_def` attribute set by Level-2 passes
- **Connected**: Two ROCm nodes are connected if one directly consumes the output of the other
- **Directly**: No intermediate non-ROCm node exists between them
- **Group**: A set of connected ROCm nodes that will be merged

## Data Structures

### Producer Map

Maps output tensor names to their producing nodes:

```cpp
std::unordered_map<std::string, const Node*> producer_map;

// Build by iterating all ROCm nodes
for (auto* node : rocm_nodes) {
  for (auto* output : node_get_output_node_args(*node)) {
    producer_map[node_arg_get_name(*output)] = node;
  }
}
```

### Connectivity Check

Two nodes are connected if one produces an input consumed by the other:

```cpp
bool are_connected(const Node& node_a, const Node& node_b,
                   const ProducerMap& producer_map) {
  for (auto* input : node_get_input_node_args(node_b)) {
    auto name = node_arg_get_name(*input);
    if (producer_map.count(name) && producer_map[name] == &node_a) {
      return true;
    }
  }
  return false;
}
```

## Algorithm

### Input
- `R`: List of ROCm fused nodes (in topological order)

### Output
- List of groups, where each group contains connected ROCm nodes

### Pseudo-code

```
function find_groups(R):
    producer_map = build_producer_map(R)
    assigned = {}      // Set of nodes already assigned to a group
    groups = []        // Result: list of groups
    
    for each node n in R:
        if n in assigned:
            continue
        
        // Start new group with BFS from n
        group = []
        queue = [n]
        assigned.add(n)
        
        while queue not empty:
            current = queue.pop_front()
            group.append(current)
            
            // Find all connected ROCm nodes
            for each other in R:
                if other in assigned:
                    continue
                    
                // Check both directions
                if are_connected(current, other, producer_map) or
                   are_connected(other, current, producer_map):
                    queue.append(other)
                    assigned.add(other)
        
        groups.append(group)
    
    return groups
```

## Example Walkthrough

### Input Graph

```
Original ONNX Graph:
  X ─────→ [Conv1] ─→ T1 ─→ [Conv2] ─→ T2 ─→ [ReLU] ─→ T3 ─→ [Gemm] ─→ Y
  W1 ──↗              W2 ──↗                            A ──↗
  B1 ──↗              B2 ──↗                            B ──↗

After Level-2 passes (Conv1, Conv2, Gemm are ROCm nodes; ReLU is CPU):
  X ─────→ [FusedConv1] ─→ T1 ─→ [FusedConv2] ─→ T2 ─→ [ReLU] ─→ T3 ─→ [FusedGemm] ─→ Y
```

### Step-by-Step

1. **Identify ROCm nodes**: `R = [FusedConv1, FusedConv2, FusedGemm]`

2. **Build Producer Map**:
   ```
   producer_map = {
     "T1" → FusedConv1,
     "T2" → FusedConv2,
     "Y"  → FusedGemm
   }
   ```
   Note: `T3` is NOT in the map (produced by ReLU, which is not in R)

3. **Find Groups**:

   - **Iteration 1**: Start with FusedConv1
     - BFS from FusedConv1
     - Check FusedConv2: `are_connected(FusedConv1, FusedConv2)?`
       - FusedConv2's input T1 is in producer_map → FusedConv1 ✓
     - Add FusedConv2 to queue
     - Check FusedGemm: `are_connected(FusedConv1, FusedGemm)?`
       - FusedGemm's inputs are T3, A, B - T3 not in producer_map (ReLU) ✗
     - Process FusedConv2 from queue
     - Check FusedGemm from FusedConv2: 
       - FusedGemm's input T3 not produced by FusedConv2 ✗
     - Queue empty → **Group 1: {FusedConv1, FusedConv2}**

   - **Iteration 2**: Start with FusedGemm (only unassigned node)
     - BFS from FusedGemm
     - All other nodes already assigned
     - **Group 2: {FusedGemm}**

4. **Result**: `[[FusedConv1, FusedConv2], [FusedGemm]]`

## Why Non-ROCm Nodes Act as Barriers

```
[Conv1] ─→ [ReLU_CPU] ─→ [Conv2]
```

When ReLU runs on CPU:
1. Conv1 output must be copied from GPU to CPU
2. ReLU executes on CPU
3. Result must be copied back to GPU for Conv2

This forces a **synchronization point** - Conv1 and Conv2 cannot share the same HIP stream for overlapped execution.

## Complexity

- **Time**: O(N²) where N is the number of ROCm nodes
  - For each node, we check connectivity with all other nodes
  - In practice, N is typically small (< 100 nodes)

- **Space**: O(N) for the producer map and assigned set

## Implementation

See `level-1-pass-rocm/src/pass_main.cpp`:
- `build_producer_map()` - Builds the producer map
- `are_connected()` - Checks if two nodes are connected
- `find_mergeable_groups()` - Main grouping algorithm

## See Also

- [02_LEVEL1_PASS_DESIGN.md](02_LEVEL1_PASS_DESIGN.md) - Level-1 pass overview
