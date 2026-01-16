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

### Key Insight: DAG Property

Since ONNX graphs are **DAGs (Directed Acyclic Graphs)**, we can leverage topological order:

1. Nodes are processed in topological order (producers before consumers)
2. When we process a node, all its input producers have already been processed
3. We can directly merge a node into its producer's group using Union-Find

This avoids the O(N²) nested loop and gives us O(N α(N)) ≈ **O(N) time complexity**.

### Input
- `R`: List of ROCm fused nodes (in topological order)

### Output
- List of groups, where each group contains connected ROCm nodes

### Pseudo-code (Union-Find Based)

```
function find_groups(R):
    // Union-Find data structure
    parent = {}      // parent[node] = parent node (or self if root)
    
    for each node in R:
        parent[node] = node  // Initialize: each node is its own group
    
    // Build map: output_name → node
    producer_map = build_producer_map(R)
    
    // Process in topological order
    for each node in R:
        for each input of node:
            if input in producer_map:
                producer = producer_map[input]
                // Merge node's group with producer's group
                union(parent, node, producer)
    
    // Collect groups from Union-Find
    groups = {}
    for each node in R:
        root = find(parent, node)
        if root not in groups:
            groups[root] = []
        groups[root].append(node)
    
    return values(groups)

function find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])  // Path compression
    return parent[node]

function union(parent, a, b):
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a != root_b:
        parent[root_a] = root_b  // Merge groups
```

### Why DAG Ensures No Cycles After Fusion

When we merge a group of nodes into a single fused node:

1. **External inputs** come from nodes outside the group (processed earlier in topo order)
2. **External outputs** go to nodes outside the group (processed later in topo order)
3. The fused node takes the topological position of the **last** node in the group

Since the original graph has no cycles, and we only merge nodes with direct data-flow edges, the resulting graph is still a DAG.

```
Original:      After Merging:
  A → B → C      A → [Merged BC] → D
      ↓   ↓              ↓
      E   D              E

The merged node [BC] still respects DAG property:
- It receives from A
- It outputs to D and E
- No cycles introduced
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

### Step-by-Step (Union-Find)

1. **Identify ROCm nodes**: `R = [FusedConv1, FusedConv2, FusedGemm]` (topological order)

2. **Initialize Union-Find**:
   ```
   parent = {
     FusedConv1 → FusedConv1,
     FusedConv2 → FusedConv2,
     FusedGemm  → FusedGemm
   }
   ```

3. **Build Producer Map** (only ROCm nodes):
   ```
   producer_map = {
     "T1" → FusedConv1,
     "T2" → FusedConv2,
     "Y"  → FusedGemm
   }
   ```
   Note: `T3` is NOT in the map (produced by ReLU, which is not in R)

4. **Process nodes in topological order**:

   - **FusedConv1**: No inputs from ROCm nodes (X, W1, B1 not in producer_map)
   
   - **FusedConv2**: Input T1 is in producer_map → FusedConv1
     - `union(FusedConv2, FusedConv1)` → parent[FusedConv2] = FusedConv1
   
   - **FusedGemm**: Input T3 not in producer_map (produced by ReLU)
     - No union operation

5. **Collect groups**:
   ```
   find(FusedConv1) → FusedConv1  → Group: {FusedConv1, FusedConv2}
   find(FusedConv2) → FusedConv1  
   find(FusedGemm)  → FusedGemm   → Group: {FusedGemm}
   ```

6. **Result**: `[[FusedConv1, FusedConv2], [FusedGemm]]`

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

- **Time**: O(N × α(N)) ≈ O(N) where N is the number of ROCm nodes
  - Single pass through nodes in topological order
  - Union-Find with path compression: α(N) is the inverse Ackermann function, nearly constant
  - Building producer map: O(N × M) where M is average outputs per node

- **Space**: O(N) for the Union-Find parent map and producer map

## Implementation

See `level-1-pass-rocm/src/pass_main.cpp`:
- `build_producer_map()` - Builds the producer map
- `are_connected()` - Checks if two nodes are connected
- `find_mergeable_groups()` - Main grouping algorithm

## See Also

- [02_LEVEL1_PASS_DESIGN.md](02_LEVEL1_PASS_DESIGN.md) - Level-1 pass overview
