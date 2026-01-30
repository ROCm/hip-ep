<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# morphizen-pattern

Pattern matching library for ONNX computational graphs.

## Overview

`morphizen-pattern` provides a powerful pattern matching system for identifying and transforming subgraphs in ONNX models. This component is used extensively in graph optimization passes to find opportunities for fusion, quantization, and other transformations.

## Features

- **12 Pattern Types**: Wildcard, Node, Constant, CommutableNode, Or, Sequence, Where, GraphInput, GraphOutput, NodeOutputArg, and more
- **PatternBuilder**: Declarative API for constructing complex patterns
- **Binder**: Captures matched nodes and values for use in transformations
- **RewriteRule**: Framework for applying graph transformations based on patterns
- **Protocol Buffers**: Serializable pattern representation for external tools

## Pattern Types

| Pattern Type | Description | Example Use Case |
|--------------|-------------|------------------|
| `wildcard` | Matches any node | Template patterns |
| `node` | Matches specific operator type | Find "Conv" nodes |
| `constant` | Matches constant initializers | Find constant weights |
| `commutable_node` | Matches commutative ops (Add, Mul) | Order-independent matching |
| `or` | Matches one of several alternatives | "Conv OR Gemm" |
| `sequence` | Matches ordered sequence | "Conv → Relu → MaxPool" |
| `where` | Conditional matching with predicates | "Conv with stride=1" |
| `graph_input` | Matches graph inputs | Input validation |
| `graph_output` | Matches graph outputs | Output analysis |
| `node_output_arg` | Matches specific node outputs | Track specific tensors |

## Architecture

```
┌──────────────────────────────────────────────┐
│    morphizen-pattern (~2,200 LOC)           │
│                                              │
│  Pattern Matching:                           │
│  - 12 pattern types                          │
│  - PatternBuilder (declarative API)          │
│  - Binder (captures matched nodes)           │
│  - Pattern serialization (protobuf)          │
│                                              │
│  Rewrite Framework:                          │
│  - RewriteRule base class                    │
│  - Rule chaining                             │
│  - Action callbacks                          │
│                                              │
│  Uses morphizen-graph wrappers               │
│  (NOT MORPHIZEN_ORT_API directly)           │
│  ↓                                           │
└──────────────────────────────────────────────┘
         │
         ↓
┌──────────────────────────────────────────────┐
│      morphizen-graph                         │
│      Graph/Node/NodeArg wrappers             │
└──────────────────────────────────────────────┘
```

## Dependencies

- **PUBLIC**: `morphizen::morphizen-graph` - Graph wrapper utilities
- **PRIVATE**: `protobuf::libprotobuf`, `glog::glog`, `Microsoft.GSL::GSL`

## Usage

### Basic Pattern Matching

```cpp
#include <morphizen/pattern.hpp>

using namespace morphizen;

// Create a simple pattern to match Conv → Relu sequence
auto pattern = PatternBuilder()
    .node("Conv", "")
    .outputs({"conv_out"})
    .build();

// Match pattern against graph
const Graph& graph = ...;
for (const auto& node : graph_get_nodes(graph)) {
    Binder binder;
    if (pattern->match(graph, node, binder)) {
        // Pattern matched! Access captured nodes via binder
        const Node* conv = binder.get_node("conv_out");
        // ... apply transformation
    }
}
```

### Complex Pattern with Conditions

```cpp
// Match Conv → Relu where Conv has specific attributes
auto pattern = PatternBuilder()
    .node("Conv", "")
    .where([](const Node& node) {
        return node_get_attr_int(node, "group") == 1;
    })
    .outputs({"conv_out"})
    .node("Relu", "")
    .inputs({"conv_out"})
    .build();
```

### Rewrite Rules

```cpp
#include <morphizen/rewrite_rule.hpp>

// Create a rewrite rule
auto rule = Rule::create_rule(
    pattern,
    [](onnxruntime::Graph* graph, binder_t& binder) -> bool {
        // Access matched nodes
        const Node* conv = binder.get_node("conv_out");
        const Node* relu = binder.get_node_arg("relu_out")->node;

        // Apply transformation
        // ... fuse Conv and Relu into ConvRelu

        return true; // Graph was modified
    }
);

// Apply rule to graph
rule->apply(graph);
```

## Configuration

Pattern matching can be disabled at build time for ultra-lite builds:

```bash
cmake -Dmorphizen_ENABLE_PATTERN_MATCHING=OFF ..
```

When disabled:
- `morphizen-pattern` component is not built
- `morphizen-core` has `MORPHIZEN_HAS_PATTERN_MATCHING=0`
- Pattern-dependent features are excluded

## Size

- ~2,200 lines of code
- 12 pattern implementation files
- Protocol buffer definitions
- RewriteRule framework

## Building

```bash
cmake -B build -Dmorphizen_ENABLE_PATTERN_MATCHING=ON
cmake --build build --target morphizen-pattern
```

## Use Cases

1. **Graph Fusion**: Identify and fuse operator sequences (Conv+Relu → ConvRelu)
2. **Quantization**: Find quantization opportunities (DQ → Op → Q patterns)
3. **Optimization**: Apply peephole optimizations (constant folding, dead code elimination)
4. **Analysis**: Validate graph structure, find specific patterns for profiling
5. **Code Generation**: Identify subgraphs for custom operator mapping

## Design Decisions

### Pattern::enable_trace() - No-op Implementation

During component extraction, we deliberately made `Pattern::enable_trace()` a no-op function that does nothing.

**Rationale:**

The original implementation relied on `env_config` from morphizen-core, which would create a circular dependency. We considered three alternatives:

1. **Add morphizen-utils dependency** (for ENV_PARAM)
   - ❌ Contradicts extraction goal of minimal dependencies
   - ❌ Increases coupling

2. **Use std::getenv() directly**
   - ⚠️ Platform-specific, less robust than ENV_PARAM
   - ⚠️ Still adds code complexity

3. **Make it a no-op** ✅ **CHOSEN**
   - ✅ Zero dependencies, maintains component independence
   - ✅ Simpler build graph
   - ✅ API preserved for backward compatibility
   - ✅ MY_LOG in `pattern_log.hpp` still provides logging
   - ⚠️ Lost runtime trace level control

**For Users Needing Trace Control:**

If you need to control pattern matching verbosity:

```bash
# Use glog's verbosity control instead
export GLOG_v=2  # Verbose logging
export GLOG_v=0  # Minimal logging
```

Or modify `pattern_log.hpp` in a custom build:

```cpp
// pattern_log.hpp
#define MY_LOG(n) if (n <= CUSTOM_TRACE_LEVEL) LOG(INFO)
```

This design prioritizes component independence over convenience features, keeping morphizen-pattern lightweight and reusable.

## License

Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
