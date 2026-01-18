# morphizen-rocm Design Document

## 1. Overview

### 1.1 Purpose

**morphizen-rocm** is a unified ROCm Execution Provider for ONNX Runtime that combines:
- **MIOpen**: AMD's optimized library for convolution operations
- **hipBLASLt**: AMD's high-performance library for GEMM (matrix multiplication)

This project consolidates the functionality of `morphizen-miopen` and `morphizen-hipblaslt` into a single, cohesive execution provider with a shared HIP context for implicit operation fusion.

### 1.2 Problem Statement

Deep learning models often contain sequences like:
```
Input → [Convolution] → [Reshape] → [Gemm/MatMul] → Output
```

Previously, these operations were handled by separate projects:
- `morphizen-miopen`: Conv operations
- `morphizen-hipblaslt`: Gemm operations

This led to:
- Duplicate infrastructure code
- No shared GPU context
- Potential for redundant memory transfers

### 1.3 Solution

Combine both libraries into a single EP with:
- **Unified Level-1/Level-2 pass architecture**
- **Shared HIP stream** for all operations
- **Single custom op library** supporting both operation types
- **Implicit fusion** through stream ordering (no explicit sync needed)
- **Subgraph execution** with intermediate tensors staying on GPU

---

## 2. Architecture

### 2.1 High-Level Architecture

<pre style="font-family: 'Courier New', Consolas, monospace; line-height: 1.2;"><code>┌─────────────────────────────────────────────────────────────────────────┐
│                           ONNX Runtime                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    VitisAI EP (VAIP)                             │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              Level-1 Pass: vaip-pass_level1_rocm          │   │   │
│  │  │  • Checks AMD GPU availability                            │   │   │
│  │  │  • Initializes shared HIP context                         │   │   │
│  │  │  • Orchestrates Level-2 sub-passes                        │   │   │
│  │  │  • Builds RocmSubgraphProto from grouped nodes            │   │   │
│  │  └─────────────────────┬────────────────────────────────────┘   │   │
│  │                        │                                         │   │
│  │          ┌─────────────┴─────────────┐                          │   │
│  │          ▼                           ▼                          │   │
│  │  ┌──────────────────┐      ┌──────────────────┐                 │   │
│  │  │ Level-2 Pass:    │      │ Level-2 Pass:    │                 │   │
│  │  │ rocm_conv        │      │ rocm_gemm        │                 │   │
│  │  │ (MIOpen)         │      │ (hipBLASLt)      │                 │   │
│  │  └────────┬─────────┘      └────────┬─────────┘                 │   │
│  │           │                         │                            │   │
│  │           └─────────────┬───────────┘                            │   │
│  │                         ▼                                        │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              Custom Op: custom-op-rocm                    │   │   │
│  │  │  • Shared HIP stream                                      │   │   │
│  │  │  • Executes RocmSubgraphProto nodes in order              │   │   │
│  │  │  • Keeps intermediate tensors on GPU                      │   │   │
│  │  │  • Async H2D/D2H transfers                                │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AMD GPU (ROCm)                                  │
│  ┌────────────────────┐    ┌────────────────────┐                      │
│  │      MIOpen        │    │     hipBLASLt      │                      │
│  │   (Convolution)    │    │      (GEMM)        │                      │
│  └────────────────────┘    └────────────────────┘                      │
│                     │              │                                    │
│                     └──────┬───────┘                                    │
│                            ▼                                            │
│              ┌──────────────────────────┐                              │
│              │    Shared HIP Stream     │                              │
│              └──────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
</code></pre>

### 2.2 Component Overview

| Component | Plugin Name | Description |
|-----------|-------------|-------------|
| Level-1 Pass | `vaip-pass_level1_rocm` | Orchestrates sub-passes, builds subgraph |
| Level-2 Conv Pass | `vaip-pass_level2_rocm_conv` | Conv pattern matching |
| Level-2 Gemm Pass | `vaip-pass_level2_rocm_gemm` | Gemm pattern matching |
| Custom Op | `custom-op-rocm` | Subgraph runtime execution |

---

## 3. Project Structure

```
morphizen-rocm/
├── .clinerules                           # Cline development rules
├── .gitignore                            # Git ignore patterns
├── build.bat                             # Windows build script
├── CMakeLists.txt                        # Root CMake configuration
├── LICENSE                               # MIT License
├── README.md                             # Project overview
│
├── cmake/
│   ├── deps.cmake                        # MorphiZen dependency fetch
│   └── presets.cmake                     # CMake presets generation
│
├── proto/
│   ├── CMakeLists.txt                    # Protobuf build config
│   └── rocm.proto                        # Message definitions
│
├── patterns/
│   ├── conv.json                         # Conv pattern definition
│   └── gemm.json                         # Gemm pattern definition
│
├── level-1-pass-rocm/                    # → vaip-pass_level1_rocm.dll
│   ├── CMakeLists.txt
│   └── src/
│       └── pass_main.cpp                 # Level-1 orchestrator
│
├── level-2-pass-rocm-conv/               # → vaip-pass_level2_rocm_conv.dll
│   ├── CMakeLists.txt
│   ├── cmake/
│   │   └── generate_pattern_inc.cmake
│   └── src/
│       └── pass_main.cpp                 # Conv pattern matching
│
├── level-2-pass-rocm-gemm/               # → vaip-pass_level2_rocm_gemm.dll
│   ├── CMakeLists.txt
│   ├── cmake/
│   │   └── generate_pattern_inc.cmake
│   └── src/
│       └── pass_main.cpp                 # Gemm pattern matching
│
├── custom-op-rocm/                       # → custom-op-rocm.dll
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp                      # Custom op registration
│       ├── custom_op.hpp                 # Custom op interface
│       └── custom_op.cpp                 # Subgraph execution
│
├── test/
│   ├── CMakeLists.txt
│   ├── test_conv.cpp                     # Conv unit tests
│   ├── test_gemm.cpp                     # Gemm unit tests
│   ├── gen_conv_model.py                 # Conv model generator
│   ├── gen_gemm_model.py                 # Gemm model generator
│   └── run_test_with_therock.bat         # Test runner script
│
├── etc/
│   └── vaip_config.json                  # Pass configuration
│
├── doc/
│   ├── 01_DESIGN.md                      # This document
│   ├── 02_LEVEL1_PASS_DESIGN.md          # Level-1 pass details
│   ├── 03_GROUPING_ALGORITHM.md          # Union-Find grouping
│   └── ...                               # Other documentation
│
└── tools/
    └── initialize-cmake-preset.py        # CMake preset generator
```

---

## 4. Pass Architecture

### 4.1 Level-1 Pass: Orchestrator

The Level-1 pass (`vaip-pass_level1_rocm`) serves as the main entry point and orchestrator:

1. Checks AMD GPU availability
2. Runs Level-2 sub-passes for pattern matching
3. Groups consecutive ROCm nodes using Union-Find algorithm
4. Builds `RocmSubgraphProto` with complete topology
5. Creates merged fused nodes in the original graph

### 4.2 Level-2 Passes: Pattern Matching

Each Level-2 pass handles pattern matching for a specific operation type:

- **Conv Pass**: Matches Conv patterns, extracts weights, creates `RocmParamProto`
- **Gemm Pass**: Matches Gemm patterns, extracts weights, creates `RocmParamProto`

### 4.3 Pass Execution Order

The `vaip_config.json` defines the execution order:

```json
{
  "passes": [
    {
      "name": "init",
      "plugin": "vaip-pass_init"
    },
    {
      "name": "fuse_ROCm",
      "plugin": "vaip-pass_level1_rocm",
      "pass_generic_param": "{\"sub_pass_names\": [\"vaip-pass_level2_rocm_conv\", \"vaip-pass_level2_rocm_gemm\"]}"
    }
  ]
}
```

---

## 5. Shared HIP Context

### 5.1 Design Goals

- **Per-session HIP stream** for all operations within a session
- **Lazy initialization** (only create resources when needed)
- **Session-scoped lifetime** (resources cleaned up when session ends)
- **Multi-session support** (each ORT session gets its own context)

### 5.2 Implementation

Each ORT session gets its own `HipContext` via `SessionContextRegistry`:

```cpp
class HipContext {
public:
  HipContext() {
    hipStreamCreate(&stream_);
    miopenCreate(&miopen_handle_);
    miopenSetStream(miopen_handle_, stream_);
    hipblasLtCreate(&hipblaslt_handle_);
  }
  
  ~HipContext() {
    hipblasLtDestroy(hipblaslt_handle_);
    miopenDestroy(miopen_handle_);
    hipStreamDestroy(stream_);
  }
  
  hipStream_t stream() { return stream_; }
  miopenHandle_t miopen_handle() { return miopen_handle_; }
  hipblasLtHandle_t hipblaslt_handle() { return hipblaslt_handle_; }

private:
  hipStream_t stream_ = nullptr;
  miopenHandle_t miopen_handle_ = nullptr;
  hipblasLtHandle_t hipblaslt_handle_ = nullptr;
};

// Each RocmCustomOp gets a shared_ptr to its session's HipContext
class RocmCustomOp {
  std::shared_ptr<HipContext> hip_context_;
};
```

> **Note:** For detailed resource lifecycle and multi-session architecture, see [08_ROCM_RESOURCE_MANAGEMENT.md](08_ROCM_RESOURCE_MANAGEMENT.md).

### 5.3 Implicit Fusion via Shared Stream

```
Subgraph: Input → Conv1 → Conv2 → Gemm → Output

Runtime execution on shared stream:
┌─────────────────────────────────────────────────────────────────────────┐
│                        HIP Stream Timeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [H2D Input] → [Conv1] → [Conv2] → [Gemm] → [D2H Output]               │
│                    │          │        │                                │
│                    └──────────┴────────┘                                │
│                    Intermediates stay on GPU                            │
│                                                                         │
│  ◀────────────────── Same stream = auto-sequenced ─────────────────────▶│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Proto Definitions (Subgraph Representation)

### 6.1 Design Philosophy

The proto design follows these principles:

1. **Explicit Topology**: The subgraph structure is explicitly represented, not implicit
2. **Type-Safe References**: Use `oneof` instead of sentinel values like `-1`
3. **Async-Ready**: External output mappings enable overlapped D2H transfers
4. **Self-Documenting**: Message structure clearly conveys intent

### 6.2 Core Messages

The subgraph representation uses four core protobuf messages that work together to describe the execution topology:

| Message | Purpose |
|---------|---------|
| `RocmSubgraphProto` | The complete subgraph container |
| `RocmNodeProto` | Represents a single operation node in the subgraph |
| `TensorRefProto` | Tracks where a tensor comes from (external input or another node's output) |
| `ExternalOutputProto` | Maps subgraph outputs back to ORT output tensors |

These messages form a graph structure where:
- `RocmSubgraphProto` contains a list of `RocmNodeProto` nodes in topological order
- Each `RocmNodeProto` uses `TensorRefProto` to reference its inputs
- `ExternalOutputProto` identifies which node outputs should be copied back to the host

#### RocmSubgraphProto - Complete Subgraph

```protobuf
// Represents a ROCm subgraph to be executed as a single fused operation
message RocmSubgraphProto {
  repeated RocmNodeProto nodes = 1;           // Nodes in topological order
  repeated ExternalOutputProto outputs = 2;  // External outputs with sources
}
```

**Why this design?**
- Top-level container that the custom op receives and executes
- `nodes` in topological order ensures correct execution sequence
- `outputs` list enables async D2H scheduling

#### RocmNodeProto - Operation Node

```protobuf
// A node in the subgraph execution plan
message RocmNodeProto {
  int32 node_id = 1;                      // Unique ID (0-indexed, topological order)
  RocmParamProto params = 2;              // Operation parameters (conv/gemm)
  repeated TensorRefProto inputs = 3;     // Input tensor references
  repeated string output_names = 4;       // Output names (for debugging)
}
```

**Why this design?**
- `node_id` enables references between nodes
- `inputs` tracks exactly where each input comes from
- `params` contains full operation parameters (shapes, weights, etc.)

#### TensorRefProto - Tensor Source Tracking

```protobuf
// Reference to another node's output within the subgraph
message InternalTensorRefProto {
  int32 producer_node_id = 1;  // Which node produces this tensor (0-indexed)
  int32 output_index = 2;      // Which output of that node (usually 0)
}

// Represents a tensor reference - either external (from ORT) or internal
message TensorRefProto {
  oneof source {
    string external_name = 1;          // External input from ORT context
    InternalTensorRefProto internal = 2; // From another node in subgraph
  }
}
```

**Why this design?**
- `oneof` makes it explicit that a tensor comes from exactly one source
- No magic sentinel values (like `-1` for external)
- Easy to extend (could add `constant` source type later)
- Clean C++ code: `if (ref.has_external_name()) {...} else {...}`

#### ExternalOutputProto - Output Mapping

```protobuf
// Mapping of an external output to its source within the subgraph
message ExternalOutputProto {
  string name = 1;                 // ORT output tensor name
  int32 producer_node_id = 2;      // Which node produces this
  int32 output_index = 3;          // Which output of that node
}
```

**Why this design?**
- Maps ORT outputs back to their producing nodes
- Enables async D2H scheduling: when node N completes, immediately copy its external outputs
- Allows overlapping D2H with subsequent GPU kernels

### 6.3 Subgraph Example

Consider this subgraph: `X → Conv1 → Conv2 → Y`

```protobuf
RocmSubgraphProto {
  nodes: [
    RocmNodeProto {
      node_id: 0
      params: { op_type: "conv", conv_params: {...} }
      inputs: [
        TensorRefProto { external_name: "X" }  // External input
      ]
      output_names: ["conv1_out"]
    },
    RocmNodeProto {
      node_id: 1
      params: { op_type: "conv", conv_params: {...} }
      inputs: [
        TensorRefProto { internal: { producer_node_id: 0, output_index: 0 } }  // From Conv1
      ]
      output_names: ["conv2_out"]
    }
  ]
  outputs: [
    ExternalOutputProto { name: "Y", producer_node_id: 1, output_index: 0 }
  ]
}
```

### 6.4 Async Execution Pipeline

The `ExternalOutputProto` mapping enables this optimization:

```
If Conv2's output is ALSO an external output (branching case):

Time →
┌───────────────────────────────────────────────────────────────────────────┐
│ H2D(X) │ Conv1 │ Conv2 │ D2H(conv2_out) │ Gemm │ D2H(gemm_out) │ sync    │
└───────────────────────────────────────────────────────────────────────────┘
                         ↑                        ↑
                         └── overlapped with Gemm!
```

The custom op can issue `hipMemcpyAsync` for `conv2_out` immediately after Conv2 completes, while Gemm executes on the same stream.

### 6.5 Operation Parameters

Individual operations use `RocmParamProto`:

```protobuf
message RocmParamProto {
  string op_type = 1;              // "conv" or "gemm"
  ConvParamProto conv_params = 2;  // Populated if op_type == "conv"
  GemmParamProto gemm_params = 3;  // Populated if op_type == "gemm"
}
```

See [proto/rocm.proto](../proto/rocm.proto) for complete definitions of `ConvParamProto` and `GemmParamProto`.

---

## 7. Custom Op Implementation

### 7.1 Subgraph Executor

The custom op receives a `RocmSubgraphProto` and executes all nodes:

```cpp
void RocmCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  // 1. Upload external inputs to GPU (async)
  for (const auto& node : subgraph_.nodes()) {
    for (const auto& input : node.inputs()) {
      if (input.has_external_name()) {
        UploadExternalInput(input.external_name(), context);
      }
    }
  }
  
  // 2. Execute nodes in topological order
  for (const auto& node : subgraph_.nodes()) {
    ExecuteNode(node);
    
    // 3. Check if this node produces any external outputs
    for (const auto& output : subgraph_.outputs()) {
      if (output.producer_node_id() == node.node_id()) {
        // Schedule async D2H copy
        ScheduleOutputCopy(output, context);
      }
    }
  }
  
  // 4. Synchronize stream with timeout
  SyncStreamWithTimeout();
}
```

### 7.2 Memory Management

- **External inputs**: Copied from ORT to GPU (H2D)
- **Intermediate tensors**: Allocated on GPU, reused between nodes
- **External outputs**: Copied from GPU to ORT (D2H)
- **Weights**: Loaded once at construction, cached on GPU

---

## 8. Pattern Definitions

### 8.1 Conv Pattern (patterns/conv.json)

```json
{
  "patterns": [
    {"id": "input_X", "wildcard": {}},
    {"id": "input_W", "wildcard": {}},
    {"id": "input_B", "wildcard": {}},
    {
      "id": "output",
      "isRoot": true,
      "callNode": {
        "opType": "Conv",
        "args": [
          {"name": "input_X"},
          {"name": "input_W"},
          {"name": "input_B"}
        ],
        "optionalArgs": [false, false, true]
      }
    }
  ]
}
```

### 8.2 Gemm Pattern (patterns/gemm.json)

```json
{
  "patterns": [
    {"id": "input_A", "wildcard": {}},
    {"id": "input_B", "wildcard": {}},
    {"id": "input_C", "wildcard": {}},
    {
      "id": "output",
      "isRoot": true,
      "callNode": {
        "opType": "Gemm",
        "args": [
          {"name": "input_A"},
          {"name": "input_B"},
          {"name": "input_C"}
        ],
        "optionalArgs": [false, false, true]
      }
    }
  ]
}
```

---

## 9. Build System

### 9.1 Build Script (build.bat)

```batch
@echo off
REM Set TheRock environment
set THEROCK_DIST=C:\Develop\m\dist\therock
set HIP_PLATFORM=amd

REM Set up MSVC
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Configure and build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -B C:/Develop/m/build/morphizen-rocm -S .
cmake --build C:/Develop/m/build/morphizen-rocm
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

- **test_conv.cpp**: Tests MIOpen Conv execution
- **test_gemm.cpp**: Tests hipBLASLt Gemm execution
- **test_timeout.cpp**: Tests GPU timeout handling

### 10.2 Integration Tests

Test the full pipeline: model loading → pass execution → subgraph execution

### 10.3 Model Generators

Python scripts to generate test ONNX models:
- `gen_conv_model.py`: Creates Conv models
- `gen_gemm_model.py`: Creates Gemm models

---

## 11. Future Enhancements

### 11.1 Additional Operations

The architecture supports adding new operations:
1. Create new Level-2 pass (e.g., `level-2-pass-rocm-pool`)
2. Add pattern file (e.g., `patterns/pool.json`)
3. Add execution logic in custom op
4. Update proto with new message type

### 11.2 Performance Optimizations

- Algorithm caching across runs
- Workspace sharing between operations
- Memory pool for intermediate tensors

---

## 12. References

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

*Document Version: 2.0*
*Last Updated: January 2026*
