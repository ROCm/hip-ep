<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# MorphiZen Architecture

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture Overview](#2-architecture-overview)
- [3. Component Catalog](#3-component-catalog)
  - [3.1 Core Components](#31-core-components)
  - [3.2 Foundation Libraries](#32-foundation-libraries)
  - [3.3 Backend Implementations](#33-backend-implementations)
  - [3.4 Tools and Executables](#34-tools-and-executables)
  - [3.5 Pass Plugins](#35-pass-plugins)
- [4. ONNX Runtime Integration](#4-onnx-runtime-integration)
- [5. Graph Optimization Pipeline](#5-graph-optimization-pipeline)
- [6. Compilation Flow](#6-compilation-flow)
- [7. MLIR Backend](#7-mlir-backend)
- [8. Build System](#8-build-system)
- [9. Extension Points](#9-extension-points)
- [10. Appendices](#10-appendices)

---

## 1. Overview

### 1.1 Project Description

**MorphiZen** is a hardware-agnostic AI compiler framework that enables dynamic manipulation of ONNX graphs for optimization and transformation. It provides pattern matching, rewrite rules, and a pass-based architecture for implementing custom compiler passes targeting diverse hardware platforms.

**Key Capabilities:**
- ONNX Runtime Execution Provider for hardware acceleration
- Pattern-based graph optimization with 12 pattern types
- Pass framework for modular graph transformations
- Multiple backend support (ONNX IR, MLIR)
- Extensible plugin architecture

### 1.2 Target Audience

- **Compiler Developers**: Building optimization passes and transformations
- **Hardware Vendors**: Integrating custom accelerators with ONNX Runtime
- **ML Researchers**: Experimenting with graph optimizations and quantization
- **Platform Engineers**: Deploying optimized models on specialized hardware

---

## 2. Architecture Overview

### 2.1 System Layering

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Applications & Tools                               │
│  graph-opt, onnx-grep, pattern-gen, tar                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Framework Core (morphizen-core)                        │
│  - Pass management and execution                           │
│  - Graph compilation and optimization                      │
│  - Model and tensor handling                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Graph Manipulation & Pattern Matching             │
│  ┌─────────────────┐         ┌──────────────────────────┐  │
│  │ morphizen-graph │         │ morphizen-pattern        │  │
│  │ Graph wrappers  │         │ Pattern matching (12)    │  │
│  └─────────────────┘         └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Backend Abstraction (morphizen-ort-api-ext)      │
│  - 111 function pointers for graph operations             │
│  - Runtime backend selection                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Backend Implementations                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐     │
│  │ ONNX IR  │    │   MLIR   │    │  ORT Execution   │     │
│  │ Backend  │    │ Backend  │    │  Provider Bridge │     │
│  └──────────┘    └──────────┘    └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 0: Foundation Libraries                              │
│  morphizen-io, mem_binary, encryption, morphizen-utils         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

1. **Backend Independence**
   - All graph operations go through `MORPHIZEN_ORT_API` abstraction layer
   - Runtime backend selection via environment variables
   - Multiple backends can coexist (ONNX and MLIR)

2. **Modular Component Architecture**
   - Clear separation: graph wrappers → pattern matching → framework → backends
   - Each component has well-defined APIs and minimal dependencies
   - Optional features can be disabled for lightweight builds

3. **Pass-Based Transformation**
   - Pluggable passes implement `IPass` interface
   - Passes execute sequentially on graph copies
   - Configuration-driven via JSON

4. **Extensibility**
   - Custom passes via plugin mechanism
   - Pattern-based optimization rules
   - Custom operator registration
   - Backend plugin architecture

---

## 3. Component Catalog

### 3.1 Core Components

#### 3.1.1 morphizen-graph (Graph Wrappers)

**Location**: `morphizen-graph/`
**Purpose**: Type-safe C++ wrappers over MORPHIZEN_ORT_API for low-level graph operations

**Key Classes**:
- `Graph`, `GraphRef`, `GraphConstRef` - Graph manipulation
- `Node`, `NodeRef`, `NodeConstRef` - Node operations
- `NodeArg`, `NodeArgRef` - Tensor/argument handling
- `NodeInput` - Combines NodeArg with optional producer Node
- `NodeAttr`, `NodeAttributesBuilder` - Attribute handling

**Public Headers**:
```
include/morphizen/
├── graph.hpp          # Graph operations
├── node.hpp           # Node operations
├── node_arg.hpp       # Argument operations
├── node_attr.hpp      # Attribute handling
└── node_input.hpp     # Input abstraction
```

**Dependencies**: `morphizen-ort-api-ext` (PUBLIC), `glog::glog` (PRIVATE)
**Build Artifact**: Static library `morphizen-graph`

**Key Features**:
- ~4,000 lines of C++ code
- Single point of MORPHIZEN_ORT_API calls
- Backend-independent (works with any MORPHIZEN_ORT_API implementation)
- RAII and type-safe wrappers

**Example Usage**:
```cpp
#include <morphizen/graph.hpp>

// Get graph nodes
auto nodes = graph_get_nodes(graph);

// Access node properties
for (const auto* node : nodes) {
    std::string op_type = node_op_type(*node);
    auto inputs = node_get_inputs(*node);
    auto outputs = node_get_output_node_args(*node);
}
```

---

#### 3.1.2 morphizen-pattern (Pattern Matching)

**Location**: `morphizen-pattern/`
**Purpose**: Powerful pattern matching system for identifying and transforming subgraphs

**Key Classes**:
- `Pattern` - Base pattern class with 12 subtypes
- `PatternBuilder` - Declarative API for constructing patterns
- `Binder` - Captures matched nodes and values
- `RewriteRule` - Framework for applying graph transformations

**12 Pattern Types**:
1. `wildcard` - Matches any node
2. `node` - Matches specific operator type
3. `constant` - Matches constant initializers
4. `commutable_node` - Matches commutative ops (Add, Mul)
5. `or` - Matches one of several alternatives
6. `sequence` - Matches ordered sequence
7. `where` - Conditional matching with predicates
8. `graph_input` - Matches graph inputs
9. `graph_output` - Matches graph outputs
10. `node_output_arg` - Matches specific node outputs
11. `pattern_graph_input` - Graph input patterns
12. `pattern_graph_output` - Graph output patterns

**Public Headers**: `include/morphizen/pattern.hpp`

**Dependencies**:
- `morphizen::morphizen-graph` (PUBLIC)
- `protobuf::libprotobuf`, `glog::glog`, `Microsoft.GSL::GSL` (PRIVATE)

**Build Artifact**: Static library `morphizen-pattern`

**Configuration**: Can be disabled via `morphizen_ENABLE_PATTERN_MATCHING=OFF`

**Example Usage**:
```cpp
#include <morphizen/pattern.hpp>

// Define a Conv+Relu fusion pattern
PatternBuilder builder;
auto input = builder.wildcard();
auto weight = builder.constant();
auto conv = builder.node2("Conv", {input, weight});
auto relu = builder.node2("Relu", {conv});

auto pattern = relu;  // Root of pattern

// Match against graph
if (pattern->match(graph, node)) {
    Binder& binder = ...;
    // Access matched nodes for transformation
    const Node* conv_node = binder[conv.id()];
    const Node* relu_node = binder[relu.id()];
}
```

---

#### 3.1.3 morphizen-core (Framework Core)

**Location**: `morphizen-core/`
**Purpose**: Core framework providing pass system, model/graph interfaces, and ORT integration

**Key Components**:
- **morphizen-core-static**: Compilation models, pass execution, configuration
- **morphizen-core-dynamic**: Exported as `onnxruntime_vitisai_ep.dll` (execution provider)

**Public Headers**:
```
include/morphizen/
├── pass.hpp             # Pass framework base classes
├── pass_context.hpp     # Execution context
├── model.hpp            # Model handling
├── node_builder.hpp     # High-level node construction
├── rewrite_rule.hpp     # Rewrite rule framework
├── anchor_point.hpp     # Quantization anchor points
├── tensor_proto.hpp     # Tensor protocol buffers
├── config_reader.hpp    # Configuration reading
├── custom_op_imp.hpp    # Custom operator interface
└── op_invoker.hpp       # Operation invocation
```

**Dependencies (morphizen-core-static)**:
- PUBLIC: `onnxruntime::onnxruntime`, `protobuf::libprotobuf`, `morphizen-io`, `glog::glog`, `morphizen::encryption`, `morphizen::mem_binary`, `morphizen-utils`, `morphizen-ort-api-ext`, `morphizen::morphizen-graph`
- Conditional: `morphizen::morphizen-pattern` (if enabled)

**Build Artifacts**:
- Static library: `morphizen-core-static`
- Dynamic library: `onnxruntime_vitisai_ep.dll` (configurable name)

**Key Features**:
- Pass-based architecture for graph transformations
- Execution context management
- TAR-based model caching
- Protocol buffer configuration
- Platform-specific utilities (Windows/Linux)

---

### 3.2 Foundation Libraries

#### 3.2.1 morphizen-utils (Utility Library)

**Location**: `morphizen-utils/`
**Purpose**: General-purpose C++ utilities including environment configuration, weak references, and plugin loading

**Key Classes**:
- `ENV_PARAM` - Type-safe environment variable access
- `WeakSingleton<T>` - Singleton without preventing destruction
- `WeakStore<K,V>` - Key-value store with automatic cleanup
- `parse_value()` - String-to-type conversion

**Public Headers**:
```
include/morphizen-utils/
├── env_config.hpp        # Environment parameters
├── weak_refs.hpp         # Weak reference patterns
├── parse_value.hpp       # String parsing
├── morphizen_plugin.hpp  # Plugin loading
└── cleanup.hpp           # Resource cleanup
```

**Build Artifact**: Static library `morphizen-utils`

**Example Usage**:
```cpp
#include <morphizen-utils/env_config.hpp>

// Define environment parameter
DEF_ENV_PARAM(DEBUG_MODE, "0");

// Access value
int debug = ENV_PARAM(DEBUG_MODE);
```

---

#### 3.2.2 morphizen-ort-api-ext (ORT API Extension)

**Location**: `morphizen-ort-api-ext/`
**Purpose**: Defines MORPHIZEN_ORT_API interface - abstraction layer for graph operations across backends

**Key Interface**:
- Defines 111+ function pointers for graph operations
- Backend-agnostic abstraction layer
- Implemented by backends (onnx-ir-imp, mlir-imp, ORT native)

**Function Categories**:
| Category | Examples |
|----------|----------|
| Model Operations | `model_load`, `model_delete`, `model_clone`, `model_main_graph` |
| Graph Operations | `graph_get_name`, `graph_nodes_unsafe`, `graph_get_inputs`, `graph_fuse` |
| Node Operations | `node_get_name`, `node_op_type`, `node_get_inputs_unsafe` |
| Node Arguments | `node_arg_get_name`, `node_arg_get_shape`, `node_arg_set_shape` |
| Attributes | `attr_proto_new_*`, `attr_proto_get_*` (int, float, string, tensor) |
| Tensor Handling | `tensor_proto_new_*` (floats, ints, doubles, fp16, bf16, booleans) |

**Public Headers**: `include/morphizen/morphizen-ort-api-ext.hpp`

**Dependencies**: `onnxruntime::onnxruntime`, `Microsoft.GSL::GSL`, `glog::glog` (PUBLIC)

**Build Artifact**: Static library `morphizen-ort-api-ext`

---

#### 3.2.3 Other Foundation Libraries

**morphizen-io**
- **Purpose**: Stream I/O utilities
- **Dependencies**: `glog::glog`
- **Build Artifact**: Static library `morphizen-io`

**mem_binary**
- **Purpose**: Packs files directly into library at build time with optional compression
- **Features**: Binary embedding, zlib compression, runtime decompression
- **Dependencies**: Optional `ZLIB::ZLIB`
- **Build Artifact**: Static library `mem_binary`

**encryption**
- **Purpose**: Encryption/decryption functionality for models
- **Dependencies**: Optional `OpenSSL::SSL` and `OpenSSL::Crypto`
- **Build Artifact**: Static library `morphizen-encryption`

---

### 3.3 Backend Implementations

#### 3.3.1 ort-bridge (ORT Execution Provider Bridge)

**Location**: `ort-bridge/`
**Purpose**: Bridges MorphiZen to ORT as an execution provider

**Key Components**:
- `ort-bridge.cpp` - Main bridge implementation
- `morphizen-ep.hpp/.cpp` - Execution provider implementation
- `morphizen-ep-factory.hpp/.cpp` - EP factory
- `ir-converter.hpp/.cpp` - IR conversion
- `ort-graph-wrapper.hpp/.cpp` - ORT graph wrapping

**Entry Points**:
```cpp
// Modern ORT API
extern "C" OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories);
```

**Configuration**: Compile-time backend selection (ONNX or MLIR backend required)

---

#### 3.3.2 onnx-ir-imp (ONNX IR Backend)

**Location**: `onnx-ir-imp/`
**Purpose**: ONNX-based implementation of MORPHIZEN_ORT_API interface

**Key Components**:
- `model.hpp/.cpp` - ONNX model implementation
- `graph.hpp/.cpp` - Graph representation
- `node.hpp/.cpp` - Node operations
- `node-arg.hpp/.cpp` - NodeArg operations
- `morphizen-ort-api.cpp` - API implementation

**Dependencies**: `onnxruntime::onnxruntime`, `morphizen-ort-api-ext`, `onnx`, `onnx_proto`

**Build Artifact**: Static library `onnx-ir`

**Linking**: Linked to morphizen-core-dynamic with WHOLE_ARCHIVE

---

#### 3.3.3 mlir-imp (MLIR Backend)

**Location**: `mlir-imp/`
**Purpose**: MLIR-based implementation of MORPHIZEN_ORT_API interface

**Key Components**:
- `mlir-context-manager.hpp/.cpp` - MLIR context management
- `mlir-model.hpp/.cpp` - MLIR module representation
- `mlir-graph.hpp/.cpp` - Graph using MLIR blocks
- `mlir-node.hpp/.cpp` - MLIR operation representation
- `morphizen-ort-api.cpp` - API bridge implementation

**MLIR Representation**:
```
MLIRModel
  ├── mlir::ModuleOp (root)
  └── MLIRGraph (main_graph)
      ├── mlir::func::FuncOp
      └── mlir::Block (graph operations)
```

**Dependencies**: `MLIR`, `LLVM` libraries, `morphizen-ort-api-ext`, `morphizen-utils`

**Build Artifact**: Static library `mlir-imp`

**Status**: Partial implementation (placeholder for graph traversal)

---

### 3.4 Tools and Executables

#### 3.4.1 graph-opt (Graph Optimization Tool)

**Location**: `graph-opt/`
**Purpose**: Tool for developing and testing MorphiZen passes

**Usage**:
```bash
morphizen-graph-opt -i <input.onnx> -o <output.onnx> -t <output.txt> -p <pass1> [pass2...]
```

**Options**:
- `-i` Input ONNX model
- `-o` Output ONNX model
- `-t` Output text representation
- `-p` List of passes to apply

**Build Artifact**: Executable `morphizen-graph-opt`
**Requirements**: `morphizen_ENABLE_BOOST=ON`

---

#### 3.4.2 onnx-grep (Pattern Matching Debugger)

**Location**: `onnx-grep/`
**Purpose**: Debugging tool for testing pattern matching

**Usage**:
```bash
morphizen-onnx-grep -f <model.onnx> -p <pattern_file> [-n <node_arg>] [-v]
```

**Options**:
- `-f` ONNX model file
- `-p` Pattern file (JSON or Python)
- `-n` Specific node_arg to trace
- `-v` Verbose mode

**Build Artifact**: Executable `morphizen-onnx-grep`
**Requirements**: `morphizen_ENABLE_BOOST=ON`

---

#### 3.4.3 pattern-gen (Pattern Generation Tool)

**Location**: `pattern-gen/`
**Purpose**: Generate pattern definitions from sample ONNX models

**Usage**:
```bash
morphizen-pattern-gen -f <model.onnx> -i <input> -o <output> \
                      [-m <mermaid>] [-j <json>] [-c <cpp>]
```

**Build Artifact**: Executable `morphizen-pattern-gen`
**Requirements**: `morphizen_ENABLE_BOOST=ON`

---

#### 3.4.4 tar (TAR Utility)

**Location**: `tar/`
**Purpose**: Utility for handling TAR archives used in MorphiZen caching

**Build Artifact**: Executable `morphizen-tar`
**Requirements**: `morphizen_ENABLE_BOOST=ON`

---

### 3.5 Pass Plugins

#### 3.5.1 morphizen-pass-init (Initialization Pass)

**Location**: `morphizen-pass-init/`
**Purpose**: Basic initialization pass for graph setup and preparation

**Build Artifact**: Static library `morphizen-pass_init`
**Linking**: Linked with WHOLE_ARCHIVE to morphizen-core-dynamic

---

#### 3.5.2 custom-op-generic (Generic Custom Operators)

**Location**: `custom-op-generic/`
**Purpose**: Generic custom operator implementation framework

**Build Artifact**: Static library `morphizen-custom-op-generic`
**Linking**: Linked with WHOLE_ARCHIVE

---

## 4. ONNX Runtime Integration

### 4.1 Execution Provider Architecture

MorphiZen functions as an **ONNX Runtime Execution Provider (EP)** through a dual-layer architecture:

**Modern ORT Bridge (New API)**:
- Implements `OrtEpFactory` interface
- Provides `CreateEpFactories()` as main entry point
- Creates `MorphiZenEP` instances inheriting from `OrtEp`

**Legacy Provider (Backward Compatibility)**:
- Implements classic ORT provider registration
- Exports `initialize_onnxruntime_morphizen_ep()` and `compile_onnx_model_morphizen_ep_*` functions

### 4.2 Registration and Lifecycle

**Device Detection**:
- Vendor ID: 0x1022 (AMD)
- Device Type: NPU (Neural Processing Unit)
- Fallback: CPU (controlled by `MORPHIZEN_EP_ENABLE_CPU_DEVICE`)

**Factory Pattern**:
```
CreateEpFactories()
    ↓
MorphiZenEpFactory
    ├─ GetSupportedDevicesImpl() → Device enumeration
    ├─ CreateEpImpl() → MorphiZenEP instance
    └─ ValidateCompiledModelCompatibilityInfoImpl() → Compatibility check
```

**Execution Provider Methods**:
```cpp
class MorphiZenEP : public OrtEp {
    // Determine which nodes EP can handle
    GetCapabilityImpl(graph) → EpGraphSupportInfo

    // Compile selected nodes to EP format
    CompileImpl(graphs, fused_nodes) → NodeComputeInfos

    // Cleanup
    ReleaseNodeComputeInfosImpl()

    // Compatibility validation
    GetCompiledModelCompatibilityInfoImpl() → JSON info
};
```

### 4.3 Model Compilation Flow

```
User Code
    ↓
session_options.AppendExecutionProvider_VitisAI(options)
    ↓
CreateEpFactories() [ort-bridge]
    ↓
CreateEpImpl() → MorphiZenEP instance
    ↓
GetCapabilityImpl(graph)
    ├─ Setup backend API via setup_global_vaip_ort_api()
    ├─ Convert ORT graph to MorphiZen model via IRConverter
    ├─ Call compile_onnx_model_morphizen_ep_v4()
    ├─ Identify supported nodes via get_supported_nodes()
    └─ Add nodes to fuse via EpGraphSupportInfo_AddNodesToFuse()
    ↓
CompileImpl(graphs, fused_nodes, count)
    ├─ Create EP context nodes (if enabled)
    ├─ For each execution provider:
    │   ├─ CompileSubgraph()
    │   ├─ Update input/output indices
    │   └─ Register compute operations
    └─ Return node compute infos
```

### 4.4 Configuration Options

**Session Options** (from ORT):
- `ep.context_enable` - Enable EP context compilation (0/1)
- `ep.context_file_path` - Output file for EP context model
- `ep.context_embed_mode` - Embed context in model (0/1)
- `ep.share_ep_contexts` - Share contexts across sessions
- `ep_dynamic_options.workload_type` - "Default" or "Efficient"

**Provider Options**:
- Prefix: `ep.morphizenexecutionprovider.*`
- Examples: target device, optimization levels, cache settings

### 4.5 Custom Operation Registration

Two registration mechanisms:
1. **Legacy**: `vaip_op_def_info()` (deprecated, memory leak warning)
2. **Modern**: `morphizen_register_ops()` callback function

Supports multiple domains (e.g., "com.xilinx", "com.microsoft")

---

## 5. Graph Optimization Pipeline

### 5.1 Pass Framework Architecture

The compilation pipeline uses a sophisticated pass system:

```
Graph Input (ONNX)
    ↓
PassContext (Shared configuration/cache)
    ↓
IPass::create_passes() [Load plugin-based passes]
    ↓
IPass::run_passes() [Apply passes sequentially]
    ↓
Pattern Matching + Graph Transformation
    ↓
Optimized Graph Output
```

**Pass Execution**:
1. **Pass Creation**: `IPass::create_passes()` loads configured passes from `morphizen_config.json`
2. **Graph Traversal**: Each pass iterates through nodes via `graph_get_nodes()`
3. **Pattern Matching**: For each node, match against patterns using `Pattern::match()`
4. **Conditional Transformation**: If pattern matches and conditions met, apply transformation
5. **Graph Modification**: Use `graph_set_node_*()` functions to modify graph

### 5.2 Pattern Matching System

**Pattern Types and Usage**:

```cpp
// 1. Wildcard - matches any node
auto input = builder.wildcard();

// 2. Node - matches specific operator
auto conv = builder.node2("Conv", {input, weight});

// 3. Constant - matches constant initializers
auto weight = builder.constant();

// 4. Commutable - handles commutative ops
auto add = builder.commutable_node("Add", {a, b});

// 5. Or - matches one of several alternatives
auto activation = builder.or_({relu, sigmoid, tanh});

// 6. Where - conditional matching
auto filtered = builder.where(conv, [](const Node& n) {
    return node_get_attr_int(n, "group") == 1;
});

// 7. Sequence - ordered matching
auto seq = builder.sequence({conv, bn, relu});

// 8-10. Graph I/O
auto graph_in = builder.graph_input();
auto graph_out = builder.graph_output();
auto node_out = builder.node_output_arg(conv, 0);
```

**Pattern Matching Algorithm**:
1. **Output Iteration**: For each output of the node
2. **Cached Matching**: `match_cached()` implements recursive pattern matching
3. **Pattern-Specific Logic**: Each pattern type implements `match_uncached()`
4. **Immutable Map Storage**: Pattern state uses immutable maps for functional semantics

### 5.3 Transformation and Rewrite Rules

**BasicRule Structure**:
```cpp
struct BasicRule : public Rule {
    virtual bool action(Graph* graph, binder_t& binder) const {
        // Access matched nodes
        const Node* node1 = binder[pattern_id_1];
        const Node* node2 = binder[pattern_id_2];

        // Apply transformation (fusion, quantization, etc.)
        // Modify graph in-place
        return true;  // true = graph modified
    }
};
```

**Rule Application**:
- `BaseRule::apply(graph)` - Apply rule to entire graph
- `Rule::apply_once(graph, node)` - Apply rule to single node
- `RuleChain` - Chain multiple rules together

**Common Transformations**:
- Operator Fusion (Conv+Relu → ConvRelu)
- Quantization (Insert Q/DQ nodes)
- Peephole Optimizations
- Dead Code Elimination
- Constant Folding
- Custom Operator Mapping

---

## 6. Compilation Flow

### 6.1 End-to-End Pipeline

```
Step 1: Model Loading
    ONNX Model (.onnx)
        ↓
    Load via onnxruntime::Model
        ↓
    Create Graph wrapper (morphizen::Graph or MLIRModel)

Step 2: PassContext Setup
    Config → PassContext creation
        ├─ Cache directory initialization
        ├─ Provider options loading
        ├─ Session config parsing
        └─ Logging setup

Step 3: Graph Optimization Passes
    For each configured pass plugin:
        ↓
    IPass::create_pass(context, pass_proto)
        ↓
    Pass::execute()
        ├─ For each node in graph:
        │   ├─ Pattern matching
        │   ├─ Condition checking
        │   └─ Apply transformation
        └─ Graph modification (in-place)

Step 4: Graph Transformations
    Pattern-based transformations include:
        • Operator Fusion
        • Quantization
        • Peephole Optimizations
        • Dead Code Elimination
        • Constant Folding
        • Custom Operator Mapping

Step 5: MLIR Compilation (Optional)
    If MLIR backend enabled:
        ↓
    Convert optimized graph to MLIR
        ├─ Create MLIRModel
        ├─ Build func::FuncOp from nodes
        └─ Register dialects and transformations

Step 6: Execution Provider Selection
    compile_onnx_model_3_internal()
        ├─ Create ExecutionProvider instances
        ├─ Assign nodes to providers
        └─ Generate execution plan

Step 7: Output
    Optimized Model
        ├─ Modified ONNX graph
        ├─ Cache artifacts
        └─ Execution metadata
```

### 6.2 Key Compilation Functions

**compile_onnx_model_internal()**:
- Main compilation entry point
- Applies graph optimization passes sequentially
- Each pass can modify the graph
- Supports incremental compilation with caching

**compile_onnx_model_3_internal()**:
- Execution provider compilation
- Generates runtime execution plans
- Creates provider-specific kernels

### 6.3 Graph Modification Operations

Available graph operations during passes:
- Node Creation: `graph.add_node()`
- Node Removal: `graph.remove_node()`
- Edge Modification: `graph.add_edge()`, `remove_edge()`
- Constant Creation: `IPass::create_const()`
- Node Replacement: Replace node outputs with new operations
- Attribute Updates: `node_set_attr_*()`

---

## 7. MLIR Backend

### 7.1 MLIR Integration Architecture

**MLIR Model Representation**:
```
MLIRModel
    ├── mlir::ModuleOp (Root MLIR module)
    ├── MLIRGraph (main_graph function)
    │   ├── mlir::func::FuncOp
    │   ├── mlir::Block (Graph operations)
    │   └── Symbol Table
    └── Metadata (opset info, paths)
```

### 7.2 Key MLIR Classes

**MLIRModel** (`mlir-model.hpp/.cpp`):
- Owns `mlir::OwningOpRef<mlir::ModuleOp>` for lifetime management
- Factory methods: `create()`, `create_empty()`, `load()`
- Metadata storage: `set_metadata_prop()`, `get_metadata_prop()`
- Graph access: `main_graph()` returns `MLIRGraph&`

**MLIRGraph** (`mlir-graph.hpp/.cpp`):
- Represents ONNX graph as MLIR function
- Node operations: `add_node()`, `nodes_unsafe()`, `producer_node()`
- Value management: `node_arg_new()`, `get_node_arg_index()`
- I/O handling: `get_inputs()`, `get_outputs()`, `set_inputs()`, `set_outputs()`
- Persistence: `save()` for MLIR serialization

### 7.3 MLIR Context Management

**MLIRContextManager (Singleton)**:
```cpp
auto& context = MLIRContextManager::getInstance().getContext();
// Dialect loading happens automatically
// MLIR context is thread-safe singleton
```

### 7.4 MLIR-ORT API Bridge

The MLIR backend implements the same `VaipOrtApiExt` interface as ONNX-IR:
```
VaipOrtApiExt (111 function pointers)
    ↑
    ├─ onnx-ir-imp (ONNX Protocol Buffer backend)
    └─ mlir-imp (MLIR IR backend)
```

Both backends are interchangeable through runtime selection.

### 7.5 Backend Selection

**Compile-Time Default**: MLIR backend (or ONNX if MLIR disabled)
**Runtime Override**: `MORPHIZEN_ORT_BRIDGE_BACKEND` environment variable

---

## 8. Build System

### 8.1 CMake Configuration Options

| Option | Default | Purpose |
|--------|---------|---------|
| `morphizen_ENABLE_MORPHIZEN_CORE_DYNAMIC` | ON | Build dynamic execution provider DLL |
| `morphizen_ENABLE_PATTERN_MATCHING` | ON | Include pattern matching library |
| `morphizen_ENABLE_BOOST` | OFF | Enable Boost-dependent tools |
| `morphizen_ENABLE_ORT_BRIDGE` | ON | Build ORT bridge and EP |
| `morphizen_ENABLE_ONNX_BACKEND` | OFF | Use ONNX IR backend |
| `morphizen_ENABLE_MLIR_BACKEND` | ON | Use MLIR backend |
| `morphizen_ENABLE_ONNX_SCHEMA_SUPPORT` | OFF | Enable ONNX schema |
| `morphizen_ENABLE_UNIT_TEST` | ON | Build unit tests |

### 8.2 Build Settings

**Build Configuration**:
- Build Directory: `../../build/$(basename $PWD)`
- Install Prefix: `../../local`
- Build Type: Debug
- Shared Libraries: OFF (static linking)
- Runtime Library: Static `/MTd` (Debug) via `CMAKE_MSVC_RUNTIME_LIBRARY`

**Critical CMake Flags**:
```cmake
-DBUILD_SHARED_LIBS=OFF
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_PREFIX_PATH="<absolute-path-to-local>"
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
-Dmorphizen_ENABLE_UNIT_TEST=ON
```

### 8.3 Dependency Management

**Required Dependencies**:
- ONNX Runtime (must be built manually, cannot be auto-fetched)
- protobuf, gtest, glog (can be auto-fetched or pre-built)

**Optional Dependencies**:
- Boost (for tools: graph-opt, onnx-grep, pattern-gen, tar)
- MLIR/LLVM (for MLIR backend)
- OpenSSL (for encryption)
- Zlib (for mem_binary compression)

### 8.4 Build Artifacts

**Static Libraries**:
- `morphizen-graph`, `morphizen-pattern`, `morphizen-utils`
- `morphizen-ort-api-ext`, `morphizen-io`, `mem_binary`, `morphizen-encryption`
- `morphizen-core-static`
- `onnx-ir`, `mlir-imp`, `ort-bridge`

**Dynamic Libraries**:
- `onnxruntime_vitisai_ep.dll` (execution provider)

**Executables** (if Boost enabled):
- `morphizen-graph-opt`, `morphizen-onnx-grep`
- `morphizen-pattern-gen`, `morphizen-tar`

---

## 9. Extension Points

### 9.1 Custom Pass Development

**Creating a Custom Pass**:

1. **Implement IPass Interface**:
```cpp
class MyCustomPass : public IPass {
public:
    static std::unique_ptr<IPass> create(
        std::shared_ptr<PassContext> context,
        const PassProto& pass_proto);

    void run(Graph* graph) override {
        // Implement optimization logic
        for (const auto* node : graph_get_nodes(*graph)) {
            // Pattern matching and transformation
        }
    }
};
```

2. **Register Pass Plugin**:
```cpp
extern "C" void register_my_pass() {
    // Register pass with MorphiZen
}
```

3. **Configure in morphizen_config.json**:
```json
{
  "passes": [
    {
      "plugin": "my-custom-pass",
      "parameters": { ... }
    }
  ]
}
```

### 9.2 Pattern-Based Optimization

**Define Custom Patterns**:
```cpp
// JSON pattern definition
{
  "pattern": {
    "op_type": "Conv",
    "inputs": [
      {"type": "wildcard"},
      {"type": "constant"}
    ]
  },
  "action": "fuse_with_relu"
}
```

**C++ Pattern API**:
```cpp
PatternBuilder builder;
auto pattern = builder.node2("MyOp", {input1, input2})
    .where([](const Node& n) {
        // Custom predicate
        return check_attribute(n);
    })
    .build();
```

### 9.3 Custom Operators

**Register Custom Operators**:
```cpp
void morphizen_register_ops(
    std::vector<OrtCustomOpDomain*>& domains) {
    auto domain = Ort::CustomOpDomain("com.mycompany");
    domain.Add(new MyCustomOp());
    domains.push_back(domain.release());
}
```

### 9.4 Configuration and Environment Variables

**Key Environment Variables**:
- `MORPHIZEN_ORT_BRIDGE_BACKEND` - Backend selection ("onnx" or "mlir")
- `DEBUG_REWRITE_RULE` - Pattern matching debug level (0/1/2)
- `XLNX_ONNX_EP_VERBOSE` - Verbose logging (0/1/2)
- `MORPHIZEN_DEBUG_ORT_EP_API` - Debug EP API calls

---

## 10. Appendices

### 10.1 Directory Structure

```
morphizen/
├── 3rd-party/              # Third-party dependencies
│   ├── hash-library/       # Hash library (git submodule)
│   └── onnxruntime-morphizen-headers/  # ORT headers
├── cmake/                  # CMake configuration files
├── common/                 # Shared headers
├── custom-op-generic/      # Generic custom operators
├── docs/                   # Documentation
│   ├── architecture.md     # This document
│   └── developer-guide.md  # Build instructions
├── encryption/             # Encryption utilities
├── graph-opt/              # Graph optimization tool
├── mem_binary/             # Embedded resource management
├── mlir-imp/               # MLIR backend implementation
├── morphizen-graph/        # Graph wrapper library
├── morphizen-pattern/      # Pattern matching library
├── morphizen-utils/        # Utility library
├── onnx-grep/              # Pattern matching debugger
├── onnx-ir-imp/            # ONNX IR backend
├── ort-bridge/             # ORT execution provider bridge
├── pattern-gen/            # Pattern generation tool
├── tar/                    # TAR utility
├── unit-test/              # Unit tests
├── morphizen-core/         # Framework core
├── morphizen-ort-api-ext/  # ORT API extension
├── morphizen-io/           # I/O utilities
└── morphizen-pass-init/    # Initialization pass
```

### 10.2 Key File Locations

**Component Headers**:
- `morphizen-graph/include/morphizen/*.hpp` - Graph wrappers
- `morphizen-pattern/include/morphizen/pattern.hpp` - Pattern matching
- `morphizen-core/include/morphizen/*.hpp` - Framework APIs
- `morphizen-ort-api-ext/include/morphizen/morphizen-ort-api-ext.hpp` - API abstraction
- `3rd-party/onnxruntime-morphizen-headers/morphizen/morphizen_ort_api.h` - C interface

**Documentation**:
- `morphizen-graph/README.md` - Graph wrapper overview
- `morphizen-pattern/README.md` - Pattern system overview
- `mlir-imp/README.md` - MLIR backend overview
- `ort-bridge/doc/ORT-BRIDGE-DESIGN.md` - Type-safe index design
- `docs/developer-guide.md` - Build and development guide

**Build System**:
- `CMakeLists.txt` (root) - Top-level build configuration
- Component-specific `CMakeLists.txt` files

### 10.3 Glossary

**ONNX**: Open Neural Network Exchange - standard format for representing ML models

**ORT**: ONNX Runtime - inference engine for ONNX models

**EP**: Execution Provider - plugin interface for hardware acceleration in ORT

**MorphiZen**: Versatile AI Inference Pipeline - MorphiZen's internal framework name

**MORPHIZEN_ORT_API**: Function pointer interface abstracting graph operations (111 functions)

**Pass**: Modular graph transformation component implementing IPass interface

**Pattern**: Declarative specification of subgraph structure for matching

**Binder**: Object capturing matched nodes/values during pattern matching

**IR**: Intermediate Representation (ONNX or MLIR)

**MLIR**: Multi-Level Intermediate Representation - compiler infrastructure from LLVM

**DLL Safe**: Cross-DLL data passing mechanism for STL containers

### 10.4 Related Documentation

- **[Developer Guide](developer-guide.md)**: Build instructions and prerequisites
- **[Graph Wrapper README](../morphizen-graph/README.md)**: morphizen-graph component details
- **[Pattern README](../morphizen-pattern/README.md)**: Pattern matching system
- **[MLIR Backend README](../mlir-imp/README.md)**: MLIR backend implementation
- **[ORT Bridge Design](../ort-bridge/doc/ORT-BRIDGE-DESIGN.md)**: Type-safe index design

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Related PR**: #27 (Extract morphizen-graph and morphizen-pattern components)
