# HipDNN Graph Serialization Implementation Guide

**Version:** 1.0  
**Date:** January 9, 2026  
**Author:** Morphizen HipDNN Team

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Protocol Buffer Changes](#3-protocol-buffer-changes)
4. [Level-1 Pass Implementation](#4-level-1-pass-implementation)
5. [Custom Op Implementation](#5-custom-op-implementation)
6. [File I/O Helpers](#6-file-io-helpers)
7. [UID Extraction and Management](#7-uid-extraction-and-management)
8. [Complete Code Examples](#8-complete-code-examples)
9. [Testing Strategy](#9-testing-strategy)
10. [Migration Path](#10-migration-path)

---

## 1. Overview

### 1.1 Motivation

The current implementation passes Conv operation parameters (pads, strides, dilations, etc.) through the protocol buffer and rebuilds the hipDNN graph in the custom op. This approach has limitations:

- **Complexity**: Need to maintain parameter passing and graph reconstruction logic
- **Pattern Dependency**: Requires pattern matching for each operation type
- **Duplication**: Graph construction logic exists in both pass and custom op
- **Scalability**: Adding new operations requires modifying both components

### 1.2 Proposed Solution

**Serialize the hipDNN graph in the pass, deserialize and compile in the custom op.**

This approach:
- ✅ Simplifies the architecture - only filename is passed
- ✅ Eliminates pattern matching - just check op type
- ✅ Centralizes graph creation logic in the pass
- ✅ Enables debugging - serialized graphs can be inspected
- ✅ Scales easily - same mechanism for all operations

### 1.3 Key Insight from hipDNN Frontend

The `Graph` class internally uses **backend descriptors** for all operations:

```cpp
// Graph.hpp internal implementation
std::unique_ptr<ScopedHipdnnBackendDescriptor> _graphDesc;
std::unique_ptr<ScopedHipdnnBackendDescriptor> _executionPlanDesc;

Error build_operation_graph(hipdnnHandle_t handle) {
    auto serializedGraph = buildFlatbufferOperationGraph();
    _graphDesc = std::make_unique<ScopedHipdnnBackendDescriptor>(
        serializedGraph.data(), serializedGraph.size());
    // ... set handle, finalize ...
}
```

**We can use the same approach** - load serialized data into descriptors and compile directly!

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         ONNX Graph                              │
│                    (with Conv operations)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Level-1 Pass (Compiler Time)                 │
├─────────────────────────────────────────────────────────────────┤
│  1. Identify Conv nodes (no pattern matching needed)            │
│  2. Build hipDNN Graph from Conv node attributes                │
│  3. Serialize graph using buildFlatbufferOperationGraph()       │
│  4. Save to file: hipdnn_graph_{output_name}.bin               │
│  5. Pass filename via proto parameter                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Fused Node (HIPDNN Custom Op)                   │
│                   graph_file_name = "hipdnn_graph_X.bin"        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Custom Op Constructor (Load Time)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Read graph_file_name from proto                             │
│  2. Load serialized graph from file                             │
│  3. Create graphDesc from buffer (ScopedHipdnnBackendDescriptor)│
│  4. Set handle and finalize descriptor                          │
│  5. Run compilation pipeline:                                   │
│     - initializeHeuristicDescriptor()                          │
│     - initializeEngineConfig()                                 │
│     - create execution plan                                    │
│     - get workspace size                                       │
│  6. Extract UIDs using GraphWrapper for variant pack           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                Custom Op Compute (Inference Time)               │
├─────────────────────────────────────────────────────────────────┤
│  1. Build variant pack (UID → tensor pointer mapping)          │
│  2. Create variant pack descriptor                              │
│  3. Execute: backendExecute(handle, plan, variantPack)         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Key APIs Used |
|-----------|---------------|---------------|
| **Proto** | Define interface (filename only) | google::protobuf |
| **Level-1 Pass** | Build & serialize graphs | hipdnn_frontend::graph::Graph |
| **Custom Op** | Load, compile, & execute | hipdnnBackend() APIs, GraphWrapper |

---

## 3. Protocol Buffer Changes

### 3.1 New Proto Definition

**File:** `proto/hipdnn.proto`

```protobuf
// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
syntax = "proto3";
package hipdnn;

message HipdnnParamProto {
  // File path to the serialized hipDNN graph
  string graph_file_name = 1;
}
```

### 3.2 Rationale

- **Single Parameter**: Only the filename is needed
- **Simple**: No need to pass operation-specific parameters
- **Extensible**: Same mechanism works for all operation types
- **Debuggable**: Graph files can be inspected offline

---

## 4. Level-1 Pass Implementation

The pass identifies Conv nodes, builds hipDNN graphs, serializes them, and creates fused nodes without pattern matching.

---

## 5. Custom Op Implementation

The custom op loads the serialized graph, compiles it using backend descriptors, and executes it.

---

## Appendix A: Key References

- **hipDNN Frontend**: `c:/Develop/TheRock/include/hipdnn/frontend/hipdnn_frontend/Graph.hpp`
- **GraphWrapper**: `c:/Develop/TheRock/include/hipdnn/data_sdk/hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp`
- **hipDNNEP Kernel**: `external/hipDNNEP/src/kernel.cc`
- **hipDNNEP EP**: `external/hipDNNEP/src/ep.cc`

## Appendix B: Glossary

- **UID**: Unique Identifier for tensors in hipDNN graph
- **Variant Pack**: Mapping of UIDs to tensor data pointers for execution
- **Backend Descriptor**: Low-level hipDNN descriptor objects
- **ScopedHipdnnBackendDescriptor**: RAII wrapper for backend descriptors
- **GraphWrapper**: Utility to read FlatBuffer serialized graphs

---

**End of Document**
