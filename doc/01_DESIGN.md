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

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
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
│  │  │  • ConvExecutor (MIOpen)                                  │   │   │
│  │  │  • GemmExecutor (hipBLASLt)                               │   │   │
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
```

### 2.2 Component Overview

| Component | Plugin Name | Description |
|-----------|-------------|-------------|
| Level-1 Pass | `vaip-pass_level1_rocm` | Orchestrates sub-passes, GPU checks |
| Level-2 Conv Pass | `vaip-pass_level2_rocm_conv` | Conv pattern matching |
| Level-2 Gemm Pass | `vaip-pass_level2_rocm_gemm` | Gemm pattern matching |
| Custom Op | `custom-op-rocm` | Runtime execution |

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
│       ├── custom_op.cpp                 # Custom op implementation
│       ├── hip_context.hpp               # Shared HIP context singleton
│       ├── conv_executor.hpp             # MIOpen Conv executor
│       ├── conv_executor.cpp
│       ├── gemm_executor.hpp             # hipBLASLt Gemm executor
│       └── gemm_executor.cpp
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
│   ├── 02_BUILD.md                       # Build instructions
│   └── 03_API_REFERENCE.md               # API reference
│
└── tools/
    └── initialize-cmake-preset.py        # CMake preset generator
```

---

## 4. Pass Architecture

### 4.1 Level-1 Pass: Orchestrator

The Level-1 pass (`vaip-pass_level1_rocm`) serves as the main entry point and orchestrator:

```cpp
// level-1-pass-rocm/src/pass_main.cpp
struct Level1Rocm {
  Level1Rocm(IPass& self) : self_{self} {}
  
  void process_run_subpasses(Graph& graph) {
    auto& pass_proto = self_.get_pass_proto();
    
    // Dynamically create Level-2 passes from config
    all_passes_ = IPass::create_passes(
        self_.get_context(),
        pass_proto.pass_rocm_param().sub_pass());
    
    // Run all Level-2 passes (Conv, Gemm)
    IPass::run_passes(all_passes_, graph);
  }
  
  void process(IPass& self, Graph& graph) {
    // 1. Check AMD GPU availability
    int device_count;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
      LOG(WARNING) << "[ROCm EP] No AMD GPU found, skipping";
      return;
    }
    
    // 2. Log device info
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    LOG(INFO) << "[ROCm EP] Using device: " << props.name;
    
    // 3. Run Level-2 sub-passes
    process_run_subpasses(graph);
  }
  
  IPass& self_;
  std::vector<std::shared_ptr<IPass>> all_passes_;
};

DEFINE_VAIP_PASS(Level1Rocm, vaip_pass_level1_rocm)
```

### 4.2 Level-2 Passes: Pattern Matching

Each Level-2 pass handles pattern matching for a specific operation type:

#### Conv Pass (Level-2)

```cpp
// level-2-pass-rocm-conv/src/pass_main.cpp
struct Level2RocmConv {
  Level2RocmConv(IPass& self) : self_{self} {}
  
  std::unique_ptr<Rule> create_rule(IPass* self) {
    auto pattern = PatternBuilder().create_by_json(
        std::string((const char*)conv_json));
    
    return Rule::create_rule(pattern, [=](Graph* graph, binder_t& binder) {
      // Match Conv pattern
      auto input_x = binder["input_X"];
      auto input_w = binder["input_W"];
      auto output = binder["output"];
      bool has_bias = binder["input_B"].node_arg != nullptr;
      
      // Build RocmParamProto with op_type = "conv"
      auto rocm_param = rocm::RocmParamProto();
      rocm_param.set_op_type("conv");
      // ... populate conv_params
      
      // Fuse the pattern
      auto [meta_def, error] = self->try_fuse(...);
      if (meta_def) {
        self->fuse(*graph, std::move(*meta_def));
        return true;
      }
      return false;
    });
  }
  
  void process(IPass& self, Graph& graph) {
    create_rule(&self)->apply(&graph);
  }
  
  IPass& self_;
};

DEFINE_VAIP_PASS(Level2RocmConv, vaip_pass_level2_rocm_conv)
```

#### Gemm Pass (Level-2)

```cpp
// level-2-pass-rocm-gemm/src/pass_main.cpp
struct Level2RocmGemm {
  // Similar structure, matches Gemm pattern
  // Sets op_type = "gemm" in RocmParamProto
};

DEFINE_VAIP_PASS(Level2RocmGemm, vaip_pass_level2_rocm_gemm)
```

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
      "passRocmParam": {
        "subPass": [
          {
            "name": "rocm_conv",
            "plugin": "vaip-pass_level2_rocm_conv"
          },
          {
            "name": "rocm_gemm",
            "plugin": "vaip-pass_level2_rocm_gemm"
          }
        ],
        "maxWorkspaceSize": 67108864,
        "enableExhaustiveSearch": false
      }
    }
  ],
  "target": "ROCm_default",
  "targets": [
    {
      "name": "ROCm_default",
      "pass": ["init", "fuse_ROCm"]
    }
  ]
}
```

---

## 5. Shared HIP Context

### 5.1 Design Goals

- **Single HIP stream** for all operations
- **Lazy initialization** (only create resources when needed)
- **Thread-safe** singleton pattern
- **Automatic cleanup** on destruction

### 5.2 Implementation

```cpp
// custom-op-rocm/src/hip_context.hpp
#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>

namespace rocm_ep {

class HipContext {
public:
  static HipContext& instance() {
    static HipContext ctx;
    return ctx;
  }
  
  // Delete copy/move constructors
  HipContext(const HipContext&) = delete;
  HipContext& operator=(const HipContext&) = delete;
  
  // Accessors
  hipStream_t stream() {
    ensure_initialized();
    return stream_;
  }
  
  miopenHandle_t miopen_handle() {
    ensure_initialized();
    return miopen_handle_;
  }
  
  hipblasLtHandle_t hipblaslt_handle() {
    ensure_initialized();
    return hipblaslt_handle_;
  }

private:
  HipContext() = default;
  
  ~HipContext() {
    if (initialized_) {
      hipblasLtDestroy(hipblaslt_handle_);
      miopenDestroy(miopen_handle_);
      hipStreamDestroy(stream_);
    }
  }
  
  void ensure_initialized() {
    std::call_once(init_flag_, [this]() {
      // Create shared stream
      hipStreamCreate(&stream_);
      
      // Initialize MIOpen with shared stream
      miopenCreate(&miopen_handle_);
      miopenSetStream(miopen_handle_, stream_);
      
      // Initialize hipBLASLt (stream passed per-call)
      hipblasLtCreate(&hipblaslt_handle_);
      
      initialized_ = true;
    });
  }
  
  std::once_flag init_flag_;
  bool initialized_ = false;
  hipStream_t stream_ = nullptr;
  miopenHandle_t miopen_handle_ = nullptr;
  hipblasLtHandle_t hipblaslt_handle_ = nullptr;
};

} // namespace rocm_ep
```

### 5.3 Implicit Fusion via Shared Stream

```
Graph: Input → Conv → Reshape → Gemm → Output

Runtime execution order on shared stream:
┌─────────────────────────────────────────────────────────────────────┐
│                        HIP Stream Timeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Conv Kernel]  ────────▶  [Reshape (CPU)]  ────────▶  [Gemm Kernel]│
│       │                                                      │      │
│       └─── miopenConvolutionForward(stream)                  │      │
│                                                              │      │
│                                     hipblasLtMatmul(stream) ─┘      │
│                                                                     │
│  ◀────────────────── Same stream = auto-sequenced ─────────────────▶│
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: No explicit `hipStreamSynchronize()` needed between operations on the same stream. The GPU automatically sequences operations in submission order.

---

## 6. Proto Definitions

### 6.1 rocm.proto

```protobuf
// proto/rocm.proto
syntax = "proto3";
package rocm;

import "vaip_core.proto";

// Pass-level configuration
message RocmPassParamProto {
  // Sub-passes to run (Conv, Gemm, etc.)
  repeated vaip_core.PassProto sub_pass = 1;
  
  // Shared configuration
  int64 max_workspace_size = 2;
  bool enable_exhaustive_search = 3;
}

// Convolution parameters (for MIOpen)
message ConvParamProto {
  // Input tensor dimensions [N, C, H, W]
  int64 batch_size = 1;
  int64 in_channels = 2;
  int64 in_height = 3;
  int64 in_width = 4;
  
  // Filter dimensions [K, C, R, S]
  int64 out_channels = 5;
  int64 filter_height = 6;
  int64 filter_width = 7;
  
  // Convolution parameters
  int32 pad_h = 8;
  int32 pad_w = 9;
  int32 stride_h = 10;
  int32 stride_w = 11;
  int32 dilation_h = 12;
  int32 dilation_w = 13;
  int32 group_count = 14;
  
  // Scalars
  float alpha = 15;
  float beta = 16;
  
  // Options
  bool has_bias = 17;
  int32 algorithm_index = 18;
  bool exhaustive_search = 19;
  
  // Data types
  string data_type_x = 20;
  string data_type_w = 21;
  string data_type_y = 22;
  
  // Input/output names
  repeated string input_names = 23;
  repeated string output_names = 24;
  
  // EP context
  string ep_context_file_name = 25;
  int64 ep_context_file_size = 26;
}

// GEMM parameters (for hipBLASLt)
message GemmParamProto {
  // Matrix dimensions
  int64 m = 1;
  int64 n = 2;
  int64 k = 3;
  int64 batch_count = 4;
  
  // Leading dimensions
  int64 lda = 5;
  int64 ldb = 6;
  int64 ldc = 7;
  int64 ldd = 8;
  
  // Transpose operations
  int32 trans_a = 9;
  int32 trans_b = 10;
  
  // Scalars
  float alpha = 11;
  float beta = 12;
  
  // Epilogue (bias, relu, gelu)
  string epilogue = 13;
  
  // Data types
  string data_type_a = 14;
  string data_type_b = 15;
  string data_type_c = 16;
  string data_type_d = 17;
  string compute_type = 18;
  
  // Options
  int32 algorithm_index = 19;
  int64 max_workspace_size = 20;
  
  // Input/output names
  repeated string input_names = 21;
  repeated string output_names = 22;
  
  // EP context
  string ep_context_file_name = 23;
  int64 ep_context_file_size = 24;
}

// Custom op parameters (stored in MetaDef)
message RocmParamProto {
  // Operation type: "conv" or "gemm"
  string op_type = 1;
  
  // Operation-specific parameters
  ConvParamProto conv_params = 2;
  GemmParamProto gemm_params = 3;
  
  // General metadata
  string ep_context_file_name = 4;
  int64 ep_context_file_size = 5;
}
```

---

## 7. Custom Op Implementation

### 7.1 Custom Op Class

```cpp
// custom-op-rocm/src/custom_op.hpp
#pragma once

#include "rocm.pb.h"
#include "hip_context.hpp"
#include "morphizen/vaip.hpp"

namespace rocm_ep {

class RocmCustomOp : public vaip_core::CustomOpImp {
public:
  RocmCustomOp(std::shared_ptr<const vaip_core::PassContext> context,
               const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
               onnxruntime::Model* model);
  
  virtual ~RocmCustomOp();

private:
  void Compute(const OrtApi* api, OrtKernelContext* context) const override;
  
  // Route to appropriate executor based on op_type
  void ExecuteConv(const OrtApi* api, OrtKernelContext* context) const;
  void ExecuteGemm(const OrtApi* api, OrtKernelContext* context) const;

private:
  rocm::RocmParamProto rocm_proto_;
  
  // Cached weights from model
  std::vector<uint8_t> weight_data_;
  std::vector<int64_t> weight_shape_;
  std::vector<uint8_t> bias_data_;
  std::vector<int64_t> bias_shape_;
  bool has_bias_ = false;
};

} // namespace rocm_ep
```

### 7.2 Compute Routing

```cpp
// custom-op-rocm/src/custom_op.cpp
void RocmCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  const auto& op_type = rocm_proto_.op_type();
  
  if (op_type == "conv") {
    ExecuteConv(api, context);
  } else if (op_type == "gemm") {
    ExecuteGemm(api, context);
  } else {
    throw std::runtime_error("Unknown op_type: " + op_type);
  }
}
```

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

### 9.1 Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.29)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
project(morphizen-rocm VERSION 1.0.0 LANGUAGES C CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/presets.cmake)

add_subdirectory(proto)
add_subdirectory(level-1-pass-rocm)
add_subdirectory(level-2-pass-rocm-conv)
add_subdirectory(level-2-pass-rocm-gemm)
add_subdirectory(custom-op-rocm)

if(morphizen_ENABLE_UNIT_TEST)
  add_subdirectory(test)
endif()
```

### 9.2 Build Script (build.bat)

```batch
@echo off
REM Set TheRock environment
set THEROCK_DIST=C:\Develop\m\dist\therock
set HIP_PLATFORM=amd

REM Set up MSVC
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Configure and build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -B C:/Develop/m/build/morphizen-rocm -S .
cmake --build C:/Develop/m/build/morphizen-rocm
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

- **test_conv.cpp**: Tests MIOpen Conv execution
- **test_gemm.cpp**: Tests hipBLASLt Gemm execution

### 10.2 Integration Tests

Test the full pipeline: model loading → pass execution → custom op execution

### 10.3 Model Generators

Python scripts to generate test ONNX models:
- `gen_conv_model.py`: Creates Conv models
- `gen_gemm_model.py`: Creates Gemm models

---

## 11. Future Enhancements

### 11.1 Explicit Conv→Gemm Fusion

Currently, fusion is implicit through the shared stream. Future work could add:
- Explicit graph-level fusion pass
- Intermediate buffer elimination
- Kernel fusion for specific patterns

### 11.2 Additional Operations

The architecture supports adding new operations:
1. Create new Level-2 pass (e.g., `level-2-pass-rocm-pool`)
2. Add pattern file (e.g., `patterns/pool.json`)
3. Add executor (e.g., `pool_executor.cpp`)
4. Update proto with new message type

### 11.3 Performance Optimizations

- Algorithm caching across runs
- Workspace sharing between operations
- Pinned memory for async transfers

---

## 12. References

- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

*Document Version: 1.0*
*Last Updated: January 2026*
