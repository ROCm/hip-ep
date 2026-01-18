# Logging in morphizen-rocm

## Overview

This document describes the logging infrastructure used in morphizen-rocm, including how to enable debug output, the logging hierarchy, and integration with ONNX Runtime's logging system.

## MY_LOG Macro

The primary logging mechanism for ROCm-specific debug output is the `MY_LOG` macro:

```cpp
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)
```

### Usage

```cpp
MY_LOG(1) << "[ROCm CustomOp] Important message";  // Level 1: Key events
MY_LOG(2) << "[ROCm CustomOp] Debug details";       // Level 2: Verbose debug
```

### Log Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| 0 | Disabled | No MY_LOG output (default) |
| 1 | Important events | Pass execution, key operations, errors |
| 2 | Verbose debug | Detailed debugging, variable values, flow tracing |

## Environment Variables

### `MORPHIZEN_DEBUG_ROCM`

Controls the verbosity level for MY_LOG output.

```cmd
REM Disable debug logging (default)
set MORPHIZEN_DEBUG_ROCM=0

REM Enable level 1 logging (important events)
set MORPHIZEN_DEBUG_ROCM=1

REM Enable level 2 logging (verbose debug)
set MORPHIZEN_DEBUG_ROCM=2
```

### `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE`

> **⚠️ IMPORTANT:** This variable is REQUIRED to see MY_LOG output from passes!

Enables the VitisAI EP CPU device. Without this setting, the VitisAI Execution Provider will not expose devices to ONNX Runtime, and the passes will not be triggered.

**Why is this needed?**
- The VitisAI EP uses the ORT 2.0 V2 device API
- By default, no devices are exposed to ORT
- Setting this variable enables a CPU-based device for testing/debugging
- When enabled, ORT can select VitisAI EP and run the passes

**Required for:**
- Running Level-1 and Level-2 passes
- Executing custom ops (Conv, Gemm)
- Seeing MY_LOG debug output from morphizen-rocm code

```cmd
REM REQUIRED for passes to run
set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1
```

**Without this variable:**
```
[Test] Found 1 EP device(s)
[Test]   - EP device: CPUExecutionProvider
[Test] Note: VitisAI EP registered but not exposing V2 devices
[Test] VitisAI EP V2 device API not yet implemented (EP registered OK)
```

**With this variable:**
```
[Test] Found 2 EP device(s)
[Test]   - EP device: CPUExecutionProvider
[Test]   - EP device: VitisAI
[Test] Creating session with VitisAI EP (ROCm backend)...
```

### `MORPHIZEN_GPU_TIMEOUT_MS`

Controls GPU operation timeout (see `05_GPU_TIMEOUT_HANDLING.md`).

```cmd
set MORPHIZEN_GPU_TIMEOUT_MS=5000
```

## Logging Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application                              │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │ Level-1     │   │ Level-2     │   │ Custom Op           │   │
│  │ Pass        │   │ Passes      │   │ (Conv/Gemm)         │   │
│  │             │   │             │   │                     │   │
│  │ MY_LOG(1)   │   │ MY_LOG(2)   │   │ MY_LOG(1), MY_LOG(2)│   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                      │              │
│         └────────────────┬┴──────────────────────┘              │
│                          ▼                                      │
│                 ┌────────────────┐                              │
│                 │  glog (LOG_IF) │                              │
│                 └───────┬────────┘                              │
│                         ▼                                       │
│              ┌──────────────────────┐                           │
│              │    LoggerAdapter     │                           │
│              │ (google::LogSink)    │                           │
│              └───────────┬──────────┘                           │
│                          ▼                                      │
│              ┌──────────────────────┐                           │
│              │   ORT Logger         │                           │
│              │ (Ort::Logger)        │                           │
│              └───────────┬──────────┘                           │
│                          ▼                                      │
│                    ┌──────────┐                                 │
│                    │ Console  │                                 │
│                    │ Output   │                                 │
│                    └──────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ORT Logging Integration

The morphizen framework integrates glog with ONNX Runtime's logging system through `LoggerAdapter`:

```cpp
// From morphizen/vaip-core/src/logger_adapter.hpp
class LoggerAdapter : public google::LogSink {
public:
  static std::shared_ptr<LoggerAdapter> create(const Ort::Logger& logger);
  // ...
};
```

This integration:
1. Captures all glog output (including MY_LOG)
2. Routes it through ORT's logging infrastructure
3. Provides consistent logging format with timestamps

### ORT Log Levels

When creating an ORT Environment, set the log level to INFO to see MY_LOG output:

```cpp
// Enable INFO level to see glog->ORT messages
Ort::Env env(ORT_LOGGING_LEVEL_INFO, "MyApp");
```

## Code Examples

### In Level-1 Pass (pass_main.cpp)

```cpp
#include "morphizen/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

void Level1Rocm::process(IPass& self, Graph& graph) {
  MY_LOG(1) << "[HIP EP Level-1] Starting ROCm pass";
  MY_LOG(2) << "[HIP EP Level-1] Detailed process info...";
  
  // ... pass implementation
  
  MY_LOG(1) << "[HIP EP Level-1] Completed";
}
```

### In Custom Op (custom_op.cpp)

```cpp
#include "morphizen/env_config.hpp"
#include <glog/logging.h>

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
DEF_ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS, "5000")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

void RocmCustomOp::ExecuteConv(...) {
  MY_LOG(1) << "[ROCm CustomOp] ExecuteConv (MIOpen)";
  MY_LOG(2) << "[ROCm CustomOp] Input shape: [" << ... << "]";
  
  // ... convolution implementation
  
  MY_LOG(2) << "[ROCm CustomOp] Conv completed";
}
```

## Log Output Format

### ORT-Integrated Output

When integrated with ORT, log messages appear with timestamps:

```
2026-01-17 19:37:20.7273017 [I:onnxruntime:, vitisai-ep-factory.cpp:141 ...] Creating VitisAI EP
```

### Expected MY_LOG Output

```
2026-01-17 19:37:20.xxx [I:onnxruntime:, pass_main.cpp:xxx] [HIP EP Level-1] Starting ROCm pass
2026-01-17 19:37:20.xxx [I:onnxruntime:, pass_main.cpp:xxx] [HIP EP Level-1] pass_generic_param: {...}
2026-01-17 19:37:20.xxx [I:onnxruntime:, custom_op.cpp:xxx] [ROCm CustomOp] Constructor called
2026-01-17 19:37:20.xxx [I:onnxruntime:, custom_op.cpp:xxx] [ROCm CustomOp] ExecuteConv (MIOpen)
```

## Troubleshooting

### No MY_LOG Output

1. **Check environment variable:**
   ```cmd
   echo %MORPHIZEN_DEBUG_ROCM%
   ```
   Should return `1` or `2`.

2. **Enable VitisAI EP:**
   ```cmd
   set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1
   ```

3. **Verify ORT log level is INFO:**
   ```cpp
   Ort::Env env(ORT_LOGGING_LEVEL_INFO, "MyApp");
   ```

4. **Ensure VitisAI EP is registered:**
   Look for: `VitisAI EP registered successfully`

### MY_LOG Compiles but No Output at Runtime

The `ENV_PARAM` macro reads the environment variable lazily (on first use). Ensure:
- Environment variable is set BEFORE running the executable
- No typo in variable name (case-sensitive on Linux)

### glog Assertion Failures

If you see glog assertion failures, ensure:
- glog is properly initialized (usually handled by ORT)
- DEF_ENV_PARAM is defined once per compilation unit

## Best Practices

1. **Use Level 1 for key events:**
   - Pass start/end
   - Operation completion
   - Errors and warnings

2. **Use Level 2 for debugging:**
   - Parameter values
   - Tensor shapes
   - Memory allocations
   - API call details

3. **Prefix messages consistently:**
   - `[HIP EP Level-1]` for Level-1 pass
   - `[HIP EP Level-2 Conv]` for Conv pattern pass
   - `[ROCm CustomOp]` for custom ops
   - `[HipContext]` for HIP context operations

4. **Include context in messages:**
   ```cpp
   MY_LOG(2) << "[ROCm CustomOp] Conv params - batch=" << params.batch_size()
             << ", out_channels=" << params.out_channels();
   ```

## Testing Logging

Run the integration test with debug enabled:

```cmd
set MORPHIZEN_DEBUG_ROCM=2
set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1
cd C:\Develop\m\build\morphizen-rocm\bin
ort_integration_test.exe
```

## Related Documentation

- `05_GPU_TIMEOUT_HANDLING.md` - GPU timeout configuration
- `01_DESIGN.md` - Overall architecture
- `02_LEVEL1_PASS_DESIGN.md` - Level-1 pass design

## Changelog

- **2026-01-17**: Initial documentation
  - Documented MY_LOG macro and environment variables
  - Added ORT logging integration details
  - Created troubleshooting guide
