# GPU Timeout Handling

## Overview

This document describes the GPU timeout handling mechanism implemented to prevent indefinite GPU waits that can cause processes to become stuck in kernel mode and unresponsive to termination.

## Background

This feature was implemented in response to a critical incident where the `ort_integration_test.exe` process (PID 11940) became stuck in a GPU driver wait state and could not be terminated by standard Windows commands (`Stop-Process -Force`, `taskkill /F`). The process was blocked in kernel mode waiting for a GPU operation that would never complete.

The immediate solution was to use **WMI (Windows Management Instrumentation)** termination:

```powershell
$process = Get-WmiObject Win32_Process -Filter "ProcessId = 11940"
$process.Terminate()  # ReturnValue: 0 (success)
```

This worked where standard termination methods failed because WMI operates at a different kernel privilege level. However, this is not a sustainable solution for production systems. The timeout protection implemented here prevents such incidents from occurring in the first place.

## Problem Statement

Without timeout protection, GPU operations can hang indefinitely when:
- The GPU driver encounters an internal error
- GPU hardware hangs or crashes
- Operations queue too many commands
- Memory errors occur during execution
- Driver bugs cause deadlocks

When this happens, the process becomes stuck in a kernel-mode wait state that cannot be terminated by standard Windows commands (`taskkill`, `Stop-Process`, etc.), requiring either advanced WMI termination or a system reboot.

## Solution: Timeout-Protected Stream Synchronization

### Architecture

Instead of using blocking `hipStreamSynchronize()`, we implement a polling-based timeout mechanism:

1. **Non-blocking Query**: Use `hipStreamQuery()` to check stream status without blocking
2. **Polling Loop**: Poll at regular intervals (10ms) to check completion
3. **Timeout Detection**: Track elapsed time and abort if threshold exceeded
4. **Early Exit**: Return immediately on success or error

### Implementation Components

#### 1. TimeoutStatus Enum

```cpp
enum class TimeoutStatus {
  SUCCESS,   // Operation completed successfully
  TIMEOUT,   // Operation timed out
  ERROR      // Error occurred
};
```

#### 2. WaitStreamWithTimeout Function

Core polling-based timeout implementation in `custom-op-rocm/src/custom_op.hpp`:

```cpp
inline TimeoutStatus WaitStreamWithTimeout(hipStream_t stream, int timeout_ms) {
  const int poll_interval_ms = 10;  // Poll every 10ms
  auto start = std::chrono::steady_clock::now();
  
  while (true) {
    hipError_t err = hipStreamQuery(stream);
    
    if (err == hipSuccess) {
      return TimeoutStatus::SUCCESS;
    } else if (err == hipErrorNotReady) {
      auto elapsed = std::chrono::steady_clock::now() - start;
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
      
      if (elapsed_ms >= timeout_ms) {
        LOG(ERROR) << "[ROCm Timeout] GPU operation timed out after " << elapsed_ms << "ms";
        return TimeoutStatus::TIMEOUT;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    } else {
      LOG(ERROR) << "[ROCm Timeout] hipStreamQuery failed: " << hipGetErrorString(err);
      return TimeoutStatus::ERROR;
    }
  }
}
```

#### 3. HipContext Timeout Method

Added to `HipContext` class:

```cpp
TimeoutStatus sync_stream_with_timeout(int timeout_ms = 0) {
  ensure_initialized();
  if (!initialized_) {
    return TimeoutStatus::ERROR;
  }
  
  // Use environment variable default if not specified
  if (timeout_ms <= 0) {
    timeout_ms = ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS);
  }
  
  LOG(INFO) << "[HipContext] Synchronizing stream with " << timeout_ms << "ms timeout...";
  return WaitStreamWithTimeout(stream_, timeout_ms);
}
```

#### 4. Usage in Custom Operations

Both `ExecuteConv()` and `ExecuteGemm()` now use timeout-protected synchronization:

```cpp
auto timeout_status = hip_ctx.sync_stream_with_timeout();

if (timeout_status == TimeoutStatus::TIMEOUT) {
  LOG(ERROR) << "[ROCm CustomOp] Operation TIMED OUT!";
  throw std::runtime_error("ROCm CustomOp: GPU operation timed out");
} else if (timeout_status == TimeoutStatus::ERROR) {
  LOG(ERROR) << "[ROCm CustomOp] Stream synchronization ERROR!";
  throw std::runtime_error("ROCm CustomOp: Stream synchronization error");
}
```

## Configuration

### Environment Variables

#### `MORPHIZEN_GPU_TIMEOUT_MS`
- **Purpose**: Set the GPU operation timeout in milliseconds
- **Default**: `5000` (5 seconds)
- **Usage**: `set MORPHIZEN_GPU_TIMEOUT_MS=10000` (for 10 second timeout)
- **Recommendations**:
  - Development/Testing: `5000-10000` ms (catch hangs quickly)
  - Production with large models: `30000-60000` ms (allow complex operations)
  - Debugging: `120000` ms (2 minutes, for very slow operations)

#### `MORPHIZEN_GPU_WATCHDOG_ENABLED`
- **Purpose**: Enable/disable watchdog monitoring (reserved for future use)
- **Default**: `1` (enabled)
- **Usage**: `set MORPHIZEN_GPU_WATCHDOG_ENABLED=0` (to disable)

### Setting Timeouts

**Windows (PowerShell):**
```powershell
$env:MORPHIZEN_GPU_TIMEOUT_MS = "10000"
```

**Windows (CMD):**
```cmd
set MORPHIZEN_GPU_TIMEOUT_MS=10000
```

**In build.bat or test scripts:**
```batch
@echo off
set MORPHIZEN_GPU_TIMEOUT_MS=15000
.\ort_integration_test.exe
```

## Behavior and Error Handling

### Success Case
```
[HipContext] Synchronizing stream with 5000ms timeout...
[ROCm CustomOp] Stream synchronized successfully
[ROCm CustomOp] Conv completed
```

### Timeout Case
```
[ROCm Timeout] GPU operation timed out after 5000ms
[ROCm CustomOp] Conv operation TIMED OUT!
[ROCm CustomOp] This indicates a GPU hang or extremely slow operation.
[ROCm CustomOp] Adjust MORPHIZEN_GPU_TIMEOUT_MS if needed, or investigate GPU issues.
Exception: ROCm CustomOp Conv: GPU operation timed out
```

### Error Case
```
[ROCm Timeout] hipStreamQuery failed: hipErrorInvalidHandle (400)
[ROCm CustomOp] Conv stream synchronization ERROR!
Exception: ROCm CustomOp Conv: Stream synchronization error
```

## Benefits

1. **Prevents Stuck Processes**: Operations timeout instead of hanging indefinitely
2. **Graceful Degradation**: Process can be terminated normally after timeout
3. **Early Problem Detection**: Identifies GPU issues within seconds rather than requiring manual intervention
4. **Configurable**: Adjust timeout based on workload and hardware
5. **Informative**: Clear error messages indicate timeout vs. other errors
6. **Low Overhead**: 10ms polling interval has minimal CPU impact

## Performance Considerations

### Polling Overhead
- **Polling Interval**: 10ms
- **CPU Impact**: Minimal - thread sleeps between polls
- **Typical Operations**: Complete in <100ms, so overhead is negligible

### Timeout Tuning
- **Too Short**: False positives on legitimate long operations
- **Too Long**: Delayed detection of real hangs
- **Recommended Range**: 5-60 seconds depending on model complexity

## Testing

### Unit Testing Timeout
To test the timeout mechanism:

```cpp
// Simulate a long operation
for (int i = 0; i < 1000000; i++) {
  hipLaunchKernelGGL(...);  // Launch many operations
}

// This should timeout with default settings
auto status = hip_ctx.sync_stream_with_timeout(100);  // 100ms timeout
assert(status == TimeoutStatus::TIMEOUT);
```

### Integration Testing
Run tests with various timeout values:

```batch
REM Quick timeout for testing
set MORPHIZEN_GPU_TIMEOUT_MS=500
.\ort_integration_test.exe

REM Normal timeout
set MORPHIZEN_GPU_TIMEOUT_MS=5000
.\ort_integration_test.exe

REM Extended timeout for large models
set MORPHIZEN_GPU_TIMEOUT_MS=30000
.\ort_integration_test.exe
```

## Future Enhancements

### 1. Watchdog Thread (Phase 2)
Implement a background watchdog thread that:
- Monitors all GPU operations
- Can forcefully abort stuck operations
- Provides telemetry on operation durations

### 2. Per-Operation Timeouts
Different timeout values for different operation types:
```cpp
// Conv operations might need more time
timeout_ms = op_type == "conv" ? 10000 : 5000;
```

### 3. Adaptive Timeouts
Learn typical operation durations and set timeouts dynamically:
```cpp
// Set timeout to 3x the average duration
timeout_ms = avg_duration_ms * 3;
```

### 4. GPU Health Checks
Periodic GPU health monitoring:
- Memory availability checks
- Temperature monitoring
- Driver responsiveness tests

## Troubleshooting

### Process Still Hanging
If timeouts are not triggering:
1. Verify environment variable is set: `echo %MORPHIZEN_GPU_TIMEOUT_MS%`
2. Check logs for timeout initialization
3. Ensure code is calling `sync_stream_with_timeout()` not `hipStreamSynchronize()`

### False Timeout Alarms
If legitimate operations timeout:
1. Increase timeout: `set MORPHIZEN_GPU_TIMEOUT_MS=30000`
2. Profile operation duration to determine appropriate timeout
3. Check for GPU performance issues (thermal throttling, etc.)

### Timeout Not Helping
If process still gets stuck despite timeouts:
1. The hang might occur during operation submission, not synchronization
2. Add timeouts to other blocking calls (memory allocation, kernel launches)
3. Consider using `hipStreamCreateWithFlags()` with non-blocking flags

## References

- **HIP API Documentation**: https://rocm.docs.amd.com/projects/HIP/en/latest/
- **WMI Process Termination**: Used as workaround when process gets stuck
- **Related Issue**: Process stuck in kernel mode wait (PID 11940 incident)

## Implementation Details

For historical context on the implementation, see `doc/archive/IMPLEMENTATION_SUMMARY_TIMEOUT.md`.

### Files Modified

**Core Implementation:**
- `custom-op-rocm/src/custom_op.hpp` - Timeout mechanism and HipContext integration
- `custom-op-rocm/src/custom_op.cpp` - Applied timeout to Conv and Gemm operations

**Testing:**
- `test/test_timeout.cpp` - Comprehensive test suite
- `test/CMakeLists.txt` - Build configuration
- `test/run_timeout_test.bat` - Test runner script

**Documentation:**
- `README.md` - Updated with timeout feature
- This document

## Changelog

- **2026-01-17**: Initial implementation of timeout-protected stream synchronization
  - Added `WaitStreamWithTimeout()` helper function
  - Added `sync_stream_with_timeout()` method to `HipContext`
  - Updated `ExecuteConv()` and `ExecuteGemm()` to use timeout protection
  - Added environment variable configuration
  - Created documentation
