# HIP Stream and Resource Management

## Overview

This document explains how ROCm resources (HIP stream, MIOpen handle, hipBLASLt handle) are managed in the custom op implementation. The key design decision is using a **shared singleton context** that provides a single HIP stream for all ROCm operations.

## HIP Stream Fundamentals

A HIP stream is a **command queue** to the GPU, not a persistent execution context. Understanding this is crucial:

```
┌───────────────────────────────────────────────────────────────────┐
│                        HIP Stream Lifecycle                       │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  hipStreamCreate() → Empty queue created                          │
│                                                                   │
│  Queue operations (non-blocking):                                 │
│    hipMemcpyAsync() → [op1] added to queue                       │
│    miopenConvolutionForward() → [op2] added                      │
│    hipMemcpyAsync() → [op3] added                                │
│                                                                   │
│  Queue state: [op1] [op2] [op3]                                   │
│               └──── GPU executes in FIFO order                   │
│                                                                   │
│  hipStreamSynchronize() → Blocks until queue drains              │
│                                                                   │
│  Queue state: [empty] ← Sync returns, CPU unblocked              │
│                                                                   │
│  (Stream sits idle, can queue new operations later)              │
│                                                                   │
│  hipStreamDestroy() → Stream destroyed                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Key Properties

1. **Asynchronous queueing** - CPU queues operations without waiting for GPU
2. **FIFO execution** - Operations execute in order within a stream
3. **Explicit synchronization** - CPU must sync to wait for results
4. **Stream is just a queue** - No need to "keep alive" between batches

## HipContext Singleton Design

All `RocmCustomOp` instances share a single `HipContext`:

```cpp
class HipContext {
public:
  static HipContext& instance() {
    static HipContext ctx;  // Meyer's singleton
    return ctx;
  }
  
  hipStream_t stream();
  miopenHandle_t miopen_handle();
  hipblasLtHandle_t hipblaslt_handle();
  
private:
  hipStream_t stream_;
  miopenHandle_t miopen_handle_;
  hipblasLtHandle_t hipblaslt_handle_;
};
```

### Why Singleton?

1. **Resource efficiency** - One stream/handle set for entire process
2. **Implicit serialization** - All GPU ops go through same queue
3. **Library handle reuse** - MIOpen/hipBLASLt handles are expensive to create
4. **Cache locality** - MIOpen caches kernel tuning per handle

### Resource Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Process Lifetime Resources                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Program start                                                      │
│       │                                                             │
│       ▼                                                             │
│  First RocmCustomOp::Compute() called                               │
│       │                                                             │
│       ▼                                                             │
│  HipContext::instance() → ensure_initialized()                      │
│       │                                                             │
│       ├─→ hipStreamCreate(&stream_)                                 │
│       ├─→ miopenCreate(&miopen_handle_)                             │
│       ├─→ miopenSetStream(miopen_handle_, stream_)                  │
│       └─→ hipblasLtCreate(&hipblaslt_handle_)                       │
│                                                                     │
│  ... (all RocmCustomOp instances use these resources) ...           │
│                                                                     │
│  Program exit → Static destructor ~HipContext()                     │
│       │                                                             │
│       ├─→ hipblasLtDestroy(hipblaslt_handle_)                       │
│       ├─→ miopenDestroy(miopen_handle_)                             │
│       └─→ hipStreamDestroy(stream_)                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Multi-Subgraph Execution

When a model has multiple ROCm subgraphs separated by CPU ops:

```
Original graph: X → [Conv1→Conv2] → ReLU_CPU → [Conv3→Conv4] → Y

After fusion:
  X → [rocm_subgraph_0] → ReLU_CPU → [rocm_subgraph_1] → Y
           ↓                              ↓
     RocmCustomOp #1               RocmCustomOp #2
     (uses HipContext)             (uses same HipContext)
```

### Execution Timeline

```
Time →
────────────────────────────────────────────────────────────────────────

RocmCustomOp#1.Compute():
    ┌─────────────────────────────────────────────────────┐
    │ Stream queue: [H2D] [Conv1] [Conv2] [D2H]          │
    │                └───── GPU executing ─────┘          │
    │                                                     │
    │ sync_stream_with_timeout() ← CPU BLOCKS            │
    │                                                     │
    │ Stream queue: [empty] ← sync returns               │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼ D2H complete, data valid
ReLU_CPU:
    ┌─────────────────────────────────────────────────────┐
    │ Stream queue: [empty] (unchanged)                  │
    │                                                     │
    │ CPU executes ReLU, GPU idle                        │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼ CPU work complete
RocmCustomOp#2.Compute():
    ┌─────────────────────────────────────────────────────┐
    │ Stream queue: [H2D] [Conv3] [Conv4] [D2H]          │
    │                └───── GPU executing ─────┘          │
    │                                                     │
    │ sync_stream_with_timeout() ← CPU BLOCKS            │
    │                                                     │
    │ Stream queue: [empty] ← sync returns               │
    └─────────────────────────────────────────────────────┘
```

### Why This Works

1. **Stream is just a queue** - After sync, queue is empty (not "paused")
2. **CPU work doesn't affect stream** - Queue remains empty during ReLU_CPU
3. **Next Compute() queues fresh ops** - Uses same stream, starts new batch
4. **ORT executes topologically** - Guarantees correct dependency order

## Synchronization Strategy

### Per-Compute Synchronization

```cpp
void RocmCustomOp::Compute(...) const {
  // Phase 1-3: Queue async GPU operations
  UploadExternalInputs(api, context);    // hipMemcpyAsync
  ExecuteSubgraph(api, context);          // MIOpen/hipBLASLt ops
  DownloadExternalOutputs(api, context);  // hipMemcpyAsync
  
  // Phase 4: MANDATORY sync before return
  auto status = hip_ctx.sync_stream_with_timeout();
  // ... handle timeout/error ...
}
```

### Why Sync is Mandatory

```
Without sync at end of Compute():

  RocmCustomOp#1.Compute():
      [H2D] [Conv1] [Conv2] [D2H] ← still executing!
                                   ↓
  ReLU_CPU:           reads garbage ← D2H not complete!

With sync at end of Compute():

  RocmCustomOp#1.Compute():
      [H2D] [Conv1] [Conv2] [D2H] [sync] ← waits here
                                          ↓
  ReLU_CPU:                       reads valid data ✓
```

### Timeout Protection

```cpp
TimeoutStatus HipContext::sync_stream_with_timeout(int timeout_ms) {
  const int poll_interval_ms = 10;
  auto start = std::chrono::steady_clock::now();
  
  while (true) {
    hipError_t err = hipStreamQuery(stream_);  // Non-blocking check
    
    if (err == hipSuccess) {
      return TimeoutStatus::SUCCESS;  // All ops complete
    } else if (err == hipErrorNotReady) {
      // Check timeout
      auto elapsed = std::chrono::steady_clock::now() - start;
      if (elapsed >= timeout_ms) {
        return TimeoutStatus::TIMEOUT;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    } else {
      return TimeoutStatus::ERROR;
    }
  }
}
```

## Resource Categories

### 1. Process-Lifetime Resources (HipContext Singleton)

| Resource | Creation | Destruction |
|----------|----------|-------------|
| `hipStream_t` | First `Compute()` | Program exit |
| `miopenHandle_t` | First `Compute()` | Program exit |
| `hipblasLtHandle_t` | First `Compute()` | Program exit |

### 2. Per-CustomOp Resources (RocmCustomOp Instance)

| Resource | Creation | Destruction |
|----------|----------|-------------|
| `node_data_` (per-node buffers) | First `Compute()` (lazy) | `~RocmCustomOp()` |
| `external_input_buffers_` | `UploadExternalInputs()` | `~RocmCustomOp()` |
| Host weight data | Constructor | `~RocmCustomOp()` |

### 3. Per-Node Resources (NodeRuntimeData)

| Resource | Allocation | Deallocation |
|----------|------------|--------------|
| `output_buffers` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` |
| `d_weight` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` |
| `d_bias` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` |
| `workspace` | `ExecuteConvNode()` | `~NodeRuntimeData()` |

### 4. Per-Execution Resources (Transient)

| Resource | Creation | Destruction |
|----------|----------|-------------|
| `miopenTensorDescriptor_t` | `ExecuteConvNode()` | End of `ExecuteConvNode()` |
| `miopenConvolutionDescriptor_t` | `ExecuteConvNode()` | End of `ExecuteConvNode()` |

## Memory Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Memory Layout                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Per RocmCustomOp instance:                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  external_input_buffers_["X"]     → [float*] H2D buffer     │   │
│  │  external_input_buffers_["other"] → [float*] H2D buffer     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Per Node (NodeRuntimeData):                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  node_data_[0]:                                              │   │
│  │    d_weight      → [float*] Weights (cached)                │   │
│  │    d_bias        → [float*] Bias (cached)                   │   │
│  │    output_buffers[0] → [float*] Node output                 │   │
│  │    workspace     → [void*]  MIOpen workspace                │   │
│  │                                                              │   │
│  │  node_data_[1]:                                              │   │
│  │    d_weight      → [float*] Weights (cached)                │   │
│  │    output_buffers[0] → [float*] Node output                 │   │
│  │    workspace     → [void*]  MIOpen workspace                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Thread Safety Considerations

### Current Design: Single-Threaded

The current implementation assumes ORT calls `Compute()` sequentially:

```
Thread 1: RocmCustomOp#1.Compute()
          ↓ (blocks on sync)
          returns
Thread 1: ReLU_CPU
Thread 1: RocmCustomOp#2.Compute()
          ↓ (blocks on sync)
          returns
```

### Multi-Threaded Considerations

If ORT uses parallel executor:

1. **Stream operations are thread-safe** - Multiple threads can queue to same stream
2. **But sync waits for ALL queued ops** - Subgraph A's sync waits for Subgraph B's ops too
3. **Handle calls may have limitations** - Check MIOpen/hipBLASLt thread safety docs

For true parallelism, would need:
- Per-thread streams
- Per-thread handles
- More complex resource management

## See Also

- [05_GPU_TIMEOUT_HANDLING.md](05_GPU_TIMEOUT_HANDLING.md) - Timeout mechanism details
- [07_ASYNC_EXECUTION_PIPELINE.md](07_ASYNC_EXECUTION_PIPELINE.md) - Async transfer pipeline
- [01_DESIGN.md](01_DESIGN.md) - Overall architecture
