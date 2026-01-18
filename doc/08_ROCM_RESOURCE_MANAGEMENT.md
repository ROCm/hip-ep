# ROCm Resource Management

## Overview

This document explains how ROCm resources are managed in the custom op implementation, including:
- **HIP stream** - Command queue for GPU operations
- **MIOpen handle** - Library context for convolution operations
- **hipBLASLt handle** - Library context for GEMM operations
- **Descriptors** - Metadata structures describing tensors and operations
- **GPU buffers** - Device memory for data

The key design decisions are:
1. **Per-session HipContext** - Each ORT session gets its own stream and handles (not singleton)
2. **Session-scoped resource lifetime** - Resources cleaned up when session ends via PassContext
3. **Per-operation descriptors** that can be destroyed before stream sync
4. **GPU buffers** that must remain valid until operations complete

> **Design Evolution:** The original singleton design was replaced with per-session contexts to support multiple ORT sessions running in parallel. See "Design Decisions" section for details.

## Resource Categories

### 1. Session-Scoped Resources (Per-Session HipContext)

| Resource | Creation | Destruction | Notes |
|----------|----------|-------------|-------|
| `hipStream_t` | First CustomOp in session | Session destroyed | Per-session command queue |
| `miopenHandle_t` | First CustomOp in session | Session destroyed | Per-session kernel cache |
| `hipblasLtHandle_t` | First CustomOp in session | Session destroyed | Per-session GEMM context |

### 2. Per-CustomOp Resources (RocmCustomOp Instance)

| Resource | Creation | Destruction | Notes |
|----------|----------|-------------|-------|
| `node_data_` | First `Compute()` (lazy) | `~RocmCustomOp()` | Per-node buffers |
| `external_input_buffers_` | `UploadExternalInputs()` | `~RocmCustomOp()` | H2D staging buffers |
| Host weight data | Constructor | `~RocmCustomOp()` | CPU cache for weights |

### 3. Per-Node Resources (NodeRuntimeData)

| Resource | Allocation | Deallocation | Notes |
|----------|------------|--------------|-------|
| `output_buffers` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` | Intermediate outputs |
| `d_weight` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` | Cached weights on GPU |
| `d_bias` | `AllocateIntermediateBuffers()` | `~NodeRuntimeData()` | Cached bias on GPU |
| `workspace` | `ExecuteConvNode()` | `~NodeRuntimeData()` | MIOpen workspace |

### 4. Per-Operation Resources (Transient)

| Resource | Creation | Destruction | Notes |
|----------|----------|-------------|-------|
| `miopenTensorDescriptor_t` | `ExecuteConvNode()` | End of `ExecuteConvNode()` | CPU metadata only |
| `miopenConvolutionDescriptor_t` | `ExecuteConvNode()` | End of `ExecuteConvNode()` | CPU metadata only |

## HIP Stream Fundamentals

A HIP stream is a **command queue** to the GPU, not a persistent execution context:

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

## Per-Session HipContext Design

Each ORT session gets its own `HipContext`, stored via VitisAI's `PassContext`:

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
  hipStream_t stream_;
  miopenHandle_t miopen_handle_;
  hipblasLtHandle_t hipblaslt_handle_;
  std::mutex algo_find_mutex_;  // For thread-safe algorithm search
};
```

### Per-Session Context via SessionContextRegistry

Session-scoped `HipContext` sharing is implemented using a **process-global registry** 
(`SessionContextRegistry`) that maps `PassContext*` to `shared_ptr<HipContext>`:

```cpp
/**
 * SessionContextRegistry: Per-Session HipContext Manager
 * 
 * Manages HipContext instances keyed by PassContext address.
 * Uses weak_ptr to track if any CustomOps are still using the context.
 */
class SessionContextRegistry {
public:
  static SessionContextRegistry& instance();
  
  // Get or create HipContext for a session
  std::shared_ptr<HipContext> get_or_create(const vaip_core::PassContext* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_expired();  // Remove stale entries
    
    auto it = contexts_.find(ctx);
    if (it != contexts_.end()) {
      if (auto locked = it->second.lock()) {
        return locked;  // Reuse existing
      }
    }
    
    // Create new context for this session
    auto new_context = std::make_shared<HipContext>();
    contexts_[ctx] = new_context;  // Store as weak_ptr
    return new_context;
  }
  
private:
  std::mutex mutex_;
  std::unordered_map<const PassContext*, std::weak_ptr<HipContext>> contexts_;
};

// In RocmCustomOp constructor:
RocmCustomOp::RocmCustomOp(std::shared_ptr<const PassContext> context, ...) {
  hip_context_ = SessionContextRegistry::instance().get_or_create(context.get());
}
```

**Key design points:**
- **Process-global registry**: Singleton pattern for cross-session management
- **PassContext pointer as key**: All CustomOps in the same session share the same PassContext pointer
- **weak_ptr for cleanup**: When all CustomOps are destroyed, the HipContext is automatically cleaned up
- **Thread-safe**: Mutex protects registry access
- **No framework dependency**: Works without requiring changes to VitisAI PassContext API

### Why Per-Session?

1. **Parallel session support** - Each session has isolated GPU resources
2. **No cross-session interference** - Session A's sync doesn't affect Session B
3. **Automatic cleanup** - Resources freed when session ends (not process end)
4. **GPU parallelism** - Multiple sessions can execute on GPU concurrently

### Resource Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Session Lifetime Resources                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ORT Session Created                                                │
│       │                                                             │
│       ▼                                                             │
│  Level-1 Pass runs (or cache hit) → RocmCustomOp instances created  │
│       │                                                             │
│       ▼                                                             │
│  RocmCustomOp constructor calls SessionContextRegistry::get_or_create│
│       │                                                             │
│       ├─→ Checks registry for existing HipContext (by PassContext*) │
│       │   (not found - first CustomOp in session)                   │
│       │                                                             │
│       ├─→ Creates new HipContext                                    │
│       │     ├─→ hipStreamCreate(&stream_)                           │
│       │     ├─→ miopenCreate(&miopen_handle_)                       │
│       │     └─→ hipblasLtCreate(&hipblaslt_handle_)                 │
│       │                                                             │
│       └─→ Stores in registry as weak_ptr                            │
│                                                                     │
│  Subsequent CustomOps in same session:                              │
│       SessionContextRegistry::get_or_create → Returns existing      │
│       HipContext (same PassContext* → same HipContext shared_ptr)   │
│                                                                     │
│  Session destroyed → All RocmCustomOp destructors called            │
│       │                                                             │
│       ▼ (shared_ptr ref count → 0, weak_ptr expires)                │
│  ~HipContext()                                                      │
│       ├─→ hipblasLtDestroy(hipblaslt_handle_)                       │
│       ├─→ miopenDestroy(miopen_handle_)                             │
│       └─→ hipStreamDestroy(stream_)                                 │
│                                                                     │
│  Next registry access → cleanup_expired() removes stale entry       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Multiple Sessions Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Multiple Sessions with Per-Session Contexts                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Session A (Thread 1):              Session B (Thread 2):              │
│  ┌─────────────────────────┐        ┌─────────────────────────┐        │
│  │ PassContext_A           │        │ PassContext_B           │        │
│  │  └─ HipContext_A        │        │  └─ HipContext_B        │        │
│  │      • stream_a         │        │      • stream_b         │        │
│  │      • handle_a         │        │      • handle_b         │        │
│  │                         │        │                         │        │
│  │  RocmCustomOp A1 ─────┐ │        │  RocmCustomOp B1 ─────┐ │        │
│  │  RocmCustomOp A2 ─────┼─┘        │  RocmCustomOp B2 ─────┼─┘        │
│  │      (share context_) │          │      (share context_) │          │
│  └───────────────────────┘          └───────────────────────┘          │
│                                                                         │
│  GPU:                                                                   │
│    Stream A: [A1_Conv] [A2_Conv] ←──────┐                               │
│    Stream B: [B1_Conv] [B2_Conv] ←──────┼── Can execute concurrently   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## MIOpen Handle Management

### Ownership Model

Each `HipContext` owns its own MIOpen handle:

```cpp
class HipContext {
private:
  miopenHandle_t miopen_handle_ = nullptr;  // Owned by this session's context
};
```

**Ownership**: Each `HipContext` owns its handle exclusively.
**Lifetime**: From session creation until session destruction.

### Handle Creation Cost

MIOpen handle creation involves:
- Internal resource allocation
- Kernel cache setup
- Device context binding

This is **expensive** (can take hundreds of milliseconds). However:
- Cost is amortized over all operations within a session
- Most use cases run many inferences per session
- First-inference latency is acceptable for production deployment

### Stream Association

Each MIOpen handle is associated with its session's stream at creation:

```cpp
miopenCreate(&miopen_handle_);
miopenSetStream(miopen_handle_, stream_);  // Associates stream permanently
```

Each session has its own (handle, stream) pair, enabling true GPU parallelism between sessions.

### Kernel Cache Per Session

MIOpen caches kernel tuning results per handle:
- First convolution with specific dimensions → MIOpen searches for optimal algorithm
- Subsequent convolutions with same dimensions → Uses cached result
- Cache is scoped to the session (not shared across sessions)

**Note:** Cross-session cache sharing was considered but rejected because:
1. General EP can't predict model shapes → low cache hit rate across different models
2. Mutex overhead would reduce parallelism benefits
3. Simpler design is preferred for maintainability

For repeated inferences of the same model within a session, the kernel cache provides full benefit.

## Descriptor vs Buffer Lifetime

This is a critical concept for understanding GPU programming with MIOpen.

### The Key Insight

**Descriptors are CPU metadata** that describe tensor shapes and operation parameters. They are read by MIOpen at the time of the function call and their values are copied into the GPU command buffer. After the call returns, the descriptor structures are no longer needed.

**GPU buffers contain actual data** that the GPU will read/write during kernel execution. They must remain valid until the GPU operation completes.

### GPU Command Buffer Architecture

When you call `miopenConvolutionForward()`, here's what happens:

```
┌─────────────────────────────────────────────────────────────────────┐
│          miopenConvolutionForward() Internal Execution              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CPU (synchronous, before function returns):                        │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 1. Read descriptor metadata:                                  │ │
│  │    - input_desc → N=1, C=64, H=56, W=56                       │ │
│  │    - weight_desc → K=128, C=64, R=3, S=3                      │ │
│  │    - conv_desc → pad=1, stride=1                              │ │
│  │                                                                │ │
│  │ 2. Select kernel based on parameters                          │ │
│  │                                                                │ │
│  │ 3. Prepare kernel launch arguments:                           │ │
│  │    kernel_args = {                                            │ │
│  │      d_input = 0x7fff0000,   // GPU pointer (VALUE copied)    │ │
│  │      d_weight = 0x7fff1000,  // GPU pointer (VALUE copied)    │ │
│  │      d_output = 0x7fff2000,  // GPU pointer (VALUE copied)    │ │
│  │      N=1, C=64, H=56, ...    // Shape values (copied)         │ │
│  │    }                                                          │ │
│  │                                                                │ │
│  │ 4. hipLaunchKernelGGL(..., stream, kernel_args)               │ │
│  │    └─→ Submits kernel + args to command queue                 │ │
│  │        (Runtime copies kernel_args to command buffer)         │ │
│  │                                                                │ │
│  └───────────────────────────────────────────────────────────────┘ │
│  │                                                                  │
│  │ ← miopenConvolutionForward() RETURNS HERE                       │
│  │   (No sync, purely async - CPU continues immediately)          │
│  ▼                                                                  │
│                                                                     │
│  GPU (asynchronous, after function returns):                        │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ [Command buffer contains: kernel code + copied arguments]     │ │
│  │                                                                │ │
│  │ 5. GPU scheduler dequeues command when ready                  │ │
│  │ 6. Kernel executes, reads from d_input/d_weight addresses     │ │
│  │ 7. Kernel writes to d_output address                          │ │
│  │ 8. Kernel completes                                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Visual: What Gets Copied vs What Gets Referenced

```
CPU Stack/Heap           GPU Command Buffer           GPU Memory
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│ input_desc   │─VALUE ─→│ N=1,C=64,H=56... │         │              │
│  (metadata)  │ COPIED  │ (embedded)       │         │              │
└──────────────┘         │                  │         │              │
                         │ d_input=0x7fff.. │─POINTS─→│ actual data  │
                         │ (pointer copied) │   TO    │ (must exist) │
                         └──────────────────┘         └──────────────┘
       │                          │                          │
       ↓                          ↓                          ↓
   CAN DESTROY              LIVES IN CMD BUF          MUST STAY ALIVE
   AFTER CALL               (OWNED BY GPU)            UNTIL SYNC
```

### What Can Be Destroyed Before `hipStreamSynchronize()`

| Resource | Destroy Before Sync? | Reason |
|----------|---------------------|--------|
| `miopenTensorDescriptor_t input_desc` | ✅ Yes | Values copied to command buffer |
| `miopenTensorDescriptor_t weight_desc` | ✅ Yes | Values copied to command buffer |
| `miopenTensorDescriptor_t output_desc` | ✅ Yes | Values copied to command buffer |
| `miopenConvolutionDescriptor_t conv_desc` | ✅ Yes | Values copied to command buffer |
| `d_input` (GPU buffer) | ❌ No | GPU reads from this address during execution |
| `d_weight` (GPU buffer) | ❌ No | GPU reads from this address during execution |
| `d_output` (GPU buffer) | ❌ No | GPU writes to this address during execution |
| `workspace` (GPU buffer) | ❌ No | GPU uses this during execution |

### Current Code Pattern (Correct)

```cpp
void ExecuteConvNode(...) {
  // Create descriptors (CPU-side metadata)
  miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
  miopenConvolutionDescriptor_t conv_desc;
  
  miopenCreateTensorDescriptor(&input_desc);
  miopenCreateTensorDescriptor(&weight_desc);
  miopenCreateTensorDescriptor(&output_desc);
  miopenCreateConvolutionDescriptor(&conv_desc);
  
  // Setup descriptors with shape info...
  miopenSet4dTensorDescriptor(input_desc, ...);
  // ... etc ...
  
  // Launch async operation - descriptors consumed HERE
  miopenConvolutionForward(handle, &alpha,
                           input_desc, d_input,     // descriptor read, pointer copied
                           weight_desc, d_weight,   // descriptor read, pointer copied
                           conv_desc, algo, &beta,
                           output_desc, d_output,   // descriptor read, pointer copied
                           workspace, workspace_size);
  
  // ✅ SAFE: Destroy descriptors immediately after call
  // Values already copied to GPU command buffer
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(input_desc);
  
  // Return - descriptors gone, GPU still working with buffers
}

void Compute(...) {
  ExecuteSubgraph(...);  // All descriptors destroyed, ops queued
  
  // ❗ GPU buffers (d_input, d_weight, d_output, workspace) still in use
  
  sync_stream_with_timeout();  // Wait for GPU to finish with buffers
  
  // ✅ Now safe to free GPU buffers (but we keep them for reuse)
}
```

### Analogy: Recipe Card vs Ingredients

Think of descriptors like a **recipe card** and buffers like **ingredients**:

- **Recipe card** (descriptor): Chef reads it to know what to cook, then can throw it away
- **Ingredients** (buffers): Must be available while cooking is in progress

Once the chef has read the recipe and started cooking, you can discard the recipe card. But you can't take away the ingredients until the dish is done!

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
    ┌─────────────────────────────────────────────────────────────┐
    │ Stream queue: [H2D] [Conv1] [Conv2] [D2H]                  │
    │                └───── GPU executing ─────┘                  │
    │                                                             │
    │ sync_stream_with_timeout() ← CPU BLOCKS                    │
    │                                                             │
    │ Stream queue: [empty] ← sync returns                       │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼ D2H complete, data valid
ReLU_CPU:
    ┌─────────────────────────────────────────────────────────────┐
    │ Stream queue: [empty] (unchanged)                          │
    │                                                             │
    │ CPU executes ReLU, GPU idle                                │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼ CPU work complete
RocmCustomOp#2.Compute():
    ┌─────────────────────────────────────────────────────────────┐
    │ Stream queue: [H2D] [Conv3] [Conv4] [D2H]                  │
    │                └───── GPU executing ─────┘                  │
    │                                                             │
    │ sync_stream_with_timeout() ← CPU BLOCKS                    │
    │                                                             │
    │ Stream queue: [empty] ← sync returns                       │
    └─────────────────────────────────────────────────────────────┘
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

## Design Decisions

### Per-Session Context (Current Design)

The current implementation uses **per-session HipContext** managed by a **process-global `SessionContextRegistry`**. This design:

- ✅ Supports multiple ORT sessions running in parallel
- ✅ Enables GPU-level parallelism via multiple streams
- ✅ Automatically cleans up when all CustomOps in a session are destroyed (via `weak_ptr`)
- ✅ Provides isolation between sessions
- ✅ Works regardless of whether Level-1 pass runs or cache hits

**How it works:**
1. `SessionContextRegistry` is a process-global singleton
2. Each `RocmCustomOp` constructor calls `SessionContextRegistry::get_or_create(context.get())`
3. Registry uses `PassContext*` pointer as the session key
4. All CustomOps in the same session share the same `HipContext` via `shared_ptr`
5. When all CustomOps are destroyed, `weak_ptr` expires and HipContext is cleaned up

See "Per-Session HipContext Design" section above for implementation details.

### Why Singleton was Replaced (Historical)

> **Note:** The detailed singleton design history has been moved to [archive/SINGLETON_DESIGN_HISTORY.md](archive/SINGLETON_DESIGN_HISTORY.md).

In summary, the singleton pattern was replaced because it:
- Did NOT support multiple ORT sessions running in parallel
- Caused stream contention (all ops from all sessions queued to same stream)
- Had sync interference (Session A's sync waited for Session B's ops)
- Had potential MIOpen thread safety issues

The current per-session design addresses all these limitations.

### Why Cache Algorithm Search Results?

#### The Problem

`miopenFindConvolutionForwardAlgorithm()` is called to find the best convolution algorithm:

```cpp
miopenFindConvolutionForwardAlgorithm(
    handle, input_desc, d_input,
    weight_desc, d_weight,
    conv_desc, output_desc, d_output,
    4, &algo_count, perf_results,
    workspace_ptr, workspace_size, 
    false);  // exhaustiveSearch=false
```

Even with `exhaustiveSearch=false`, this call:
1. Queries internal MIOpen caches (still has overhead)
2. May run quick GPU benchmarks
3. Returns algorithm performance data
4. Takes **1-10ms per call** even on cache hit

For a model with 50 conv layers × 1000 inferences = **50,000 find calls**.

#### The Solution: Per-Node Algorithm Caching

Since node shapes are **static** (known from protobuf at construction time), we cache the algorithm result in `NodeRuntimeData`:

```cpp
struct NodeRuntimeData {
  // ... existing fields ...
  
  // Cached algorithm for conv operations
  bool conv_algo_cached = false;
  miopenConvFwdAlgorithm_t cached_conv_algo;
  size_t cached_conv_workspace_size = 0;
};
```

**First inference:** Run `miopenFindConvolutionForwardAlgorithm()`, cache result
**Subsequent inferences:** Use cached algorithm directly, skip Find call

```cpp
void ExecuteConvNode(...) {
  auto& data = *node_data_[node_id];
  
  if (!data.conv_algo_cached) {
    // First call: search for best algorithm
    miopenFindConvolutionForwardAlgorithm(..., &perf_results);
    data.cached_conv_algo = perf_results[0].fwd_algo;
    data.cached_conv_workspace_size = perf_results[0].memory;
    data.conv_algo_cached = true;
  }
  
  // All calls: use cached algorithm
  miopenConvolutionForward(..., data.cached_conv_algo, ...);
}
```

#### Performance Impact

| Scenario | Without Cache | With Cache | Improvement |
|----------|---------------|------------|-------------|
| First inference | 500ms | 500ms | 0% (warmup) |
| Per-conv overhead | ~5ms | ~0.1ms | **50x faster** |
| 100 inferences, 50 convs | 500+25000ms | 500+500ms | **25x faster** |

#### Why Not Rely on MIOpen's Internal Cache?

MIOpen does have internal caching, but:
1. **API overhead** - Even cache hits go through the full API call
2. **Hash computation** - Must hash tensor dimensions every call
3. **Thread sync** - Internal cache access may have mutex overhead
4. **No guarantee** - Cache eviction policies are internal

Our explicit caching is:
- Zero overhead after first call (just a bool check)
- Guaranteed to be available
- No internal MIOpen interactions needed

### Why NOT Cache Descriptors?

Descriptors (`miopenTensorDescriptor_t`, `miopenConvolutionDescriptor_t`) are intentionally **not** cached:

**1. They're Cheap**
```cpp
miopenCreateTensorDescriptor(&desc);   // ~1μs - just malloc
miopenSet4dTensorDescriptor(desc,...); // ~1μs - just struct fill
miopenDestroyTensorDescriptor(desc);   // ~1μs - just free
```

Total: ~3-5 microseconds per descriptor. With 4 descriptors per conv, that's ~20μs per operation - negligible compared to kernel execution time.

**2. Values are Copied, Not Referenced**
As documented in the "Descriptor vs Buffer Lifetime" section, descriptor values are copied into the GPU command buffer when `miopenConvolutionForward()` is called. The descriptor objects themselves are never accessed by the GPU.

**3. Caching Adds Complexity**
Cached descriptors would need:
- Lifetime management
- Thread-safety considerations
- Memory overhead for rarely-used benefit

**4. Idiomatic Usage**
Creating/destroying descriptors per-call is the recommended MIOpen usage pattern, consistent with cuDNN best practices.

## See Also

- [05_GPU_TIMEOUT_HANDLING.md](05_GPU_TIMEOUT_HANDLING.md) - Timeout mechanism details
- [07_ASYNC_EXECUTION_PIPELINE.md](07_ASYNC_EXECUTION_PIPELINE.md) - Async transfer pipeline
- [01_DESIGN.md](01_DESIGN.md) - Overall architecture
