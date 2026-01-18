# Singleton Design History

> **Status:** Historical - This design was replaced with per-session contexts.
> 
> **Current documentation:** See [../08_ROCM_RESOURCE_MANAGEMENT.md](../08_ROCM_RESOURCE_MANAGEMENT.md) for the current per-session design.

This document preserves the original singleton design rationale for historical reference.

## Why Singleton Was Originally Used

The original `HipContext` used a singleton pattern for the MIOpen handle.

### Benefits of Singleton

**1. Handle Creation is Expensive**
```cpp
miopenCreate(&miopen_handle_);  // Can take 100-500ms first time
```
MIOpen handle creation involves:
- HIP/CUDA context initialization
- Kernel cache directory setup
- Internal memory pool initialization
- Database connection for find-db cache

Creating a new handle per `Compute()` call would devastate performance.

**2. Kernel Cache Per Handle**
MIOpen maintains an internal algorithm cache per handle:
```
First  Conv(1,64,224,224) → Algorithm search runs (slow, ~10-100ms)
Second Conv(1,64,224,224) → Cache hit (fast, <1ms)
```
If we created new handles, this cache would be lost.

**3. Matches ORT Execution Model**
ONNX Runtime typically calls `Compute()` sequentially for a given session. A single handle with shared stream provides implicit serialization without explicit locking.

**4. Single GPU Target**
The design targeted one AMD GPU per process. Singleton aligned with this constraint.

### Trade-offs and Limitations

| Aspect | Singleton | Alternative (Pool/TLS) |
|--------|-----------|------------------------|
| Simplicity | ✅ Simple | ❌ More complex |
| Kernel cache | ✅ Shared across ops | ⚠️ Per-handle cache |
| Multi-GPU | ❌ Not supported | ✅ Per-device handles |
| Multi-thread | ❌ Not thread-safe | ✅ Thread-local or mutex |
| Memory | ✅ Minimal | ⚠️ N handles worth |

## Problems with Singleton for Parallel Sessions

The singleton design was **NOT** suitable for parallel ORT sessions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Multiple Sessions with Shared Singleton (Historical)         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Session A (Thread 1):         Session B (Thread 2):                   │
│  ┌─────────────────┐           ┌─────────────────┐                     │
│  │ CustomOp A1     │           │ CustomOp B1     │                     │
│  │ CustomOp A2     │           │ CustomOp B2     │                     │
│  └────────┬────────┘           └────────┬────────┘                     │
│           │                             │                               │
│           └──────────┬──────────────────┘                               │
│                      ▼                                                  │
│            ┌─────────────────────┐                                      │
│            │  HipContext (Singleton)                                    │
│            │  • stream_          │ ← Single stream for ALL ops          │
│            │  • miopen_handle_   │                                      │
│            └─────────────────────┘                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Problems:**

| Issue | Impact | Severity |
|-------|--------|----------|
| **Stream contention** | All ops from all sessions queue to same stream - no parallelism | ⚠️ Performance |
| **Sync interference** | Session A's sync waits for Session B's ops too | ⚠️ Performance |
| **MIOpen thread safety** | MIOpen handle may not be thread-safe for concurrent calls | ❌ Correctness |
| **Algorithm caching** | `miopenFind*` may have race conditions if called concurrently | ❌ Correctness |

**What happened in practice:**

```
Time →
─────────────────────────────────────────────────────────────────────────

Thread 1 (Session A):
  Queue [A1_H2D] [A1_Conv] [A1_D2H] → sync_stream_with_timeout()
                                                ↓
                              Waits for ALL queued ops (including B's!)
                                                ↓
Thread 2 (Session B):
  Queue [B1_H2D] [B1_Conv] [B1_D2H] → sync_stream_with_timeout()
                                                ↓
                              Also waits for ALL ops

Stream: [A1_H2D] [A1_Conv] [A1_D2H] [B1_H2D] [B1_Conv] [B1_D2H]
        └───────────────── All ops serialize on one stream ─────────┘
```

## Singleton Constraints

The singleton implementation was suitable **only** for:
- ✅ Single ORT session per process
- ✅ Sequential inference calls within that session
- ✅ Single AMD GPU

It was **NOT** suitable for:
- ❌ Multiple ORT sessions running in parallel
- ❌ ORT with parallel executor enabled
- ❌ Multi-GPU setups

## Migration to Per-Session Contexts

The singleton design was replaced with **per-session HipContext** to support:
1. Multiple ORT sessions running in parallel
2. GPU-level parallelism via multiple streams
3. Automatic cleanup when session ends
4. Isolation between sessions

See [../08_ROCM_RESOURCE_MANAGEMENT.md](../08_ROCM_RESOURCE_MANAGEMENT.md) for the current implementation.

---

*Archived: January 2026*
