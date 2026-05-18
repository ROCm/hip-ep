/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Shared device-VRAM scratch used by multiple wrap_* runtime helpers
// (GQA, hipBLASLt-backed MatMul/Gemm, LayerNorm variants, CausalConv,
// ...). Owned by an op-module on RuntimeState (see module_registry.h);
// the buffer grows on demand and never shrinks for the session.
//
// This header exists because, unlike single-owner op-modules (QmoeState,
// GqaGemmState, ...), the workspace is reached from many TUs. Callers
// inside lib/Runtime/real/*.cpp do:
//
//   #include "../workspace_state.h"
//   auto *m = workspace_module(state);
//   if (!m || m->ws.grow(needed_bytes) != 0) { ... }
//   char *base = static_cast<char *>(m->ws.data());
//
// Internal-only -- not part of hipdnn_ep_runtime.h's C-ABI surface.

#ifndef HIPDNN_EP_WORKSPACE_STATE_H
#define HIPDNN_EP_WORKSPACE_STATE_H

#include "growable_buffer.h"

struct RuntimeState;

struct WorkspaceState {
  hipdnn_ep::GrowableDeviceBuffer ws;

  // Constructor takes (and ignores) RuntimeState so HIPDNN_OP_MODULE-style
  // SFINAE detection works against this type.
  explicit WorkspaceState(RuntimeState *) {}
};

// Lazy-init accessor for the shared workspace op-module. Stable across
// calls within a session; null only on op-module allocation failure.
// First call in the process pays a process-global spec_table mutex; every
// subsequent call (within any session) is a bounds check + load + null
// branch through the ModuleRegistry.
WorkspaceState *workspace_module(RuntimeState *state);

#endif // HIPDNN_EP_WORKSPACE_STATE_H
