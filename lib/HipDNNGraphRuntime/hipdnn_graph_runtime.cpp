/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipdnn_graph_runtime.cpp - hipDNN graph registry + execute ---------===//
//
// Shared library that owns compiled HipDNNGraph objects and exposes a C API
// for model.dll to call. Mirrors hipDNNEP's Kernel::Execute() role.
//
// Same-process singleton: hip-compiler.dll populates g_default_registry and
// g_default_handle during compilation. model.dll's hipdnn_graph_execute falls
// back to these when RuntimeState fields are nullptr.
//
//===----------------------------------------------------------------------===//

#include "hipdnn_graph_runtime.h"
#include "../HipDNNGraph/HipDNNGraph.h"

#include <backend/hipdnn_backend.h>
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

// Layout-compatible mirror of RuntimeState (defined in runtime_state_internal.h).
// We cannot include the real header because runtime_types.h pulls in
// hip_runtime.h which contains MSVC-incompatible vector types.
// All handle fields (hipStream_t, miopenHandle_t, hipblasLtHandle_t) are
// pointer types, so void* is binary-compatible on the same ABI.
struct RuntimeStateLayout {
  void *stream;
  void *miopen_handle;
  void *hipblas_handle;
  void *gpu_constants_blob;
  bool constants_blob_is_host;
  void **gpu_constants;
  size_t num_constants;
  void *pool_base;
  size_t pool_size;
  size_t *buffer_offsets;
  size_t num_buffers;
  void *workspace;
  size_t workspace_size;
  void *hipdnn_handle;
  void *hipdnn_graph_registry;
};

static int ensureWorkspace(RuntimeStateLayout *state, size_t needed_size) {
  if (!state || needed_size == 0)
    return 0;
  if (state->workspace_size >= needed_size)
    return 0;
  if (state->workspace) {
    hipFree(state->workspace);
    state->workspace = nullptr;
    state->workspace_size = 0;
  }
  if (hipMalloc(&state->workspace, needed_size) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_graph_runtime: workspace hipMalloc failed for %zu bytes\n",
            needed_size);
    return -1;
  }
  state->workspace_size = needed_size;
  return 0;
}

struct GraphRegistry {
  std::vector<std::unique_ptr<hip::graph::HipDNNGraph>> graphs;

  void store(int32_t id, hip::graph::HipDNNGraph *graph) {
    if (id < 0)
      return;
    if (static_cast<size_t>(id) >= graphs.size())
      graphs.resize(id + 1);
    graphs[id].reset(graph);
  }

  hip::graph::HipDNNGraph *lookup(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= graphs.size())
      return nullptr;
    return graphs[id].get();
  }
};

// Process-level defaults for the same-process singleton pattern.
static GraphRegistry *g_default_registry = nullptr;
static void *g_default_handle = nullptr;

} // namespace

extern "C" {

void *hipdnn_graph_create_handle() {
  hipdnnHandle_t handle = nullptr;
  hipdnnStatus_t status = hipdnnCreate(&handle);
  if (status != HIPDNN_STATUS_SUCCESS) {
    fprintf(stderr, "hipdnn_graph_create_handle: hipdnnCreate failed (%d)\n",
            static_cast<int>(status));
    return nullptr;
  }
  return handle;
}

void *hipdnn_graph_registry_create() { return new GraphRegistry(); }

void hipdnn_graph_registry_destroy(void *registry) {
  delete static_cast<GraphRegistry *>(registry);
}

void hipdnn_graph_registry_store(void *registry, int32_t graph_id,
                                 void *graph) {
  if (!registry || !graph)
    return;
  static_cast<GraphRegistry *>(registry)->store(
      graph_id, static_cast<hip::graph::HipDNNGraph *>(graph));
}

void hipdnn_graph_set_default_registry(void *registry) {
  g_default_registry = static_cast<GraphRegistry *>(registry);
}

void hipdnn_graph_set_default_handle(void *handle) {
  g_default_handle = handle;
}

void hipdnn_graph_runtime_attach(void *state_ptr, void *handle,
                                 void *registry) {
  if (!state_ptr)
    return;
  auto *state = static_cast<RuntimeStateLayout *>(state_ptr);
  state->hipdnn_handle = handle;
  state->hipdnn_graph_registry = registry;
}

int32_t hipdnn_graph_execute(void *state_ptr, int32_t graph_id, int32_t num_io,
                             int64_t *uids, void **ptrs) {
  if (!state_ptr || !uids || !ptrs)
    return -1;

  auto *state = static_cast<RuntimeStateLayout *>(state_ptr);

  auto *registry =
      static_cast<GraphRegistry *>(state->hipdnn_graph_registry);
  if (!registry)
    registry = g_default_registry;
  if (!registry)
    return -2;

  auto *graph = registry->lookup(graph_id);
  if (!graph)
    return -3;

  std::unordered_map<int64_t, void *> variant_pack;
  for (int32_t i = 0; i < num_io; ++i)
    variant_pack[uids[i]] = ptrs[i];

  int64_t ws_size = graph->getWorkspaceSize();
  if (ws_size > 0) {
    int ret = ensureWorkspace(state, static_cast<size_t>(ws_size));
    if (ret != 0) {
      fprintf(stderr,
              "hipdnn_graph_execute: workspace allocation failed for graph %d\n",
              graph_id);
      return -4;
    }
  }
  void *workspace = (ws_size > 0) ? state->workspace : nullptr;

  void *handle = state->hipdnn_handle;
  if (!handle)
    handle = g_default_handle;
  if (!handle)
    return -6;

  auto status = graph->Execute(static_cast<hipdnnHandle_t>(handle),
                               variant_pack, workspace);
  if (status.failed()) {
    fprintf(stderr, "hipdnn_graph_execute: graph %d failed: %s\n", graph_id,
            status.message().c_str());
    return -5;
  }

  return 0;
}

} // extern "C"
