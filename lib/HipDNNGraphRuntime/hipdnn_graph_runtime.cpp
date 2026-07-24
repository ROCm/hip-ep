/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipdnn_graph_runtime.cpp - hipDNN graph registry + execute ---------===//
//
// C API for compiled hipDNN graph dispatch. See hipdnn_graph_runtime.h.
//
//===----------------------------------------------------------------------===//

#include "hipdnn_graph_runtime.h"
#include "../HipDNNGraph/HipDNNGraph.h"

#include <backend/hipdnn_backend.h>

#include <cstdio>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

// Error codes returned by hipdnn_graph_execute.
enum GraphExecError {
  kSuccess = 0,
  kNullArgs = -1,
  kNoRegistry = -2,
  kGraphNotFound = -3,
  kWorkspaceAlloc = -4,
  kExecuteFailed = -5,
  kNoHandle = -6,
};

// Layout-compatible mirror of RuntimeState (defined in
// runtime_state_internal.h). We cannot include the real header because
// runtime_types.h pulls in hip_runtime.h which contains MSVC-incompatible
// vector types. All handle fields (hipStream_t, miopenHandle_t,
// hipblasLtHandle_t) are pointer types, so void* is binary-compatible on the
// same ABI.
//
// IMPORTANT: this must match the field order and sizes of RuntimeState exactly.
// When RuntimeState's layout changes, update this mirror.
struct RuntimeStateLayout {
  void *stream;
  void *miopen_handle;
  void *hipblas_handle;
  void *gpu_constants_blob;
  void **gpu_constants;
  size_t num_constants;
  void *mm;                // MemoryManager* (opaque here)
  void *output_alloc_self; // hipdnn_output_allocator_t.self
  void *output_alloc_fn;   // hipdnn_output_allocator_t.allocate
  void *zp_unpack_cache;
  void *op_profile;
  void *device_error_flag; // int* (pointer-sized)
  void *hipdnn_handle;
  void *hipdnn_graph_registry;
};

// Owns compiled HipDNNGraph objects indexed by graph_id.
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
    return kNullArgs;

  auto *state = static_cast<RuntimeStateLayout *>(state_ptr);

  // Look up the graph registry -- fall back to process-level singleton.
  auto *registry = static_cast<GraphRegistry *>(state->hipdnn_graph_registry);
  if (!registry)
    registry = g_default_registry;
  if (!registry)
    return kNoRegistry;

  auto *graph = registry->lookup(graph_id);
  if (!graph)
    return kGraphNotFound;

  // Build the variant_pack mapping compile-time UIDs to runtime GPU pointers.
  std::unordered_map<int64_t, void *> variant_pack;
  for (int32_t i = 0; i < num_io; ++i)
    variant_pack[uids[i]] = ptrs[i];

  // Get the hipDNN handle -- fall back to process-level singleton.
  void *handle = state->hipdnn_handle;
  if (!handle)
    handle = g_default_handle;
  if (!handle)
    return kNoHandle;

  auto status = graph->Execute(static_cast<hipdnnHandle_t>(handle),
                               variant_pack, nullptr);
  if (status.failed()) {
    fprintf(stderr, "hipdnn_graph_execute: graph %d failed: %s\n", graph_id,
            status.message().c_str());
    return kExecuteFailed;
  }

  return kSuccess;
}

} // extern "C"
