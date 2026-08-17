/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

static int sync_count = 0;

extern "C" hipError_t hipMalloc(void **ptr, size_t size) {
  *ptr = std::malloc(size);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipFree(void *ptr) {
  std::free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int) {
  *ptr = std::malloc(size);
  return *ptr ? hipSuccess : -1;
}

extern "C" hipError_t hipHostFree(void *ptr) {
  std::free(ptr);
  return hipSuccess;
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t) {
  ++sync_count;
  return hipSuccess;
}

int main() {
  RuntimeState state{};
  state.stream = reinterpret_cast<hipStream_t>(1);
  state.legacy_pool_site_id = -1;

  void *parent = hipdnn_ep_get_pool_base(&state, 0, 0, 64);
  void *child = hipdnn_ep_get_pool_base(&state, 1, 0, 32);
  assert(parent != nullptr);
  assert(child != nullptr);
  assert(parent != child);

  std::memset(parent, 0x5a, 64);
  void *grown_child = hipdnn_ep_get_pool_base(&state, 1, 0, 128);
  assert(grown_child != nullptr);
  assert(state.pool_sites[0].pool_base[0] == parent);
  for (size_t i = 0; i < 64; ++i)
    assert(static_cast<unsigned char *>(parent)[i] == 0x5a);
  assert(sync_count == 1);

  void *parent_scratch = hipdnn_ep_get_host_scratch_base(&state, 0, 64);
  void *child_scratch = hipdnn_ep_get_host_scratch_base(&state, 1, 32);
  assert(parent_scratch != nullptr);
  assert(child_scratch != nullptr);
  assert(parent_scratch != child_scratch);

  std::memset(parent_scratch, 0xa5, 64);
  void *grown_child_scratch = hipdnn_ep_get_host_scratch_base(&state, 1, 128);
  assert(grown_child_scratch != nullptr);
  assert(state.host_scratch_sites[0].base == parent_scratch);
  for (size_t i = 0; i < 64; ++i)
    assert(static_cast<unsigned char *>(parent_scratch)[i] == 0xa5);
  assert(sync_count == 2);

  for (int site_id = 0; site_id < state.num_pool_sites; ++site_id) {
    RuntimePoolSite &site = state.pool_sites[site_id];
    for (int domain_id = 0; domain_id < site.num_domains; ++domain_id)
      std::free(site.pool_base[domain_id]);
    std::free(site.pool_base);
    std::free(site.pool_size);
  }
  std::free(state.pool_sites);
  for (int site_id = 0; site_id < state.num_host_scratch_sites; ++site_id)
    std::free(state.host_scratch_sites[site_id].base);
  std::free(state.host_scratch_sites);
  return 0;
}
