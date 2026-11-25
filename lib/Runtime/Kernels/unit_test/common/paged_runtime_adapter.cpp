/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ut_paged_runtime.h"
#include "hipdnn_ep_runtime.h"
#include "op_profile.h"
#include "op_state.h"
#include <stdexcept>

extern "C" int8_t hipdnn_ep_op_state_construct_paged_attention(RuntimeState *,
                                                              int32_t);
struct UtPagedRuntime {
  hipStream_t stream{};
  int32_t *error{};
  void *scratch{};
  size_t capacity{};
  OpState *op{};
};
static UtPagedRuntime *context(RuntimeState *p) {
  return reinterpret_cast<UtPagedRuntime *>(p);
}
static void checked(hipError_t rc) {
  if (rc != hipSuccess)
    throw std::runtime_error(hipGetErrorString(rc));
}
UtPagedRuntime *utPagedRuntimeCreate() {
  auto p = std::make_unique<UtPagedRuntime>();
  checked(hipStreamCreate(&p->stream));
  try {
    checked(hipMalloc(&p->error, sizeof(int32_t)));
    checked(hipMemsetAsync(p->error, 0, sizeof(int32_t), p->stream));
    hipdnn_ep_op_state_construct_paged_attention(
        reinterpret_cast<RuntimeState *>(p.get()), 0);
    return p.release();
  } catch (...) {
    utPagedRuntimeDestroy(p.release());
    throw;
  }
}
void utPagedRuntimeDestroy(UtPagedRuntime *p) {
  if (!p)
    return;
  checked(hipStreamSynchronize(p->stream));
  if (p->op)
    p->op->deletor(p->op);
  if (p->scratch)
    checked(hipFree(p->scratch));
  if (p->error)
    checked(hipFree(p->error));
  checked(hipStreamDestroy(p->stream));
  delete p;
}
void *utPagedRuntimeStream(UtPagedRuntime *p) { return p->stream; }
int utPagedRuntimeStatus(UtPagedRuntime *p) {
  int32_t status = 0;
  checked(hipMemcpyAsync(&status, p->error, sizeof(status),
                         hipMemcpyDeviceToHost, p->stream));
  checked(hipStreamSynchronize(p->stream));
  return status;
}
int utPagedRuntimeCall(UtPagedRuntime *p,
                       const hip_paged_attention_f16_desc &d,
                       const int32_t *metadata, int64_t count) {
  return wrap_paged_attention_v2(
      reinterpret_cast<RuntimeState *>(p), 0, d.q, d.k_new, d.v_new, d.k_pool,
      d.v_pool, d.cum, d.past_seqlens, d.block_table, d.cos_cache, d.sin_cache,
      metadata, d.o, d.tokens, d.tokens, d.tokens, d.tokens, d.batch,
      d.num_pages, d.num_pages, d.page_size, d.page_size, d.head_dim, d.batch,
      d.table_columns, d.batch + 1, count, d.max_position, d.max_position,
      d.num_heads, d.kv_num_heads, d.local_window_size, d.do_rotary,
      d.rotary_interleaved, d.scale);
}
extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *p) {
  return context(p)->stream;
}
extern "C" void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *p) {
  return context(p)->error;
}
extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *p, int value) {
  checked(hipMemcpyAsync(context(p)->error, &value, sizeof(value),
                         hipMemcpyHostToDevice, context(p)->stream));
  checked(hipStreamSynchronize(context(p)->stream));
  return 0;
}
extern "C" int hipdnn_ep_state_ensure_paged_attention_scratch(RuntimeState *s,
                                                            size_t bytes) {
  auto p = context(s);
  if (p->capacity < bytes) {
    checked(hipStreamSynchronize(p->stream));
    if (p->scratch)
      checked(hipFree(p->scratch));
    p->scratch = nullptr;
    p->capacity = 0;
    checked(hipMalloc(&p->scratch, bytes));
    p->capacity = bytes;
  }
  return 0;
}
extern "C" void *hipdnn_ep_state_get_paged_attention_scratch(RuntimeState *p) {
  return context(p)->scratch;
}
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *p, int slot) {
  return slot == 0 ? context(p)->op : nullptr;
}
void hipdnn_ep_op_state_set(RuntimeState *p, int32_t, OpState *value) {
  context(p)->op = value;
}
extern "C" void *hipdnn_ep_state_get_op_profile(RuntimeState *) { return nullptr; }
extern "C" int hipdnn_ep_roctx_enabled() { return 0; }
extern "C" int hipdnn_ep_roctx_range_push(const char *) { return -1; }
extern "C" void hipdnn_ep_roctx_range_pop() {}
void op_profile_ensure_epoch(OpProfileState *, hipStream_t) {}
int op_profile_acquire_marker(OpProfileState *) { return 0; }
hipEvent_t op_profile_get_marker_event(OpProfileState *, int) { return nullptr; }
bool op_profile_is_active(OpProfileState *) { return false; }
void op_profile_add_pending(OpProfileState *, const std::string &,
                            const std::string &, int, double, int64_t, double) {}
