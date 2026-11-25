/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Executes the real Page runtime wrapper against explicit CPU boundary stubs.
// Checks host validation, sticky errors and forwarding; never runs GPU math,
// an ONNX graph or a model. Native kernel correctness has separate HIP UTs.
#include "Kernels/hip/autotune/paged_attention/paged_attention_autotune.h"
#include "hipdnn_ep_runtime.h"
#include "op_profile.h"
#include "op_state.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int8_t hipdnn_ep_op_state_construct_paged_attention(RuntimeState *,
                                                               int32_t);

namespace {
int failures = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

struct Capture {
  int streamGets = 0, copies = 0, syncs = 0, queries = 0;
  int allocations = 0, launches = 0, prepares = 0;
  int hostErrors = 0, nativeReturn = 0, nativeStatus = 0;
  bool copyFails = false;
  hip_paged_attention_f16_desc desc{};
};
Capture cap;
int32_t sessionError = 0;
int stateToken = 0, streamToken = 0;
OpState *opState = nullptr;
alignas(256) std::array<unsigned char, 512> scratch{};

struct Fixture {
  // Small sentinels suffice for host-only validation: the native boundary
  // records the descriptor without reading or computing tensor elements.
  std::array<uint16_t, 8> q{}, k{}, v{}, kp{}, vp{}, output{};
  std::array<int32_t, 3> cum{{0, 1, 2}}, metadata{{1, 1, 0}};
  std::array<int32_t, 2> past{{0, 0}}, table{{0, 1}};
  int64_t tokens = 2, outputTokens = 2, valuePageSize = 256;
  bool data = true, withMetadata = true;
  Fixture() {
    q.fill(0x1111);
    k.fill(0x2222);
    v.fill(0x3333);
    kp.fill(0x4444);
    vp.fill(0x5555);
    output.fill(0x6666);
  }
  RuntimeState *state() {
    return reinterpret_cast<RuntimeState *>(&stateToken);
  }
  int call() {
    return wrap_paged_attention(
        state(), 0, data ? q.data() : nullptr, data ? k.data() : nullptr,
        data ? v.data() : nullptr, data ? kp.data() : nullptr,
        data ? vp.data() : nullptr, data ? cum.data() : nullptr,
        data ? past.data() : nullptr, data ? table.data() : nullptr, nullptr,
        nullptr, withMetadata ? metadata.data() : nullptr,
        data ? output.data() : nullptr, tokens, tokens, tokens, outputTokens, 2,
        2, 2, 256, valuePageSize, 256, 2, 1, 3, withMetadata ? 3 : 0, 0, 0, 24,
        4, -1, 0, 0, 0.0625f, 0.0f);
  }
  bool unchanged() const {
    for (size_t i = 0; i < q.size(); ++i)
      if (q[i] != 0x1111 || k[i] != 0x2222 || v[i] != 0x3333 ||
          kp[i] != 0x4444 || vp[i] != 0x5555 || output[i] != 0x6666)
        return false;
    return true;
  }
};

void reset() {
  cap = Capture{};
  sessionError = 0;
  scratch.fill(0);
}
void testExtentRejection() {
  for (bool outputMismatch : {false, true}) {
    reset();
    Fixture f;
    if (outputMismatch)
      f.outputTokens = 1;
    else
      f.valuePageSize = 128;
    CHECK(f.call() == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
    CHECK(cap.hostErrors == 1 && sessionError != 0);
    CHECK(cap.streamGets == 0 && cap.copies == 0 && cap.queries == 0);
    CHECK(cap.launches == 0 && cap.prepares == 0 && f.unchanged());
  }
}
void testStickyUpstreamError() {
  for (bool metadata : {false, true}) {
    reset();
    Fixture f;
    f.withMetadata = metadata;
    sessionError = 77;
    CHECK(f.call() == 77);
    CHECK(sessionError == 77 && cap.hostErrors == 0);
    CHECK(cap.copies == (metadata ? 4 : 3) && cap.syncs == 1);
    CHECK(cap.queries == 0 && cap.allocations == 0);
    CHECK(cap.launches == 0 && cap.prepares == 0);
    CHECK(f.unchanged());
  }
}
void testZeroTokens() {
  reset();
  Fixture f;
  f.tokens = f.outputTokens = 0;
  f.data = false;
  f.withMetadata = false;
  CHECK(f.call() == 0);
  CHECK(cap.queries == 1 && cap.streamGets == 0 && cap.copies == 0);
  CHECK(cap.allocations == 0 && cap.launches == 0 && cap.prepares == 0);
  CHECK(sessionError == 0 && f.unchanged());
}
void testRoutingAndStatus() {
  for (bool metadata : {false, true}) {
    reset();
    Fixture f;
    f.withMetadata = metadata;
    CHECK(f.call() == 0);
    CHECK(cap.launches == 1 && cap.prepares == 0);
    CHECK(cap.syncs == 2 && cap.copies == (metadata ? 5 : 4));
    CHECK(cap.desc.q == f.q.data() && cap.desc.k_new == f.k.data());
    CHECK(cap.desc.v_new == f.v.data() && cap.desc.o == f.output.data());
    CHECK(cap.desc.k_pool == f.kp.data() && cap.desc.v_pool == f.vp.data());
    CHECK(cap.desc.batch == 2 && cap.desc.tokens == 2);
    CHECK(cap.desc.num_heads == 24 && cap.desc.kv_num_heads == 4);
    CHECK(cap.desc.head_dim == 256 && cap.desc.page_size == 256);
    CHECK(cap.desc.phase == HIP_PAGED_ATTENTION_DECODE);
    CHECK(sessionError == 0 && f.unchanged());
  }
  reset();
  Fixture f;
  cap.nativeStatus = HIP_PAGED_ATTENTION_BAD_PAGE;
  CHECK(f.call() == HIP_PAGED_ATTENTION_BAD_PAGE); // Native status is consumed.
  CHECK(sessionError == HIP_PAGED_ATTENTION_BAD_PAGE);
  CHECK(f.unchanged());
  reset();
  cap.nativeReturn = HIP_PAGED_ATTENTION_UNSUPPORTED;
  CHECK(f.call() == HIP_PAGED_ATTENTION_UNSUPPORTED);
  CHECK(sessionError == HIP_PAGED_ATTENTION_UNSUPPORTED);
  reset();
  cap.copyFails = true;
  CHECK(f.call() == HIP_PAGED_ATTENTION_HIP_ERROR);
  CHECK(cap.launches == 0 && sessionError == HIP_PAGED_ATTENTION_HIP_ERROR);
}
void testRequestPhaseAndBounds() {
  reset();
  Fixture f;
  f.metadata = {{8, 100, 0}};
  CHECK(f.call() == 0);
  CHECK(cap.desc.phase == HIP_PAGED_ATTENTION_DECODE);
  reset();
  f.cum = {{0, 2, 2}};
  CHECK(f.call() == 0);
  CHECK(cap.desc.phase == HIP_PAGED_ATTENTION_PREFILL);
  reset();
  f.withMetadata = false;
  CHECK(f.call() == 0);
  CHECK(cap.desc.phase == HIP_PAGED_ATTENTION_PREFILL);
  reset();
  f.withMetadata = true;
  f.metadata = {{1, 100, 0}};
  CHECK(f.call() == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.launches == 0 && cap.prepares == 0 && cap.allocations == 0);
  CHECK(f.unchanged());
  reset();
  f.metadata = {{8, 1, 0}};
  CHECK(f.call() == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.launches == 0 && f.unchanged());
  reset();
  f.metadata = {{8, 100, 3}};
  CHECK(f.call() == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.launches == 0 && f.unchanged());
  reset();
  f.metadata = {{8, 100, 0}};
  f.cum = {{0, 3, 2}};
  CHECK(f.call() == HIP_PAGED_ATTENTION_BAD_CUM);
  CHECK(cap.launches == 0 && f.unchanged());
}

} // namespace

extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *) {
  ++cap.streamGets;
  return &streamToken;
}
extern "C" void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *) {
  return &sessionError;
}
extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *, int code) {
  ++cap.hostErrors;
  sessionError = code;
  return 0;
}
extern "C" int hipdnn_ep_state_ensure_paged_attention_scratch(RuntimeState *,
                                                              size_t) {
  ++cap.allocations;
  return 0;
}
extern "C" void *hipdnn_ep_state_get_paged_attention_scratch(RuntimeState *) {
  return scratch.data();
}
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *, int) { return opState; }
void hipdnn_ep_op_state_set(RuntimeState *, int32_t, OpState *p) {
  opState = p;
}
extern "C" void *hipdnn_ep_state_get_op_profile(RuntimeState *) {
  return nullptr;
}
extern "C" int hipdnn_ep_roctx_enabled() { return 0; }
extern "C" int hipdnn_ep_roctx_range_push(const char *) { return -1; }
extern "C" void hipdnn_ep_roctx_range_pop() {}
extern "C" hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return hipSuccess;
}
extern "C" hipError_t hipMemcpyAsync(void *dst, const void *src, size_t n, int,
                                     hipStream_t) {
  ++cap.copies;
  if (cap.copyFails)
    return 1;
  std::memcpy(dst, src, n);
  return hipSuccess;
}
extern "C" hipError_t hipStreamSynchronize(hipStream_t) {
  ++cap.syncs;
  return hipSuccess;
}
void op_profile_ensure_epoch(OpProfileState *, hipStream_t) {}
int op_profile_acquire_marker(OpProfileState *) { return 0; }
hipEvent_t op_profile_get_marker_event(OpProfileState *, int) {
  return nullptr;
}
bool op_profile_is_active(OpProfileState *) { return false; }
void op_profile_add_pending(OpProfileState *, const std::string &,
                            const std::string &, int, double, int64_t, double) {
}
GqaPagedConfig gqa_paged_default(const hip_paged_attention_f16_desc &) {
  return {};
}
extern "C" int
hip_paged_attention_f16_workspace_size(const hip_paged_attention_f16_desc *d,
                                       size_t *bytes) {
  ++cap.queries;
  *bytes = d->tokens ? 256 : 0;
  return 0;
}
extern "C" int hip_paged_attention_f16_configured_v2(
    void *, const hip_paged_attention_f16_desc *d, void *workspace, size_t,
    const GqaPagedConfig *, size_t) {
  ++cap.launches;
  cap.desc = *d;
  *static_cast<int32_t *>(workspace) = cap.nativeStatus;
  return cap.nativeReturn;
}

namespace hipdnn_ep {
PagedAttentionResult resolvePagedAttention(const hip_paged_attention_f16_desc &,
                                           const std::string &) {
  return {{}, "lookup:cpu-contract"};
}
int preparePagedAttention(void *, const hip_paged_attention_f16_desc &, void *,
                          size_t, const std::string &, bool,
                          PagedAttentionResult *) {
  ++cap.prepares;
  return -1;
}
} // namespace hipdnn_ep
int main() {
#ifdef _WIN32
  _putenv_s("HIPDNN_EP_PERF", "0");
  _putenv_s("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE", "lookup");
#else
  setenv("HIPDNN_EP_PERF", "0", 1);
  setenv("HIPDNN_PAGED_ATTENTION_AUTOTUNE_MODE", "lookup", 1);
#endif
  CHECK(hipdnn_ep_op_state_construct_paged_attention(
            reinterpret_cast<RuntimeState *>(&stateToken), 0) == 0);
  testExtentRejection();
  testStickyUpstreamError();
  testZeroTokens();
  testRoutingAndStatus();
  testRequestPhaseAndBounds();
  if (opState)
    opState->deletor(opState);
  if (failures)
    return 1;
  std::puts("Page runtime CPU contract: ALL PASS (host routing; no GPU math)");
  return 0;
}
