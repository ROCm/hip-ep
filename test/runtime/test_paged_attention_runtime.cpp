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
  int mropeQueries = 0, mropeLaunches = 0, mropePrepares = 0;
  int hostErrors = 0, nativeReturn = 0, nativeStatus = 0;
  bool copyFails = false, lookupHit = true;
  std::string lookupKey;
  hip_paged_attention_f16_desc desc{};
  hip_paged_attention_mrope_f16_desc mrope{};
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
  std::array<float, 8> mq{}, mk{}, cos{}, sin{};
  std::array<int64_t, 6> positions{{5, 2, 17, 11, 3, 0}};
  int64_t tokens = 2, outputTokens = 2, valuePageSize = 256;
  bool data = true, withMetadata = true;
  int64_t window = -1, doRotary = 0, dimension = 256;
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
    return wrap_paged_attention_v2(
        state(), 0, data ? q.data() : nullptr, data ? k.data() : nullptr,
        data ? v.data() : nullptr, data ? kp.data() : nullptr,
        data ? vp.data() : nullptr, data ? cum.data() : nullptr,
        data ? past.data() : nullptr, data ? table.data() : nullptr,
        doRotary ? q.data() : nullptr, doRotary ? k.data() : nullptr,
        withMetadata ? metadata.data() : nullptr,
        data ? output.data() : nullptr, tokens, tokens, tokens, outputTokens, 2,
        2, 2, 256, valuePageSize, dimension, 2, 1, 3, withMetadata ? 3 : 0,
        doRotary ? 32 : 0, doRotary ? 32 : 0, 24, 4, window, doRotary, 0,
        0.0625f);
  }
  int mropeCall(int64_t positionTokens = 2, int64_t layout = 1,
                int64_t rotary = 64) {
    return wrap_paged_attention_mrope_v2(
        state(), 0, mq.data(), mk.data(), v.data(), kp.data(), vp.data(),
        cum.data(), past.data(), table.data(), cos.data(), sin.data(),
        metadata.data(), output.data(), 2, 2, 2, 2, 2, 2, 2, 256, 256,
        dimension, 2, 1, 3, 3, 32, 32, 24, 4, window, 0, 0, 0.0625f,
        positions.data(), positionTokens, rotary, 11, 11, 10, layout, 0.5f);
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

void testRotaryWindowForwarding() {
  for (bool prefill : {false, true})
    for (int64_t window : {-1, 0, 1, 37, 4096})
      for (int64_t rotary : {0, 1}) {
        reset();
        Fixture f;
        f.window = window;
        f.doRotary = rotary;
        f.dimension = 128;
        if (prefill) {
          f.cum = {{0, 2, 2}};
          f.metadata = {{2, 2, 0}};
        }
        CHECK(f.call() == 0);
        CHECK(cap.desc.local_window_size == window &&
              cap.desc.do_rotary == rotary);
        CHECK(cap.desc.softcap ==
              0); // Legacy native descriptor stays zero-initialized.
        CHECK(cap.desc.phase == (prefill ? HIP_PAGED_ATTENTION_PREFILL
                                         : HIP_PAGED_ATTENTION_DECODE));
        CHECK(f.unchanged());
        reset();
        CHECK(f.mropeCall() == 0);
        CHECK(cap.mrope.page.local_window_size == window &&
              cap.mrope.page.do_rotary == 0);
        CHECK(cap.mrope.page.softcap == 0);
      }
}

void testMropeBoundary() {
  reset();
  Fixture f;
  CHECK(f.mropeCall() == 0);
  CHECK(cap.mropeQueries == 1 && cap.mropeLaunches == 1);
  CHECK(cap.lookupKey.rfind(
            "decode_mrope_f32_f16_r64_sections11-11-10_layout1_rscale0.5_",
            0) == 0);
  CHECK(cap.mrope.q == f.mq.data() && cap.mrope.k_new == f.mk.data());
  CHECK(cap.mrope.cos_cache == f.cos.data() &&
        cap.mrope.sin_cache == f.sin.data());
  CHECK(cap.mrope.position_ids == f.positions.data());
  CHECK(cap.mrope.rotary_dim == 64 && cap.mrope.max_position == 32);
  CHECK(cap.mrope.section_t == 11 && cap.mrope.section_h == 11 &&
        cap.mrope.section_w == 10);
  CHECK(cap.mrope.mrope_layout == 1 && cap.mrope.rotary_scale == 0.5f);
  CHECK(cap.mrope.page.do_rotary == 0 && cap.mrope.page.k_pool == f.kp.data());
  CHECK(cap.mrope.page.phase == HIP_PAGED_ATTENTION_DECODE && f.unchanged());
  reset();
  CHECK(f.mropeCall(1) == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.streamGets == 0 && cap.queries == 0 && cap.mropeLaunches == 0);
  reset();
  CHECK(f.mropeCall(2, 2) == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.streamGets == 0 && cap.queries == 0);
  reset();
  CHECK(f.mropeCall(2, 1, 32) == HIP_PAGED_ATTENTION_INVALID_ARGUMENT);
  CHECK(cap.streamGets == 0 && cap.queries == 0);
  reset();
  sessionError = 77;
  CHECK(f.mropeCall() == 77);
  CHECK(cap.queries == 0 && cap.mropeLaunches == 0 && f.unchanged());
  reset();
  cap.nativeStatus = HIP_PAGED_ATTENTION_BAD_PAGE;
  CHECK(f.mropeCall() == HIP_PAGED_ATTENTION_BAD_PAGE);
  CHECK(sessionError == HIP_PAGED_ATTENTION_BAD_PAGE && f.unchanged());
  reset();
  f.cum = {{0, 2, 2}};
  f.metadata = {{2, 2, 0}};
  CHECK(f.mropeCall() == 0);
  CHECK(cap.lookupKey.rfind("prefill_mrope_f32_f16_r64_", 0) == 0);
  CHECK(cap.mrope.page.phase == HIP_PAGED_ATTENTION_PREFILL);
  reset();
  cap.lookupHit = false;
  CHECK(f.mropeCall() == 0);
  CHECK(cap.mropePrepares == 1 && cap.mropeLaunches == 0);
  CHECK(cap.mrope.position_ids == f.positions.data() && f.unchanged());
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
extern "C" int hip_paged_attention_mrope_f16_workspace_size(
    const hip_paged_attention_mrope_f16_desc *d, size_t *bytes) {
  ++cap.mropeQueries;
  return hip_paged_attention_f16_workspace_size(&d->page, bytes);
}
extern "C" int hip_paged_attention_f16_configured_v2(
    void *, const hip_paged_attention_f16_desc *d, void *workspace, size_t,
    const GqaPagedConfig *, size_t) {
  ++cap.launches;
  cap.desc = *d;
  *static_cast<int32_t *>(workspace) = cap.nativeStatus;
  return cap.nativeReturn;
}
extern "C" int hip_paged_attention_mrope_f16_configured_v1(
    void *, const hip_paged_attention_mrope_f16_desc *d, void *workspace,
    size_t, const GqaPagedConfig *, size_t) {
  ++cap.mropeLaunches;
  cap.mrope = *d;
  *static_cast<int32_t *>(workspace) = cap.nativeStatus;
  return cap.nativeReturn;
}
extern "C" int hip_paged_attention_mrope_f16_prepare_v1(
    void *, const hip_paged_attention_mrope_f16_desc *d, void *workspace,
    size_t, GqaPagedConfig *, size_t) {
  ++cap.mropePrepares;
  cap.mrope = *d;
  *static_cast<int32_t *>(workspace) = cap.nativeStatus;
  return cap.nativeReturn;
}

namespace hipdnn_ep {
PagedAttentionResult resolvePagedAttention(const hip_paged_attention_f16_desc &,
                                           const std::string &key) {
  cap.lookupKey = key;
  return {{}, cap.lookupHit ? "lookup:cpu-contract" : "fallback:absent"};
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
  testMropeBoundary();
  testRotaryWindowForwarding();
  if (opState)
    opState->deletor(opState);
  if (failures)
    return 1;
  std::puts("Page runtime CPU contract: ALL PASS (host routing; no GPU math)");
  return 0;
}
