/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

namespace {

int failures = 0;
std::atomic<int> recordedErrors{0};

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

struct Rank1Desc {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

struct Scenario {
  std::vector<std::vector<int64_t>> extents;
  int iteration = 0;
  int failIteration = -1;
  bool runNested = false;
  RuntimeState *state = nullptr;
};

thread_local Scenario *activeScenario = nullptr;
thread_local int specialIteration = 0;

extern "C" void hipdnn_ep_test_fail_next_loop_sync();
extern "C" void hipdnn_ep_test_fail_next_loop_alloc();

int body(RuntimeState *state, HipdnnEpLoopFrame *frame, void *, void *cond,
         void **current, void **, void **next) {
  Scenario &scenario = *activeScenario;
  int iteration = scenario.iteration++;
  if (iteration == scenario.failIteration)
    return -7;

  for (size_t carrier = 0; carrier < scenario.extents.size(); ++carrier) {
    auto *in = static_cast<Rank1Desc *>(current[carrier]);
    CHECK(hipdnn_ep_loop_frame_set_current(frame, static_cast<int32_t>(carrier),
                                           in->aligned) == 0);
    if (iteration > 0)
      CHECK(in->sizes[0] == scenario.extents[carrier][iteration - 1]);
    int64_t extent = scenario.extents[carrier][iteration];
    void *data = hipdnn_ep_loop_frame_alloc(
        frame, static_cast<int32_t>(carrier), &extent, 1, sizeof(float));
    if (hipdnn_ep_loop_frame_status(frame) != 0)
      return hipdnn_ep_loop_frame_status(frame);
    auto *out = static_cast<Rank1Desc *>(next[carrier]);
    *out = {data, data, 0, {extent}, {1}};
    CHECK(hipdnn_ep_loop_frame_publish(frame, static_cast<int32_t>(carrier),
                                       data) == 0);
  }

  if (scenario.runNested && iteration == 0) {
    Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
    Rank1Desc scratch{};
    void *initial[] = {&seed};
    void *nextSet[] = {&scratch};
    void **final = nullptr;
    HipdnnEpLoopFrame *nestedFrame = nullptr;
    Scenario nested{{{3}}, 0, -1, false, state};
    Scenario *saved = activeScenario;
    activeScenario = &nested;
    int rc = hipdnn_ep_run_counted_loop(state, body, 1, true, 1, 0, initial,
                                        nextSet, nextSet, nullptr, frame,
                                        &final, &nestedFrame);
    activeScenario = saved;
    CHECK(rc == 0);
    CHECK(static_cast<Rank1Desc *>(final[0])->sizes[0] == 3);
  }

  // Dynamic-loop tests stop after the second successful body.
  if (cond && scenario.extents.front().size() == 2 && iteration == 1)
    *static_cast<int8_t *>(cond) = 0;
  return 0;
}

RuntimeState makeState() {
  RuntimeState state{};
  CHECK(hipdnn_ep_loop_state_init(&state) == 0);
  return state;
}

void cleanup(RuntimeState &state) {
  hipdnn_ep_loop_cleanup_quarantined_frames(&state);
  hipdnn_ep_loop_state_cleanup(&state);
}

void testZeroTripAliasesSeed() {
  RuntimeState state = makeState();
  Rank1Desc seed{reinterpret_cast<void *>(0x1234),
                 reinterpret_cast<void *>(0x1234),
                 0,
                 {0},
                 {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  CHECK(hipdnn_ep_run_counted_loop(&state, body, 0, true, 1, 0, initial, next,
                                   next, nullptr, nullptr, &final,
                                   &frame) == 0);
  CHECK(final == initial);
  CHECK(final[0] == &seed);
  CHECK(frame == nullptr);
  cleanup(state);
}

void testGrowShrinkRegrowAndMultipleCarriers() {
  RuntimeState state = makeState();
  Rank1Desc seed0{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc seed1{nullptr, nullptr, 0, {7}, {1}};
  Rank1Desc scratch0{}, scratch1{};
  void *initial[] = {&seed0, &seed1};
  void *next[] = {&scratch0, &scratch1};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario scenario{{{2, 5, 1, 4}, {9, 3, 8, 2}}, 0, -1, false, &state};
  activeScenario = &scenario;
  CHECK(hipdnn_ep_run_counted_loop(&state, body, 4, true, 2, 0, initial, next,
                                   next, nullptr, nullptr, &final,
                                   &frame) == 0);
  CHECK(static_cast<Rank1Desc *>(final[0])->sizes[0] == 4);
  CHECK(static_cast<Rank1Desc *>(final[1])->sizes[0] == 2);
  CHECK(static_cast<Rank1Desc *>(final[0])->aligned != nullptr);
  CHECK(hipdnn_ep_loop_frame_destroy(&state, frame) == 0);
  cleanup(state);
}

void testFailureIsTransactional() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario scenario{{{4, 8, 12}}, 0, 1, false, &state};
  activeScenario = &scenario;
  CHECK(hipdnn_ep_run_counted_loop(&state, body, 3, true, 1, 0, initial, next,
                                   next, nullptr, nullptr, &final,
                                   &frame) == -7);
  CHECK(final == initial);
  CHECK(frame == nullptr);
  CHECK(recordedErrors.load() > 0);
  cleanup(state);
}

int overflowBody(RuntimeState *, HipdnnEpLoopFrame *frame, void *, void *,
                 void **, void **, void **) {
  int64_t shape[2] = {std::numeric_limits<int64_t>::max(), 2};
  (void)hipdnn_ep_loop_frame_alloc(frame, 0, shape, 2, 8);
  return hipdnn_ep_loop_frame_status(frame);
}

int passThenAllocateBody(RuntimeState *, HipdnnEpLoopFrame *frame, void *,
                         void *, void **current, void **, void **next) {
  auto *in = static_cast<Rank1Desc *>(current[0]);
  CHECK(hipdnn_ep_loop_frame_set_current(frame, 0, in->aligned) == 0);
  auto *out = static_cast<Rank1Desc *>(next[0]);
  if (specialIteration++ == 0) {
    *out = *in;
    CHECK(hipdnn_ep_loop_frame_publish(frame, 0, in->aligned) == 0);
    return 0;
  }
  int64_t extent = 6;
  void *data = hipdnn_ep_loop_frame_alloc(frame, 0, &extent, 1, sizeof(float));
  *out = {data, data, 0, {extent}, {1}};
  CHECK(hipdnn_ep_loop_frame_publish(frame, 0, data) == 0);
  return hipdnn_ep_loop_frame_status(frame);
}

int partialMultiCarrierFailure(RuntimeState *, HipdnnEpLoopFrame *frame, void *,
                               void *, void **current, void **, void **next) {
  for (int32_t i = 0; i < 2; ++i) {
    auto *in = static_cast<Rank1Desc *>(current[i]);
    CHECK(hipdnn_ep_loop_frame_set_current(frame, i, in->aligned) == 0);
  }
  int64_t extent = 4;
  void *data = hipdnn_ep_loop_frame_alloc(frame, 0, &extent, 1, sizeof(float));
  *static_cast<Rank1Desc *>(next[0]) = {data, data, 0, {extent}, {1}};
  CHECK(hipdnn_ep_loop_frame_publish(frame, 0, data) == 0);
  return -9;
}

thread_local int allocationDispatches = 0;
int allocationFailureAfterEvolution(RuntimeState *, HipdnnEpLoopFrame *frame,
                                    void *, void *, void **current, void **,
                                    void **next) {
  auto *in = static_cast<Rank1Desc *>(current[0]);
  CHECK(hipdnn_ep_loop_frame_set_current(frame, 0, in->aligned) == 0);
  int64_t extent = allocationDispatches == 0 ? 2 : 9;
  if (allocationDispatches == 1)
    hipdnn_ep_test_fail_next_loop_alloc();
  void *data = hipdnn_ep_loop_frame_alloc(frame, 0, &extent, 1, sizeof(float));
  if (hipdnn_ep_loop_frame_status(frame) != 0)
    return hipdnn_ep_loop_frame_status(frame);
  ++allocationDispatches;
  *static_cast<Rank1Desc *>(next[0]) = {data, data, 0, {extent}, {1}};
  CHECK(hipdnn_ep_loop_frame_publish(frame, 0, data) == 0);
  return 0;
}

void testAllocationOverflow() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  CHECK(hipdnn_ep_run_counted_loop(&state, overflowBody, 1, true, 1, 0, initial,
                                   next, next, nullptr, nullptr, &final,
                                   &frame) != 0);
  CHECK(final == initial);
  cleanup(state);
}

void testNestedAndConcurrentFrames() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario outer{{{2}}, 0, -1, true, &state};
  activeScenario = &outer;
  CHECK(hipdnn_ep_run_counted_loop(&state, body, 1, true, 1, 0, initial, next,
                                   next, nullptr, nullptr, &final,
                                   &frame) == 0);

  auto run = [&state](int64_t extent) {
    Rank1Desc localSeed{nullptr, nullptr, 0, {0}, {1}};
    Rank1Desc localScratch{};
    void *localInitial[] = {&localSeed};
    void *localNext[] = {&localScratch};
    void **localFinal = nullptr;
    HipdnnEpLoopFrame *localFrame = nullptr;
    Scenario scenario{{{extent}}, 0, -1, false, &state};
    activeScenario = &scenario;
    int rc = hipdnn_ep_run_counted_loop(
        &state, body, 1, true, 1, 0, localInitial, localNext, localNext,
        nullptr, nullptr, &localFinal, &localFrame);
    CHECK(rc == 0);
    CHECK(static_cast<Rank1Desc *>(localFinal[0])->sizes[0] == extent);
    CHECK(hipdnn_ep_loop_frame_destroy(&state, localFrame) == 0);
  };
  std::thread first(run, 11);
  std::thread second(run, 13);
  first.join();
  second.join();
  CHECK(hipdnn_ep_loop_frame_destroy(&state, frame) == 0);
  cleanup(state);
}

void testPassThroughThenAllocateAndAtomicFailure() {
  RuntimeState state = makeState();
  Rank1Desc seed{reinterpret_cast<void *>(0x1234),
                 reinterpret_cast<void *>(0x1234),
                 0,
                 {2},
                 {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  specialIteration = 0;
  CHECK(hipdnn_ep_run_counted_loop(&state, passThenAllocateBody, 2, true, 1, 0,
                                   initial, next, next, nullptr, nullptr,
                                   &final, &frame) == 0);
  auto *result = static_cast<Rank1Desc *>(final[0]);
  CHECK(result->sizes[0] == 6);
  CHECK(result->aligned != seed.aligned);
  CHECK(hipdnn_ep_loop_frame_destroy(&state, frame) == 0);

  Rank1Desc seed1{reinterpret_cast<void *>(0x5678),
                  reinterpret_cast<void *>(0x5678),
                  0,
                  {3},
                  {1}};
  Rank1Desc scratch0{}, scratch1{};
  void *initial2[] = {&seed, &seed1};
  void *next2[] = {&scratch0, &scratch1};
  final = nullptr;
  frame = nullptr;
  CHECK(hipdnn_ep_run_counted_loop(&state, partialMultiCarrierFailure, 1, true,
                                   2, 0, initial2, next2, next2, nullptr,
                                   nullptr, &final, &frame) == -9);
  CHECK(final == initial2);
  CHECK(frame == nullptr);
  cleanup(state);
}

void testEarlyExitAndSyncFailureCleanup() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratch{};
  void *initial[] = {&seed};
  void *next[] = {&scratch};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario early{{{2, 5}}, 0, -1, false, &state};
  activeScenario = &early;
  CHECK(hipdnn_ep_run_loop(&state, body, 8, true, 1, 0, initial, next, next,
                           nullptr, nullptr, &final, &frame) == 0);
  CHECK(early.iteration == 2);
  CHECK(static_cast<Rank1Desc *>(final[0])->sizes[0] == 5);

  hipdnn_ep_test_fail_next_loop_sync();
  CHECK(hipdnn_ep_loop_frame_destroy(&state, frame) != 0);
  CHECK(recordedErrors.load() > 0);
  // The failed destroy quarantines only this frame. State cleanup retries a
  // successful sync and releases it without touching any active invocation.
  cleanup(state);
}

void testInjectedAllocationFailureAfterEvolution() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratchA{}, scratchB{};
  void *initial[] = {&seed};
  void *nextA[] = {&scratchA};
  void *nextB[] = {&scratchB};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  allocationDispatches = 0;
  CHECK(hipdnn_ep_run_counted_loop(&state, allocationFailureAfterEvolution, 3,
                                   true, 1, 0, initial, nextA, nextB, nullptr,
                                   nullptr, &final, &frame) != 0);
  CHECK(allocationDispatches == 1);
  CHECK(final == initial);
  CHECK(frame == nullptr);
  cleanup(state);
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) {
  ++recordedErrors;
  return 0;
}

int main() {
  testZeroTripAliasesSeed();
  testGrowShrinkRegrowAndMultipleCarriers();
  testFailureIsTransactional();
  testAllocationOverflow();
  testNestedAndConcurrentFrames();
  testPassThroughThenAllocateAndAtomicFailure();
  testEarlyExitAndSyncFailureCleanup();
  testInjectedAllocationFailureAfterEvolution();
  if (failures == 0) {
    std::printf("loop frame unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "loop frame unit test: %d FAILURE(S)\n", failures);
  return 1;
}
