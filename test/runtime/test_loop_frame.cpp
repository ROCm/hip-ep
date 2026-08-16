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
  std::vector<std::vector<void *>> allocations;
  void *nestedAllocation = nullptr;
};

thread_local Scenario *activeScenario = nullptr;
thread_local int specialIteration = 0;

struct ConcurrentScenario {
  RuntimeState *state;
  int64_t extent;
  std::atomic<int> *ready;
  std::atomic<bool> *release;
  void *allocation = nullptr;
};

thread_local ConcurrentScenario *activeConcurrentScenario = nullptr;

extern "C" void hipdnn_ep_test_fail_next_loop_sync();
extern "C" void hipdnn_ep_test_fail_next_loop_alloc();
extern "C" void hipdnn_ep_test_fail_loop_state_allocation(int allocation);

int body(RuntimeState *state, HipdnnEpLoopFrame *frame, void *, void *cond,
         void **current, void **, void **next) {
  Scenario &scenario = *activeScenario;
  int iteration = scenario.iteration++;
  if (iteration == scenario.failIteration)
    return -7;

  if (scenario.allocations.empty())
    scenario.allocations.resize(scenario.extents.size());
  for (size_t carrier = 0; carrier < scenario.extents.size(); ++carrier) {
    auto *in = static_cast<Rank1Desc *>(current[carrier]);
    CHECK(hipdnn_ep_loop_frame_set_current(frame, static_cast<int32_t>(carrier),
                                           in->aligned) == 0);
    if (iteration > 0) {
      CHECK(in->sizes[0] == scenario.extents[carrier][iteration - 1]);
      CHECK(in->allocated == scenario.allocations[carrier][iteration - 1]);
      CHECK(in->aligned == scenario.allocations[carrier][iteration - 1]);
    }
    int64_t extent = scenario.extents[carrier][iteration];
    void *data = hipdnn_ep_loop_frame_alloc(
        frame, static_cast<int32_t>(carrier), &extent, 1, sizeof(float));
    if (hipdnn_ep_loop_frame_status(frame) != 0)
      return hipdnn_ep_loop_frame_status(frame);
    scenario.allocations[carrier].push_back(data);
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
    if (!nested.allocations.empty() && !nested.allocations[0].empty())
      scenario.nestedAllocation = nested.allocations[0][0];
  }

  // Dynamic-loop tests stop after the second successful body.
  if (cond && scenario.extents.front().size() == 2 && iteration == 1)
    *static_cast<int8_t *>(cond) = 0;
  return 0;
}

int concurrentBody(RuntimeState *, HipdnnEpLoopFrame *frame, void *, void *,
                   void **current, void **, void **next) {
  ConcurrentScenario &scenario = *activeConcurrentScenario;
  auto *in = static_cast<Rank1Desc *>(current[0]);
  CHECK(hipdnn_ep_loop_frame_set_current(frame, 0, in->aligned) == 0);
  scenario.allocation =
      hipdnn_ep_loop_frame_alloc(frame, 0, &scenario.extent, 1, sizeof(float));
  auto *out = static_cast<Rank1Desc *>(next[0]);
  *out = {scenario.allocation, scenario.allocation, 0, {scenario.extent}, {1}};
  CHECK(hipdnn_ep_loop_frame_publish(frame, 0, scenario.allocation) == 0);
  ++*scenario.ready;
  while (!scenario.release->load())
    std::this_thread::yield();
  return hipdnn_ep_loop_frame_status(frame);
}

RuntimeState makeState() {
  RuntimeState state{};
  CHECK(hipdnn_ep_loop_state_init(&state) == 0);
  return state;
}

void cleanup(RuntimeState &state) { hipdnn_ep_loop_state_cleanup(&state); }

HipdnnEpLoopBankStats getStats(RuntimeState &state) {
  HipdnnEpLoopBankStats stats{};
  hipdnn_ep_test_get_loop_bank_stats(&state, &stats);
  return stats;
}

void testStateInitAllocationFailureIsAtomic() {
  for (int allocation = 0; allocation < 2; ++allocation) {
    RuntimeState state{};
    hipdnn_ep_test_fail_loop_state_allocation(allocation);
    CHECK(hipdnn_ep_loop_state_init(&state) != 0);
    CHECK(state.loop_frames_mutex == nullptr);
    CHECK(state.loop_bank_cache == nullptr);
    CHECK(state.quarantined_loop_frames == nullptr);
  }

  RuntimeState state = makeState();
  CHECK(state.loop_frames_mutex != nullptr);
  CHECK(state.loop_bank_cache != nullptr);
  cleanup(state);
}

std::vector<void *> runCarrierFrame(RuntimeState &state,
                                    const std::vector<int64_t> &extents) {
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratchA{}, scratchB{};
  void *initial[] = {&seed};
  void *nextA[] = {&scratchA};
  void *nextB[] = {&scratchB};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario scenario{{extents}, 0, -1, false, &state};
  activeScenario = &scenario;
  CHECK(hipdnn_ep_run_counted_loop(
            &state, body, static_cast<int64_t>(extents.size()), true, 1, 0,
            initial, nextA, nextB, nullptr, nullptr, &final, &frame) == 0);
  CHECK(frame != nullptr);
  if (frame)
    CHECK(hipdnn_ep_loop_frame_destroy(&state, frame) == 0);
  if (scenario.allocations.empty())
    return {};
  return scenario.allocations[0];
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

void testZeroSizeDoesNotCheckoutBank() {
  RuntimeState state = makeState();
  std::vector<void *> allocations = runCarrierFrame(state, {0});
  CHECK(allocations.size() == 1);
  if (allocations.size() == 1)
    CHECK(allocations[0] == nullptr);
  HipdnnEpLoopBankStats stats = getStats(state);
  CHECK(stats.hits == 0);
  CHECK(stats.misses == 0);
  CHECK(stats.allocations == 0);
  CHECK(stats.active_bytes == 0);
  CHECK(stats.cached_bytes == 0);
  cleanup(state);
}

void testSequentialGrowingFramesReuseTwoBlocks() {
  RuntimeState state = makeState();
  std::vector<void *> first = runCarrierFrame(state, {4, 16, 8, 32});
  CHECK(first.size() == 4);
  if (first.size() == 4) {
    CHECK(first[0] != first[2]);
    CHECK(first[1] != first[3]);
    CHECK(first[2] != first[3]);
  }
  for (int frame = 1; frame < 27; ++frame) {
    std::vector<void *> current = runCarrierFrame(state, {4, 16, 8, 32});
    CHECK(current.size() == 4);
    if (current.size() == 4 && first.size() == 4) {
      CHECK(current[0] == first[2]);
      CHECK(current[1] == first[3]);
      CHECK(current[2] == first[2]);
      CHECK(current[3] == first[3]);
    }
  }
  HipdnnEpLoopBankStats stats = getStats(state);
  CHECK(stats.hits == 52);
  CHECK(stats.misses == 4);
  CHECK(stats.allocations == 4);
  CHECK(stats.frees == 2);
  CHECK(stats.active_bytes == 0);
  CHECK(stats.cached_bytes == 160);
  CHECK(stats.peak_bytes == 160);
  cleanup(state);

  HipdnnEpLoopBankStats cleaned{};
  hipdnn_ep_test_get_last_cleaned_loop_bank_stats(&cleaned);
  CHECK(cleaned.allocations == 4);
  CHECK(cleaned.frees == 4);
  CHECK(cleaned.active_bytes == 0);
  CHECK(cleaned.cached_bytes == 0);
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

void testZeroSizePublicationAndBankReuse() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratchA{}, scratchB{};
  void *initial[] = {&seed};
  void *nextA[] = {&scratchA};
  void *nextB[] = {&scratchB};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  Scenario scenario{{{2, 5, 0, 4}}, 0, -1, false, &state};
  int errorsBefore = recordedErrors.load();
  activeScenario = &scenario;
  CHECK(hipdnn_ep_run_counted_loop(&state, body, 4, true, 1, 0, initial, nextA,
                                   nextB, nullptr, nullptr, &final,
                                   &frame) == 0);
  CHECK(frame != nullptr);
  CHECK(hipdnn_ep_loop_frame_status(frame) == 0);
  CHECK(recordedErrors.load() == errorsBefore);
  CHECK(scenario.allocations.size() == 1);
  void *expectedFinal = nullptr;
  if (scenario.allocations.size() == 1) {
    const auto &allocations = scenario.allocations[0];
    CHECK(allocations.size() == 4);
    if (allocations.size() == 4) {
      CHECK(allocations[0] != nullptr);
      CHECK(allocations[1] != nullptr);
      CHECK(allocations[0] != allocations[1]);
      CHECK(allocations[2] == nullptr);
      // Publishing zero bytes advances the logical bank without releasing
      // either retained allocation, so extent 4 reuses the extent-5 bank.
      CHECK(allocations[3] == allocations[1]);
      expectedFinal = allocations[3];
    }
  }
  auto *result = static_cast<Rank1Desc *>(final[0]);
  CHECK(result->sizes[0] == 4);
  CHECK(result->allocated != nullptr);
  CHECK(result->allocated == result->aligned);
  CHECK(result->allocated == expectedFinal);
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

thread_local int growthSyncDispatches = 0;
int growthSyncFailure(RuntimeState *, HipdnnEpLoopFrame *frame, void *, void *,
                      void **current, void **, void **next) {
  auto *in = static_cast<Rank1Desc *>(current[0]);
  CHECK(hipdnn_ep_loop_frame_set_current(frame, 0, in->aligned) == 0);
  const int64_t extents[] = {2, 3, 9};
  int64_t extent = extents[growthSyncDispatches];
  if (growthSyncDispatches == 2)
    hipdnn_ep_test_fail_next_loop_sync();
  void *data = hipdnn_ep_loop_frame_alloc(frame, 0, &extent, 1, sizeof(float));
  if (hipdnn_ep_loop_frame_status(frame) != 0)
    return hipdnn_ep_loop_frame_status(frame);
  ++growthSyncDispatches;
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
  CHECK(!outer.allocations.empty());
  CHECK(outer.nestedAllocation != nullptr);
  if (!outer.allocations.empty() && !outer.allocations[0].empty())
    CHECK(outer.allocations[0][0] != outer.nestedAllocation);

  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  void *concurrentAllocations[2] = {};
  auto run = [&state, &ready, &release,
              &concurrentAllocations](int index, int64_t extent) {
    Rank1Desc localSeed{nullptr, nullptr, 0, {0}, {1}};
    Rank1Desc localScratch{};
    void *localInitial[] = {&localSeed};
    void *localNext[] = {&localScratch};
    void **localFinal = nullptr;
    HipdnnEpLoopFrame *localFrame = nullptr;
    ConcurrentScenario scenario{&state, extent, &ready, &release};
    activeConcurrentScenario = &scenario;
    int rc = hipdnn_ep_run_counted_loop(
        &state, concurrentBody, 1, true, 1, 0, localInitial, localNext,
        localNext, nullptr, nullptr, &localFinal, &localFrame);
    CHECK(rc == 0);
    CHECK(static_cast<Rank1Desc *>(localFinal[0])->sizes[0] == extent);
    concurrentAllocations[index] = scenario.allocation;
    CHECK(hipdnn_ep_loop_frame_destroy(&state, localFrame) == 0);
  };
  std::thread first(run, 0, 11);
  std::thread second(run, 1, 13);
  while (ready.load() != 2)
    std::this_thread::yield();
  release = true;
  first.join();
  second.join();
  CHECK(concurrentAllocations[0] != nullptr);
  CHECK(concurrentAllocations[1] != nullptr);
  CHECK(concurrentAllocations[0] != concurrentAllocations[1]);
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
  HipdnnEpLoopBankStats quarantined = getStats(state);
  CHECK(quarantined.active_bytes == 28);
  CHECK(quarantined.cached_bytes == 0);
  CHECK(quarantined.quarantined_bytes == 28);
  // The failed destroy quarantines only this frame. A later successful graph
  // sync proves its banks safe to recycle without touching active invocations.
  CHECK(hipStreamSynchronize(static_cast<hipStream_t>(state.stream)) ==
        hipSuccess);
  hipdnn_ep_loop_cleanup_quarantined_frames(&state);
  HipdnnEpLoopBankStats recycled = getStats(state);
  CHECK(recycled.active_bytes == 0);
  CHECK(recycled.cached_bytes == 28);
  CHECK(recycled.quarantined_bytes == 0);
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
  HipdnnEpLoopBankStats stats = getStats(state);
  CHECK(stats.misses == 2);
  CHECK(stats.allocations == 1);
  CHECK(stats.active_bytes == 0);
  CHECK(stats.cached_bytes == 8);
  cleanup(state);
}

void testGrowthSyncFailureQuarantinesBanks() {
  RuntimeState state = makeState();
  Rank1Desc seed{nullptr, nullptr, 0, {0}, {1}};
  Rank1Desc scratchA{}, scratchB{};
  void *initial[] = {&seed};
  void *nextA[] = {&scratchA};
  void *nextB[] = {&scratchB};
  void **final = nullptr;
  HipdnnEpLoopFrame *frame = nullptr;
  growthSyncDispatches = 0;
  CHECK(hipdnn_ep_run_counted_loop(&state, growthSyncFailure, 3, true, 1, 0,
                                   initial, nextA, nextB, nullptr, nullptr,
                                   &final, &frame) != 0);
  CHECK(growthSyncDispatches == 2);
  CHECK(final == initial);
  CHECK(frame == nullptr);
  HipdnnEpLoopBankStats quarantined = getStats(state);
  CHECK(quarantined.allocations == 2);
  CHECK(quarantined.active_bytes == 20);
  CHECK(quarantined.cached_bytes == 0);
  CHECK(quarantined.quarantined_bytes == 20);

  CHECK(hipStreamSynchronize(static_cast<hipStream_t>(state.stream)) ==
        hipSuccess);
  hipdnn_ep_loop_cleanup_quarantined_frames(&state);
  HipdnnEpLoopBankStats recycled = getStats(state);
  CHECK(recycled.active_bytes == 0);
  CHECK(recycled.cached_bytes == 20);
  CHECK(recycled.quarantined_bytes == 0);
  cleanup(state);
  HipdnnEpLoopBankStats cleaned{};
  hipdnn_ep_test_get_last_cleaned_loop_bank_stats(&cleaned);
  CHECK(cleaned.allocations == 2);
  CHECK(cleaned.frees == 2);
}

void testConvPoolInvalidShapeGuards() {
  RuntimeState state = makeState();
  int errorsBefore = recordedErrors.load();

  CHECK(wrap_miopenConvolutionForward(
            &state, /*shape_valid=*/0, /*input=*/nullptr, 1, 1, 1, 1,
            /*weights=*/nullptr, 1, /*bias=*/nullptr, /*output=*/nullptr, 0, 0,
            3, 3, 1, 1, 0, 0, 0, 0, 1, 1, 1, HIPDNN_EP_DATATYPE_FLOAT) == -1);
  CHECK(recordedErrors.load() == errorsBefore + 1);

  CHECK(wrap_pool(&state, /*input=*/nullptr, /*output=*/nullptr,
                  /*indices=*/nullptr, /*shape_valid=*/0,
                  HIPDNN_EP_DATATYPE_FLOAT, HIPDNN_EP_POOL_MAX,
                  /*spatial_rank=*/1, 1, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1,
                  0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 2) == -1);
  CHECK(recordedErrors.load() == errorsBefore + 2);
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) {
  ++recordedErrors;
  return 0;
}

int main() {
  testStateInitAllocationFailureIsAtomic();
  testZeroTripAliasesSeed();
  testZeroSizeDoesNotCheckoutBank();
  testSequentialGrowingFramesReuseTwoBlocks();
  testGrowShrinkRegrowAndMultipleCarriers();
  testZeroSizePublicationAndBankReuse();
  testFailureIsTransactional();
  testAllocationOverflow();
  testNestedAndConcurrentFrames();
  testPassThroughThenAllocateAndAtomicFailure();
  testEarlyExitAndSyncFailureCleanup();
  testInjectedAllocationFailureAfterEvolution();
  testGrowthSyncFailureQuarantinesBanks();
  testConvPoolInvalidShapeGuards();
  if (failures == 0) {
    std::printf("loop frame unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "loop frame unit test: %d FAILURE(S)\n", failures);
  return 1;
}
