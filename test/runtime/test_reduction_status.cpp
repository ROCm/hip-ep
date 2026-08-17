/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

using ReductionWrapper = int (*)(RuntimeState *, void *, void *, void *,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 int64_t, int64_t);

struct ReductionCase {
  ReductionWrapper wrapper;
  std::array<int64_t, 4> supportedTypes;
  size_t supportedTypeCount;
  int64_t rejectedType;
};

void checkStatusObserved(RuntimeState *state, int status) {
  CHECK(status != 0);
  CHECK(hipdnn_ep_state_record_status(state, status) == status);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(state) == -1);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(state) == 0);
}

void testReductionStatusAggregation() {
  int deviceError = 0;
  RuntimeState state{};
  state.stream = reinterpret_cast<hipStream_t>(1);
  state.device_error_flag = &deviceError;
  int data = 1;
  int output = 0;

  const std::array<ReductionCase, 6> cases{{
      {wrap_reduce_sum,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_FLOAT,
        HIPDNN_EP_DATATYPE_INT32, HIPDNN_EP_DATATYPE_INT64},
       4,
       HIPDNN_EP_DATATYPE_BFLOAT16},
      {wrap_reduce_mean,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_FLOAT, 0, 0},
       2,
       HIPDNN_EP_DATATYPE_INT32},
      {wrap_reduce_l2,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_FLOAT, 0, 0},
       2,
       HIPDNN_EP_DATATYPE_INT64},
      {wrap_reduce_max,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_INT32,
        HIPDNN_EP_DATATYPE_INT64, 0},
       3,
       HIPDNN_EP_DATATYPE_FLOAT},
      {wrap_reduce_min,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_INT32,
        HIPDNN_EP_DATATYPE_INT64, 0},
       3,
       HIPDNN_EP_DATATYPE_FLOAT},
      {wrap_reduce_prod,
       {HIPDNN_EP_DATATYPE_HALF, HIPDNN_EP_DATATYPE_INT32,
        HIPDNN_EP_DATATYPE_INT64, 0},
       3,
       HIPDNN_EP_DATATYPE_FLOAT},
  }};

  CHECK(hipdnn_ep_state_reset_error_flag(&state) == 0);
  for (const ReductionCase &test : cases) {
    for (size_t i = 0; i < test.supportedTypeCount; ++i) {
      int status = test.wrapper(&state, &data, nullptr, &output, 1, 1, 1,
                                test.supportedTypes[i], 1, 0, 1);
      CHECK(status == 0);
      CHECK(hipdnn_ep_state_record_status(&state, status) == 0);
      CHECK(hipdnn_ep_state_read_and_clear_error_flag(&state) == 0);
    }

    checkStatusObserved(&state, test.wrapper(&state, &data, nullptr, &output, 1,
                                             1, 1, test.rejectedType, 1, 0, 1));
    checkStatusObserved(&state,
                        test.wrapper(&state, nullptr, nullptr, &output, 1, 1, 1,
                                     test.supportedTypes[0], 1, 0, 1));
    checkStatusObserved(&state,
                        test.wrapper(&state, &data, nullptr, &output, 1, 1, 1,
                                     test.supportedTypes[0], 2, 0, 1));
  }

  CHECK(hipdnn_ep_state_record_status(nullptr, -1) == -1);
}

} // namespace

int main() {
  testReductionStatusAggregation();
  if (failures == 0) {
    std::printf("Reduction status unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "Reduction status unit test: %d FAILURE(S)\n", failures);
  return 1;
}
