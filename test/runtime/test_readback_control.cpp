/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

struct RuntimeState {
  bool error = false;
};

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *state) {
  if (state)
    state->error = true;
  return 0;
}

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (condition)
    return;
  std::fprintf(stderr, "FAIL: %s\n", message);
  ++failures;
}

} // namespace

int main() {
  RuntimeState state;
  int16_t i16Value = -3;
  int32_t i32Values[] = {-7, 19};
  int64_t i64Value = std::numeric_limits<int64_t>::min();
  float f32Value = -1.25f;
  uint32_t f32Bits = 0;
  std::memcpy(&f32Bits, &f32Value, sizeof(f32Bits));
  const void *sources[] = {&i16Value, i32Values, &i64Value, &f32Value};
  int64_t counts[] = {1, 2, 1, 1};
  int64_t widths[] = {2, 4, 8, 4};
  int64_t output[] = {91, 92, 93, 94, 95};

  int status =
      hipdnn_ep_readback_control(&state, output, sources, counts, widths, 4, 5,
                                 /*require_non_negative=*/0);
  expect(status == 0 && !state.error, "valid descriptors must succeed");
  expect(output[0] == -3 && output[1] == -7 && output[2] == 19 &&
             output[3] == std::numeric_limits<int64_t>::min() &&
             static_cast<uint32_t>(output[4]) == f32Bits,
         "integer sign extension and floating bit preservation");

  // Source 0 is valid and would be copied first by the old implementation.
  // Source 1 is deliberately malformed late in the table. No partial values
  // may survive: failure leaves every declared output slot safely zero.
  state.error = false;
  for (int64_t &value : output)
    value = 81;
  int64_t malformedWidths[] = {2, 4, 8, 3};
  status = hipdnn_ep_readback_control(&state, output, sources, counts,
                                      malformedWidths, 4, 5,
                                      /*require_non_negative=*/0);
  expect(status != 0 && state.error,
         "late malformed source must set the shared error flag");
  expect(output[0] == 0 && output[1] == 0 && output[2] == 0 && output[3] == 0 &&
             output[4] == 0,
         "late malformed source must not expose a partial copy");

  int64_t shapeValues[] = {2, 3};
  const void *shapeSources[] = {shapeValues};
  int64_t shapeCounts[] = {2};
  int64_t shapeWidths[] = {8};
  int64_t shapeOutput[] = {71, 72};
  state.error = false;
  status = hipdnn_ep_readback_control(&state, shapeOutput, shapeSources,
                                      shapeCounts, shapeWidths, 1, 2,
                                      /*require_non_negative=*/1);
  expect(status == 0 && !state.error,
         "non-negative shape validation must accept valid extents");
  expect(shapeOutput[0] == 2 && shapeOutput[1] == 3,
         "valid shape extents must be preserved");

  shapeValues[1] = -3;
  shapeOutput[0] = 61;
  shapeOutput[1] = 62;
  state.error = false;
  status = hipdnn_ep_readback_control(&state, shapeOutput, shapeSources,
                                      shapeCounts, shapeWidths, 1, 2,
                                      /*require_non_negative=*/1);
  expect(status != 0 && state.error,
         "negative shape extent must set the shared error flag");
  expect(shapeOutput[0] == 0 && shapeOutput[1] == 0,
         "negative shape extent must clear the complete output group");

  if (failures != 0)
    return 1;
  std::puts("Readback control tests passed");
  return 0;
}
