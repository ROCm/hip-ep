/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>
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
  int32_t i32Values[] = {-7, 19};
  int64_t i64Value = std::numeric_limits<int64_t>::min();
  const void *sources[] = {i32Values, &i64Value};
  int64_t counts[] = {2, 1};
  int64_t widths[] = {4, 8};
  int64_t output[] = {91, 92, 93};

  int status =
      hipdnn_ep_readback_control(&state, output, sources, counts, widths, 2, 3);
  expect(status == 0 && !state.error, "valid descriptors must succeed");
  expect(output[0] == -7 && output[1] == 19 &&
             output[2] == std::numeric_limits<int64_t>::min(),
         "i32 sign extension and signed i64 preservation");

  // Source 0 is valid and would be copied first by the old implementation.
  // Source 1 is deliberately malformed late in the table. No partial values
  // may survive: failure leaves every declared output slot safely zero.
  state.error = false;
  output[0] = 81;
  output[1] = 82;
  output[2] = 83;
  int64_t malformedWidths[] = {4, 2};
  status = hipdnn_ep_readback_control(&state, output, sources, counts,
                                      malformedWidths, 2, 3);
  expect(status != 0 && state.error,
         "late malformed source must set the shared error flag");
  expect(output[0] == 0 && output[1] == 0 && output[2] == 0,
         "late malformed source must not expose a partial copy");

  if (failures != 0)
    return 1;
  std::puts("Readback control tests passed");
  return 0;
}
