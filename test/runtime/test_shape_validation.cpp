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

int checkExpand(RuntimeState &state, int64_t input, int64_t target,
                int64_t expected, int64_t prior, int64_t &extent,
                int64_t &elements, int64_t valid = 1) {
  state.error = false;
  extent = 123;
  elements = 123;
  return hipdnn_ep_checked_expand_extent(&state, &extent, &elements, valid,
                                         input, target, expected, prior);
}

} // namespace

int main() {
  RuntimeState state;
  int64_t extent = 0;
  int64_t elements = 0;

  int status = checkExpand(state, 5, 1, 5, 1, extent, elements);
  expect(status == 0 && !state.error && extent == 5 && elements == 5,
         "N broadcast with 1 must preserve N");

  status = checkExpand(state, 1, 5, 5, 1, extent, elements);
  expect(status == 0 && !state.error && extent == 5 && elements == 5,
         "1 broadcast with N must produce N");

  status = checkExpand(state, 5, 5, 5, 1, extent, elements);
  expect(status == 0 && !state.error && extent == 5 && elements == 5,
         "equal non-unit extents must be preserved");

  status = checkExpand(state, 0, 1, 0, 7, extent, elements);
  expect(status == 0 && !state.error && extent == 0 && elements == 0,
         "zero broadcast with 1 must produce zero");

  status = checkExpand(state, 1, 0, 0, 7, extent, elements);
  expect(status == 0 && !state.error && extent == 0 && elements == 0,
         "1 broadcast with zero must produce zero");

  status = checkExpand(state, 2, 5, -1, 1, extent, elements);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "incompatible input/target extents must fail safely");

  status = checkExpand(state, 5, 3, -1, 1, extent, elements);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "reverse incompatible extents must fail safely");

  status = checkExpand(state, 5, 1, 4, 1, extent, elements);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "static destination disagreement must fail safely");

  status = checkExpand(state, 2, 1, 2, std::numeric_limits<int64_t>::max(),
                       extent, elements);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "aggregate element-count overflow must fail safely");

  status = checkExpand(state, 1, 1, 1, std::numeric_limits<int64_t>::max(),
                       extent, elements);
  expect(status == 0 && !state.error && extent == 1 &&
             elements == std::numeric_limits<int64_t>::max(),
         "representable element-count boundary must succeed");

  status = checkExpand(state, 3, 1, 3, 1, extent, elements, /*valid=*/0);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "readback failure must not consume target slots");

  status = checkExpand(state, -1, 1, -1, 1, extent, elements);
  expect(status != 0 && state.error && extent == 0 && elements == 0,
         "negative extents must fail safely");

  if (failures != 0)
    return 1;
  std::puts("Shape validation tests passed");
  return 0;
}
