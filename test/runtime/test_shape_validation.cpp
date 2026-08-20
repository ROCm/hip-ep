/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

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

template <typename T> int64_t bits(T value) {
  using Storage = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
  Storage storage = 0;
  std::memcpy(&storage, &value, sizeof(value));
  return static_cast<int64_t>(storage);
}

int checkRange(RuntimeState &state, int64_t start, int64_t limit, int64_t delta,
               int64_t type, int64_t &count, int64_t valid = 1,
               int64_t expected = -1) {
  state.error = false;
  count = 123;
  return hipdnn_ep_checked_range_count(&state, &count, valid, start, limit,
                                       delta, type, expected);
}

int checkTile(RuntimeState &state, int64_t input, int64_t repeat,
              int64_t expected, int64_t &extent, int64_t valid = 1,
              int64_t prior = 1, int64_t *elementsOut = nullptr) {
  state.error = false;
  extent = 123;
  int64_t elements = 123;
  int status = hipdnn_ep_checked_tile_extent(&state, &extent, &elements, valid,
                                             input, repeat, expected, prior);
  if (elementsOut)
    *elementsOut = elements;
  return status;
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
  int64_t value = 0;

  int status = checkRange(state, 0, 4, 0, HIPDNN_EP_DATATYPE_INT64, value);
  expect(status != 0 && state.error && value == 0,
         "Range zero delta must fail safely");

  status = checkRange(state, 0, std::numeric_limits<int64_t>::max(), 1,
                      HIPDNN_EP_DATATYPE_INT64, value);
  expect(status == 0 && !state.error &&
             value == std::numeric_limits<int64_t>::max(),
         "Range INT64_MAX count boundary must succeed");

  status = checkRange(state, std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::max(), 1,
                      HIPDNN_EP_DATATYPE_INT64, value);
  expect(status != 0 && state.error && value == 0,
         "Range unrepresentable endpoint difference must fail");

  status = checkRange(state, 0, std::numeric_limits<int64_t>::min(),
                      std::numeric_limits<int64_t>::min(),
                      HIPDNN_EP_DATATYPE_INT64, value);
  expect(status == 0 && !state.error && value == 1,
         "Range INT64_MIN delta must not overflow during magnitude");

  status = checkRange(state, bits(0.0f),
                      bits(std::numeric_limits<float>::quiet_NaN()), bits(1.0f),
                      HIPDNN_EP_DATATYPE_FLOAT, value);
  expect(status != 0 && state.error && value == 0,
         "Range NaN must fail safely");

  status = checkRange(state, bits(0.0),
                      bits(std::numeric_limits<double>::infinity()), bits(1.0),
                      HIPDNN_EP_DATATYPE_DOUBLE, value);
  expect(status != 0 && state.error && value == 0,
         "Range infinity must fail safely");

  status = checkRange(state, bits(-1.0f), bits(1.0f), bits(0.5f),
                      HIPDNN_EP_DATATYPE_FLOAT, value);
  expect(status == 0 && !state.error && value == 4,
         "valid floating Range count");

  status = checkRange(state, 0, 4, 1, HIPDNN_EP_DATATYPE_INT64, value,
                      /*valid=*/1, /*expected=*/5);
  expect(status != 0 && state.error && value == 0,
         "Range descriptor mismatch must fail");

  status = checkRange(state, 0, 4, 1, HIPDNN_EP_DATATYPE_INT64, value,
                      /*valid=*/0);
  expect(status != 0 && state.error && value == 0,
         "failed readback must not expose host slots");

  status = checkTile(state, 7, -1, -1, value);
  expect(status != 0 && state.error && value == 0,
         "negative Tile repeat must fail before allocation");

  status = checkTile(state, 7, 0, 0, value);
  expect(status == 0 && !state.error && value == 0,
         "zero Tile repeat must produce a zero extent");

  status = checkTile(state, std::numeric_limits<int64_t>::max(), 2, -1, value);
  expect(status != 0 && state.error && value == 0,
         "Tile extent multiplication overflow must fail");

  status = checkTile(state, 3, 2, 7, value);
  expect(status != 0 && state.error && value == 0,
         "Tile descriptor mismatch must fail");

  status = checkTile(state, std::numeric_limits<int64_t>::max(), 1,
                     std::numeric_limits<int64_t>::max(), value);
  expect(status == 0 && !state.error &&
             value == std::numeric_limits<int64_t>::max(),
         "Tile representable boundary must succeed");

  int64_t elements = 0;
  status = checkTile(state, 2, 1, 2, value, /*valid=*/1,
                     /*prior=*/std::numeric_limits<int64_t>::max(), &elements);
  expect(status != 0 && state.error && value == 0 && elements == 0,
         "Tile aggregate element-count overflow must fail");

  status = checkTile(state, 3, 2, 6, value, /*valid=*/0);
  expect(status != 0 && state.error && value == 0,
         "Tile readback failure must not consume repeat slots");

  int64_t expandExtent = 0;
  int64_t expandElements = 0;
  status = checkExpand(state, 5, 1, 5, 1, expandExtent, expandElements);
  expect(status == 0 && !state.error && expandExtent == 5 &&
             expandElements == 5,
         "Expand N/1 must preserve N");

  status = checkExpand(state, 1, 5, 5, 1, expandExtent, expandElements);
  expect(status == 0 && !state.error && expandExtent == 5 &&
             expandElements == 5,
         "Expand 1/N must produce N");

  status = checkExpand(state, 0, 1, 0, 7, expandExtent, expandElements);
  expect(status == 0 && !state.error && expandExtent == 0 &&
             expandElements == 0,
         "Expand 0/1 must produce zero");

  status = checkExpand(state, 2, 5, -1, 1, expandExtent, expandElements);
  expect(status != 0 && state.error && expandExtent == 0 && expandElements == 0,
         "incompatible Expand extents must fail safely");

  status = checkExpand(state, 2, 1, 2, std::numeric_limits<int64_t>::max(),
                       expandExtent, expandElements);
  expect(status != 0 && state.error && expandExtent == 0 && expandElements == 0,
         "Expand aggregate element-count overflow must fail safely");

  status = checkExpand(state, 3, 1, 3, 1, expandExtent, expandElements,
                       /*valid=*/0);
  expect(status != 0 && state.error && expandExtent == 0 && expandElements == 0,
         "Expand readback failure must not consume target slots");
  if (failures != 0)
    return 1;
  std::puts("Shape validation tests passed");
  return 0;
}
