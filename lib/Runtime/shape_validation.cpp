/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"

#include <limits>

namespace {

int fail(RuntimeState *state, int64_t *output) {
  if (output)
    *output = 0;
  if (state)
    (void)hipdnn_ep_state_set_error_flag(state);
  return -1;
}

} // namespace

int hipdnn_ep_checked_expand_extent(RuntimeState *state, int64_t *host_extent,
                                    int64_t *host_elements, int64_t prior_valid,
                                    int64_t input_extent, int64_t target_extent,
                                    int64_t expected_extent,
                                    int64_t prior_elements) {
  auto failExpand = [&]() {
    if (host_elements)
      *host_elements = 0;
    return fail(state, host_extent);
  };
  if (!state || !host_extent || !host_elements)
    return failExpand();
  *host_extent = 0;
  *host_elements = 0;
  if (!prior_valid || input_extent < 0 || target_extent < 0 ||
      prior_elements < 0)
    return failExpand();

  bool compatible =
      input_extent == target_extent || input_extent == 1 || target_extent == 1;
  if (!compatible)
    return failExpand();
  int64_t result = input_extent == 1 ? target_extent : input_extent;
  if ((expected_extent >= 0 && result != expected_extent) ||
      (result != 0 &&
       prior_elements > std::numeric_limits<int64_t>::max() / result))
    return failExpand();

  *host_extent = result;
  *host_elements = prior_elements * result;
  return 0;
}
