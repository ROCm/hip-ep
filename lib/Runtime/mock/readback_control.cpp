/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"

#include <cstring>
#include <limits>

int hipdnn_ep_readback_control(RuntimeState *state, int64_t *host_out,
                               const void *const *device_sources,
                               const int64_t *element_counts,
                               const int64_t *element_bytes,
                               int64_t source_count, int64_t total_count,
                               int64_t require_non_negative) {
  auto setError = [&]() {
    if (state)
      (void)hipdnn_ep_state_set_error_flag(state);
    return -1;
  };
  bool outputRangeValid =
      host_out && total_count >= 0 &&
      static_cast<uint64_t>(total_count) <=
          std::numeric_limits<size_t>::max() / sizeof(int64_t);
  auto zeroOutput = [&]() {
    if (!outputRangeValid)
      return;
    for (int64_t i = 0; i < total_count; ++i)
      host_out[i] = 0;
  };

  if (!state || !outputRangeValid || !device_sources || !element_counts ||
      !element_bytes || source_count <= 0 ||
      static_cast<uint64_t>(source_count) >
          std::numeric_limits<size_t>::max() / sizeof(void *) ||
      (require_non_negative != 0 && require_non_negative != 1)) {
    zeroOutput();
    return setError();
  }

  // Validate the complete descriptor table before copying source 0. A malformed
  // late source must not leave an earlier source visible in host_out.
  int64_t checkedTotal = 0;
  bool descriptorsValid = true;
  for (int64_t i = 0; i < source_count; ++i) {
    int64_t count = element_counts[i];
    int64_t width = element_bytes[i];
    if (count < 0 || (width != 4 && width != 8) ||
        (count > 0 && !device_sources[i]) ||
        count > std::numeric_limits<int64_t>::max() - checkedTotal ||
        static_cast<uint64_t>(count) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(width)) {
      descriptorsValid = false;
      break;
    }
    checkedTotal += count;
  }
  if (!descriptorsValid || checkedTotal != total_count) {
    zeroOutput();
    return setError();
  }

  zeroOutput();
  int64_t outputIndex = 0;
  for (int64_t i = 0; i < source_count; ++i) {
    const unsigned char *source =
        static_cast<const unsigned char *>(device_sources[i]);
    for (int64_t j = 0; j < element_counts[i]; ++j) {
      if (element_bytes[i] == 4) {
        int32_t value = 0;
        std::memcpy(&value, source + static_cast<size_t>(j) * sizeof(value),
                    sizeof(value));
        host_out[outputIndex++] = static_cast<int64_t>(value);
      } else {
        int64_t value = 0;
        std::memcpy(&value, source + static_cast<size_t>(j) * sizeof(value),
                    sizeof(value));
        host_out[outputIndex++] = value;
      }
    }
  }
  if (require_non_negative != 0) {
    bool invalid = false;
    for (int64_t i = 0; i < total_count; ++i)
      invalid |= host_out[i] < 0;
    if (invalid) {
      zeroOutput();
      return setError();
    }
  }
  return 0;
}
