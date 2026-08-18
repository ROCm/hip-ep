/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hipdnn_ep_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace {

int fail(RuntimeState *state, int64_t *output) {
  if (output)
    *output = 0;
  if (state)
    (void)hipdnn_ep_state_set_error_flag(state);
  return -1;
}

template <typename T>
int checkedIntegerRangeCount(T start, T limit, T delta, int64_t *count) {
  if (delta == 0)
    return -1;
  if ((delta > 0 && limit <= start) || (delta < 0 && limit >= start)) {
    *count = 0;
    return 0;
  }

  uint64_t difference;
  uint64_t step;
  int64_t start64 = static_cast<int64_t>(start);
  int64_t limit64 = static_cast<int64_t>(limit);
  int64_t delta64 = static_cast<int64_t>(delta);
  if (delta > 0) {
    difference =
        static_cast<uint64_t>(limit64) - static_cast<uint64_t>(start64);
    step = static_cast<uint64_t>(delta64);
  } else {
    difference =
        static_cast<uint64_t>(start64) - static_cast<uint64_t>(limit64);
    step = uint64_t{0} - static_cast<uint64_t>(delta64);
  }
  uint64_t result = difference / step + (difference % step != 0 ? 1 : 0);
  if (result > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return -1;
  *count = static_cast<int64_t>(result);
  return 0;
}

template <typename T>
int checkedFloatRangeCount(T start, T limit, T delta, int64_t *count) {
  if (!std::isfinite(start) || !std::isfinite(limit) || !std::isfinite(delta) ||
      delta == T{0})
    return -1;
  if ((delta > T{0} && limit <= start) || (delta < T{0} && limit >= start)) {
    *count = 0;
    return 0;
  }

  // Preserve the element-type subtraction/division used by the generated
  // Range path, but validate before converting the integral result to i64.
  T quotient = (limit - start) / delta;
  if (!std::isfinite(quotient) || quotient < T{0})
    return -1;
  long double rounded = std::ceil(static_cast<long double>(quotient));
  if (!std::isfinite(rounded) || rounded < 0 ||
      rounded > static_cast<long double>(std::numeric_limits<int64_t>::max()))
    return -1;
  *count = static_cast<int64_t>(rounded);
  return 0;
}

template <typename T> T decode(int64_t bits) {
  T value{};
  using Storage = std::conditional_t<
      sizeof(T) == 8, uint64_t,
      std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>>;
  Storage storage = static_cast<Storage>(bits);
  std::memcpy(&value, &storage, sizeof(value));
  return value;
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

int hipdnn_ep_checked_range_count(RuntimeState *state, int64_t *host_count,
                                  int64_t readback_valid, int64_t start_bits,
                                  int64_t limit_bits, int64_t delta_bits,
                                  int64_t data_type, int64_t expected_count) {
  if (!state || !host_count)
    return fail(state, host_count);
  *host_count = 0;
  if (!readback_valid)
    return fail(state, host_count);

  int status = -1;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_INT16:
    status = checkedIntegerRangeCount(
        static_cast<int16_t>(start_bits), static_cast<int16_t>(limit_bits),
        static_cast<int16_t>(delta_bits), host_count);
    break;
  case HIPDNN_EP_DATATYPE_INT32:
    status = checkedIntegerRangeCount(
        static_cast<int32_t>(start_bits), static_cast<int32_t>(limit_bits),
        static_cast<int32_t>(delta_bits), host_count);
    break;
  case HIPDNN_EP_DATATYPE_INT64:
    status = checkedIntegerRangeCount(start_bits, limit_bits, delta_bits,
                                      host_count);
    break;
  case HIPDNN_EP_DATATYPE_FLOAT:
    status = checkedFloatRangeCount(decode<float>(start_bits),
                                    decode<float>(limit_bits),
                                    decode<float>(delta_bits), host_count);
    break;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    status = checkedFloatRangeCount(decode<double>(start_bits),
                                    decode<double>(limit_bits),
                                    decode<double>(delta_bits), host_count);
    break;
  default:
    break;
  }
  if (status != 0 || (expected_count >= 0 && *host_count != expected_count))
    return fail(state, host_count);
  return 0;
}

int hipdnn_ep_checked_tile_extent(RuntimeState *state, int64_t *host_extent,
                                  int64_t *host_elements,
                                  int64_t readback_valid, int64_t input_extent,
                                  int64_t repeat, int64_t expected_extent,
                                  int64_t prior_elements) {
  auto failTile = [&]() {
    if (host_elements)
      *host_elements = 0;
    return fail(state, host_extent);
  };
  if (!state || !host_extent || !host_elements)
    return failTile();
  *host_extent = 0;
  *host_elements = 0;
  if (!readback_valid || input_extent < 0 || repeat < 0 || prior_elements < 0 ||
      (repeat != 0 &&
       input_extent > std::numeric_limits<int64_t>::max() / repeat))
    return failTile();

  int64_t result = input_extent * repeat;
  if ((expected_extent >= 0 && result != expected_extent) ||
      (result != 0 &&
       prior_elements > std::numeric_limits<int64_t>::max() / result))
    return failTile();
  *host_extent = result;
  *host_elements = prior_elements * result;
  return 0;
}
