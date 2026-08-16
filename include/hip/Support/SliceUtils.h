/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_SUPPORT_SLICE_UTILS_H
#define HIP_SUPPORT_SLICE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace hipdnn_ep {
namespace slice {

/// Compute ceil(distance / divisor) without an overflow-prone addition.
inline bool checkedCeilDiv(uint64_t distance, uint64_t divisor,
                           int64_t &result) {
  if (divisor == 0)
    return false;
  uint64_t quotient = distance / divisor;
  if (distance % divisor != 0)
    ++quotient;
  if (quotient > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  result = static_cast<int64_t>(quotient);
  return true;
}

/// Compute the Slice extent from already-normalized and clamped bounds.
///
/// Unsigned subtraction is intentional: C++ defines conversion from signed to
/// unsigned modulo 2^64, so it yields the exact nonnegative distance even for
/// the extreme negative-step interval `[INT64_MAX - 1, -1)`. Computing the
/// step magnitude the same way handles `INT64_MIN` without signed negation.
inline bool computeNormalizedExtent(int64_t start, int64_t end, int64_t step,
                                    int64_t &extent) {
  if (step == 0)
    return false;
  if ((step > 0 && end <= start) || (step < 0 && start <= end)) {
    extent = 0;
    return true;
  }

  uint64_t distance =
      step > 0 ? static_cast<uint64_t>(end) - static_cast<uint64_t>(start)
               : static_cast<uint64_t>(start) - static_cast<uint64_t>(end);
  uint64_t divisor = step > 0 ? static_cast<uint64_t>(step)
                              : uint64_t{0} - static_cast<uint64_t>(step);
  return checkedCeilDiv(distance, divisor, extent);
}

/// Normalize one Slice axis according to ONNX. The returned start is always
/// safe to use when `extent > 0`; zero-sized dimensions use start=0. Signed
/// arithmetic never negates `INT64_MIN`.
inline bool normalizeAxis(int64_t dim, int64_t rawStart, int64_t rawEnd,
                          int64_t step, int64_t &start, int64_t &extent) {
  if (dim < 0 || step == 0)
    return false;
  if (dim == 0) {
    start = 0;
    extent = 0;
    return true;
  }

  auto addDimIfNegative = [dim](int64_t value) {
    return value < 0 ? value + dim : value;
  };
  int64_t end = addDimIfNegative(rawEnd);
  start = addDimIfNegative(rawStart);
  if (step > 0) {
    if (start < 0)
      start = 0;
    else if (start > dim)
      start = dim;
    if (end < 0)
      end = 0;
    else if (end > dim)
      end = dim;
  } else {
    int64_t upper = dim - 1;
    if (start < 0)
      start = 0;
    else if (start > upper)
      start = upper;
    if (end < -1)
      end = -1;
    else if (end > upper)
      end = upper;
  }
  return computeNormalizedExtent(start, end, step, extent);
}

struct NormalizedParameters {
  std::vector<int64_t> starts;
  std::vector<int64_t> steps;
  std::vector<int64_t> extents;
};

inline bool normalizeAxes(int64_t rank, const int64_t *axes, int64_t count,
                          std::vector<int64_t> &out) {
  if (rank < 0 || count < 0 || (count > 0 && !axes))
    return false;
  std::vector<int64_t> normalized;
  normalized.reserve(static_cast<size_t>(count));
  std::vector<bool> seen(static_cast<size_t>(rank), false);
  for (int64_t i = 0; i < count; ++i) {
    int64_t axis = axes[i];
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank || seen[static_cast<size_t>(axis)])
      return false;
    seen[static_cast<size_t>(axis)] = true;
    normalized.push_back(axis);
  }
  out = std::move(normalized);
  return true;
}

/// Pure, allocation-only-on-success Slice semantic rule for fully static
/// shapes and parameters. Omitted axes resolve to `[0, ..., count)` and omitted
/// steps to ones. Output arrays have exactly `rank` entries.
inline bool normalizeParameters(const int64_t *dataShape, int64_t rank,
                                const int64_t *starts, const int64_t *ends,
                                const int64_t *axes, const int64_t *steps,
                                int64_t count, NormalizedParameters &out) {
  if (rank < 0 || count < 0 || (rank > 0 && !dataShape) ||
      (count > 0 && (!starts || !ends)))
    return false;

  NormalizedParameters result;
  result.starts.assign(static_cast<size_t>(rank), 0);
  result.steps.assign(static_cast<size_t>(rank), 1);
  result.extents.resize(static_cast<size_t>(rank));
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (dataShape[axis] < 0)
      return false;
    result.extents[static_cast<size_t>(axis)] = dataShape[axis];
  }

  std::vector<int64_t> defaultAxes;
  if (!axes) {
    defaultAxes.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i)
      defaultAxes.push_back(i);
    axes = defaultAxes.data();
  }
  std::vector<int64_t> normalizedAxes;
  if (!normalizeAxes(rank, axes, count, normalizedAxes))
    return false;
  for (int64_t i = 0; i < count; ++i) {
    int64_t axis = normalizedAxes[static_cast<size_t>(i)];
    int64_t step = steps ? steps[i] : 1;
    int64_t start = 0;
    int64_t extent = 0;
    if (!normalizeAxis(dataShape[axis], starts[i], ends[i], step, start,
                       extent))
      return false;
    result.starts[static_cast<size_t>(axis)] = start;
    result.steps[static_cast<size_t>(axis)] = step;
    result.extents[static_cast<size_t>(axis)] = extent;
  }
  out = std::move(result);
  return true;
}

enum class PreflightStatus { Error, EmptyOutput, Proceed };

struct PreflightResult {
  PreflightStatus status = PreflightStatus::Error;
  uint64_t dataElements = 0;
  uint64_t outputElements = 0;
};

inline bool checkedElementCount(const int64_t *shape, int64_t rank,
                                uint64_t &elements) {
  elements = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] < 0)
      return false;
    uint64_t dim = static_cast<uint64_t>(shape[i]);
    if (dim != 0 && elements > std::numeric_limits<uint64_t>::max() / dim)
      return false;
    elements *= dim;
  }
  return true;
}

/// Validate the exact normalized Slice address contract before touching device
/// data. Empty outputs accept null allocator storage. No fixed rank cap exists.
inline PreflightResult preflight(const void *state, const void *data,
                                 const void *output, const int64_t *dataShape,
                                 const int64_t *outputShape,
                                 const int64_t *starts, const int64_t *steps,
                                 const int64_t *extents, int64_t rank,
                                 int64_t elementBytes, bool paramsValid) {
  PreflightResult result;
  if (!state || rank < 0 || elementBytes <= 0 ||
      (rank > 0 &&
       (!dataShape || !outputShape || !starts || !steps || !extents)))
    return result;
  if (!paramsValid)
    return result;
  if (!checkedElementCount(dataShape, rank, result.dataElements) ||
      !checkedElementCount(outputShape, rank, result.outputElements))
    return result;
  if (result.dataElements >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      result.outputElements >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return result;
  size_t elementSize = static_cast<size_t>(elementBytes);
  if (result.dataElements > std::numeric_limits<size_t>::max() / elementSize ||
      result.outputElements > std::numeric_limits<size_t>::max() / elementSize)
    return result;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (extents[axis] != outputShape[axis] || steps[axis] == 0)
      return result;
    if (extents[axis] == 0)
      continue;
    if (starts[axis] < 0 || starts[axis] >= dataShape[axis])
      return result;

    uint64_t advances = static_cast<uint64_t>(extents[axis] - 1);
    if (steps[axis] > 0) {
      uint64_t available =
          static_cast<uint64_t>(dataShape[axis] - 1 - starts[axis]);
      if (advances > available / static_cast<uint64_t>(steps[axis]))
        return result;
    } else {
      uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(steps[axis]);
      if (advances > static_cast<uint64_t>(starts[axis]) / magnitude)
        return result;
    }
  }
  if (result.outputElements == 0) {
    result.status = PreflightStatus::EmptyOutput;
    return result;
  }
  if (!output || (result.dataElements != 0 && !data))
    return result;
  result.status = PreflightStatus::Proceed;
  return result;
}

} // namespace slice
} // namespace hipdnn_ep

#endif // HIP_SUPPORT_SLICE_UTILS_H
