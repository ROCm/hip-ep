/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Support/SliceUtils.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (condition)
    return;
  std::fprintf(stderr, "FAIL: %s\n", message);
  ++failures;
}

void expectExtent(int64_t start, int64_t end, int64_t step, int64_t expected,
                  const char *message) {
  int64_t actual = -1;
  bool valid =
      hipdnn_ep::slice::computeNormalizedExtent(start, end, step, actual);
  expect(valid && actual == expected, message);
}

} // namespace

int main() {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();

  expectExtent(0, kMax, 1, kMax, "positive unit-step extreme");
  expectExtent(kMax - 1, -1, -1, kMax, "negative unit-step extreme");
  expectExtent(kMax - 1, -1, kMin, 1, "negative INT64_MIN step");
  int64_t ignored = 0;
  expect(!hipdnn_ep::slice::computeNormalizedExtent(0, 1, 0, ignored),
         "zero step must fail");

  int64_t start = -1;
  int64_t extent = -1;
  expect(hipdnn_ep::slice::normalizeAxis(10, kMin, kMin, kMin, start, extent) &&
             start == 0 && extent == 1,
         "INT64_MIN indices normalize without overflow");
  expect(hipdnn_ep::slice::normalizeAxis(10, -1, kMin, -1, start, extent) &&
             start == 9 && extent == 10,
         "negative reverse bounds normalize exactly");
  expect(hipdnn_ep::slice::normalizeAxis(0, kMin, kMax, kMin, start, extent) &&
             start == 0 && extent == 0,
         "zero dimension is exact and address-safe");

  int64_t shape12[12];
  for (int64_t &dim : shape12)
    dim = 2;
  int64_t starts[] = {0, 1};
  int64_t ends[] = {2, 2};
  hipdnn_ep::slice::NormalizedParameters normalized;
  expect(hipdnn_ep::slice::normalizeParameters(
             shape12, 12, starts, ends, /*axes=*/nullptr, /*steps=*/nullptr, 2,
             normalized) &&
             normalized.extents[0] == 2 && normalized.extents[1] == 1 &&
             normalized.extents[11] == 2,
         "omitted axes/steps use count, not rank");

  int64_t axis11[] = {11};
  expect(hipdnn_ep::slice::normalizeParameters(
             shape12, 12, starts, ends, axis11, nullptr, 1, normalized) &&
             normalized.starts.size() == 12 && normalized.extents[11] == 2,
         "rank greater than eight is supported");
  int64_t duplicateAxes[] = {3, -9};
  expect(!hipdnn_ep::slice::normalizeParameters(
             shape12, 12, starts, ends, duplicateAxes, nullptr, 2, normalized),
         "duplicate normalized axes reject");
  int64_t zeroSteps[] = {0};
  expect(!hipdnn_ep::slice::normalizeParameters(
             shape12, 12, starts, ends, axis11, zeroSteps, 1, normalized),
         "zero step rejects");

  int token = 0;
  int64_t dataShape[] = {2, 3};
  int64_t emptyShape[] = {2, 0};
  int64_t exactStarts[] = {0, 0};
  int64_t exactSteps[] = {1, 1};
  auto empty = hipdnn_ep::slice::preflight(
      &token, /*data=*/nullptr, /*output=*/nullptr, dataShape, emptyShape,
      exactStarts, exactSteps, emptyShape, 2, sizeof(float), true);
  expect(empty.status == hipdnn_ep::slice::PreflightStatus::EmptyOutput,
         "empty exact output permits null storage");

  int64_t nonemptyShape[] = {2, 3};
  auto proceed = hipdnn_ep::slice::preflight(
      &token, &token, &token, dataShape, nonemptyShape, exactStarts, exactSteps,
      nonemptyShape, 2, sizeof(float), true);
  expect(proceed.status == hipdnn_ep::slice::PreflightStatus::Proceed,
         "valid normalized address contract proceeds");

  int64_t badStart[] = {0, 3};
  auto badAddress = hipdnn_ep::slice::preflight(
      &token, &token, &token, dataShape, nonemptyShape, badStart, exactSteps,
      nonemptyShape, 2, sizeof(float), true);
  expect(badAddress.status == hipdnn_ep::slice::PreflightStatus::Error,
         "last-address overflow rejects defensively");

  int64_t byteOverflowExtent = static_cast<int64_t>(
      std::numeric_limits<size_t>::max() / sizeof(int64_t) + 1);
  int64_t hugeDataShape[] = {byteOverflowExtent};
  int64_t zeroShape[] = {0};
  int64_t oneStart[] = {0};
  int64_t oneStep[] = {1};
  auto inputBytesOverflow = hipdnn_ep::slice::preflight(
      &token, nullptr, nullptr, hugeDataShape, zeroShape, oneStart, oneStep,
      zeroShape, 1, sizeof(int64_t), true);
  expect(inputBytesOverflow.status == hipdnn_ep::slice::PreflightStatus::Error,
         "input element count times element bytes must fit size_t");

  int64_t unitDataShape[] = {1};
  auto outputBytesOverflow = hipdnn_ep::slice::preflight(
      &token, &token, &token, unitDataShape, hugeDataShape, oneStart, oneStep,
      hugeDataShape, 1, sizeof(int64_t), true);
  expect(outputBytesOverflow.status == hipdnn_ep::slice::PreflightStatus::Error,
         "output element count times element bytes must fit size_t");

  auto invalidParams = hipdnn_ep::slice::preflight(
      &token, nullptr, nullptr, dataShape, emptyShape, exactStarts, exactSteps,
      emptyShape, 2, sizeof(float), false);
  expect(invalidParams.status == hipdnn_ep::slice::PreflightStatus::Error,
         "params_valid=false records an error instead of launching");

  if (failures != 0)
    return 1;
  std::puts("Slice utility tests passed");
  return 0;
}
