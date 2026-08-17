/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "gather_block_quantized_utils.h"

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

void testDefaultZeroPoints() {
  using hipdnn_ep::gbq::defaultZeroPoint;
  CHECK(defaultZeroPoint(4, /*unsignedStorage=*/false) == 0);
  CHECK(defaultZeroPoint(4, /*unsignedStorage=*/true) == 8);
  CHECK(defaultZeroPoint(8, /*unsignedStorage=*/false) == 0);
  CHECK(defaultZeroPoint(8, /*unsignedStorage=*/true) == 128);
}

void testEightBitInterpretation() {
  using hipdnn_ep::gbq::decodeStorageByte;
  for (int byte = 0; byte <= 255; ++byte) {
    CHECK(decodeStorageByte(static_cast<uint8_t>(byte), 8,
                            /*signedStorage=*/false,
                            /*logicalIndex=*/0) == byte);
    int signedExpected = byte < 128 ? byte : byte - 256;
    CHECK(decodeStorageByte(static_cast<uint8_t>(byte), 8,
                            /*signedStorage=*/true,
                            /*logicalIndex=*/0) == signedExpected);
  }
}

void testPackedFourBitInterpretation() {
  using hipdnn_ep::gbq::decodeStorageByte;
  for (int low = 0; low < 16; ++low) {
    for (int high = 0; high < 16; ++high) {
      uint8_t byte = static_cast<uint8_t>(low | (high << 4));
      CHECK(decodeStorageByte(byte, 4, /*signedStorage=*/false, 0) == low);
      CHECK(decodeStorageByte(byte, 4, /*signedStorage=*/false, 1) == high);
      CHECK(decodeStorageByte(byte, 4, /*signedStorage=*/true, 0) ==
            (low < 8 ? low : low - 16));
      CHECK(decodeStorageByte(byte, 4, /*signedStorage=*/true, 1) ==
            (high < 8 ? high : high - 16));
    }
  }
}

void testAxisPolicy() {
  using hipdnn_ep::gbq::supportsUint8Axes;
  CHECK(supportsUint8Axes(8, /*unsignedStorage=*/true, 2, 0, 1));
  CHECK(!supportsUint8Axes(8, /*unsignedStorage=*/true, 2, 1, 0));
  CHECK(!supportsUint8Axes(8, /*unsignedStorage=*/true, 3, 0, 1));
  CHECK(supportsUint8Axes(8, /*unsignedStorage=*/false, 3, 1, 0));
  CHECK(supportsUint8Axes(4, /*unsignedStorage=*/true, 3, 1, 0));
}

} // namespace

int main() {
  testDefaultZeroPoints();
  testEightBitInterpretation();
  testPackedFourBitInterpretation();
  testAxisPolicy();
  if (failures != 0) {
    std::fprintf(stderr, "%d GatherBlockQuantized checks failed\n", failures);
    return 1;
  }
  std::printf("GatherBlockQuantized utility checks passed\n");
  return 0;
}
