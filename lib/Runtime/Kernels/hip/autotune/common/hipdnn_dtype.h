/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_COMMON_DTYPE_H
#define HIPDNN_EP_COMMON_DTYPE_H

#include <cstdint>

/* C++ mirror of hipdnn_dtype.fbs's HipdnnDType. Kept in lock-step by hand (it
 * is a tiny, append-only enum) so a kernel .hip file that is not flatbuffers-
 * aware can still name a dtype without pulling in the generated fbs header.
 * Values must match the .fbs exactly -- see that file for the append-only
 * rule. */

namespace hipdnn_ep {
namespace common {

enum class HipdnnDType : uint8_t {
  Any = 0,
  F32 = 1,
  F16 = 2,
  BF16 = 3,
  F64 = 4,
  I32 = 5,
  I8 = 6,
  U8 = 7,
  I4 = 8,
  U4 = 9,
  U3 = 10,
  U2 = 11,
  F8E4M3 = 12,
  F8E5M2 = 13,
};

inline const char *toString(HipdnnDType d) {
  switch (d) {
  case HipdnnDType::F32: return "f32";
  case HipdnnDType::F16: return "f16";
  case HipdnnDType::BF16: return "bf16";
  case HipdnnDType::F64: return "f64";
  case HipdnnDType::I32: return "i32";
  case HipdnnDType::I8: return "i8";
  case HipdnnDType::U8: return "u8";
  case HipdnnDType::I4: return "i4";
  case HipdnnDType::U4: return "u4";
  case HipdnnDType::U3: return "u3";
  case HipdnnDType::U2: return "u2";
  case HipdnnDType::F8E4M3: return "f8e4m3";
  case HipdnnDType::F8E5M2: return "f8e5m2";
  default: return "any";
  }
}

} // namespace common
} // namespace hipdnn_ep

#endif // HIPDNN_EP_COMMON_DTYPE_H
