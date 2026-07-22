/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device-side scatter reduction helpers shared by scatter_elements.
 * Mirrors the ScatterND reduction contract (none / add / mul / min / max).
 */

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

enum ScatterReduction {
  SCATTER_NONE = 0,
  SCATTER_ADD = 1,
  SCATTER_MUL = 2,
  SCATTER_MIN = 3,
  SCATTER_MAX = 4,
};

template <typename T> struct CasWord;
template <> struct CasWord<int32_t> {
  using type = uint32_t;
};
template <> struct CasWord<int64_t> {
  using type = unsigned long long;
};
template <> struct CasWord<float> {
  using type = uint32_t;
};

template <typename T>
__device__ inline typename CasWord<T>::type bit_cast_to_word(T v) {
  typename CasWord<T>::type w;
  __builtin_memcpy(&w, &v, sizeof(T));
  return w;
}

template <typename T>
__device__ inline T bit_cast_from_word(typename CasWord<T>::type w) {
  T v;
  __builtin_memcpy(&v, &w, sizeof(T));
  return v;
}

enum class ScatterOp { Mul, Min, Max };

template <typename T>
__device__ inline T scatter_apply_op(T cur, T upd, ScatterOp op) {
  switch (op) {
  case ScatterOp::Mul:
    return cur * upd;
  case ScatterOp::Min:
    return cur < upd ? cur : upd;
  case ScatterOp::Max:
    return cur > upd ? cur : upd;
  }
  return cur;
}

__device__ inline float scatter_apply_op(float cur, float upd, ScatterOp op) {
  switch (op) {
  case ScatterOp::Mul:
    return cur * upd;
  case ScatterOp::Min:
    if (isnan(cur) || isnan(upd))
      return __builtin_nanf("");
    return cur < upd ? cur : upd;
  case ScatterOp::Max:
    if (isnan(cur) || isnan(upd))
      return __builtin_nanf("");
    return cur > upd ? cur : upd;
  }
  return cur;
}

__device__ inline __half scatter_apply_op(__half cur, __half upd, ScatterOp op) {
  return __float2half(
      scatter_apply_op(__half2float(cur), __half2float(upd), op));
}

template <typename T>
__device__ inline void cas_scatter_apply(T *addr, T val, ScatterOp op) {
  using W = typename CasWord<T>::type;
  W *waddr = reinterpret_cast<W *>(addr);
  W old = *waddr, assumed;
  do {
    assumed = old;
    T cur = bit_cast_from_word<T>(assumed);
    T upd = scatter_apply_op<T>(cur, val, op);
    W desired = bit_cast_to_word<T>(upd);
    old = atomicCAS(waddr, assumed, desired);
  } while (assumed != old);
}

__device__ inline __half pack_load_half(uint32_t word, bool is_high) {
  uint16_t bits =
      is_high ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
              : static_cast<uint16_t>(word & 0xFFFFu);
  return __ushort_as_half(bits);
}

__device__ inline uint32_t pack_store_half(uint32_t word, __half v, bool is_high) {
  uint32_t bits = static_cast<uint32_t>(__half_as_ushort(v));
  if (is_high)
    return (word & 0x0000FFFFu) | (bits << 16);
  return (word & 0xFFFF0000u) | bits;
}

template <ScatterOp OP>
__device__ inline void atomic_apply_f16(__half *addr, __half val) {
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  bool is_high = (a & 0x3u) == 2u;
  uint32_t *base = reinterpret_cast<uint32_t *>(a & ~uintptr_t(0x3u));
  uint32_t old = *base, assumed;
  do {
    assumed = old;
    __half cur = pack_load_half(assumed, is_high);
    __half upd = scatter_apply_op(cur, val, OP);
    uint32_t desired = pack_store_half(assumed, upd, is_high);
    old = atomicCAS(base, assumed, desired);
  } while (assumed != old);
}

__device__ inline void atomic_add_f16(__half *addr, __half val) {
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  bool is_high = (a & 0x3u) == 2u;
  uint32_t *base = reinterpret_cast<uint32_t *>(a & ~uintptr_t(0x3u));
  uint32_t old = *base, assumed;
  do {
    assumed = old;
    __half cur = pack_load_half(assumed, is_high);
    __half upd = __float2half(__half2float(cur) + __half2float(val));
    uint32_t desired = pack_store_half(assumed, upd, is_high);
    old = atomicCAS(base, assumed, desired);
  } while (assumed != old);
}

__device__ inline void scatter_apply(int32_t *addr, int32_t val,
                                     int reduction_id) {
  switch (reduction_id) {
  case SCATTER_NONE:
    *addr = val;
    return;
  case SCATTER_ADD:
    atomicAdd(addr, val);
    return;
  case SCATTER_MUL:
    cas_scatter_apply<int32_t>(addr, val, ScatterOp::Mul);
    return;
  case SCATTER_MIN:
    atomicMin(addr, val);
    return;
  case SCATTER_MAX:
    atomicMax(addr, val);
    return;
  }
}

__device__ inline void scatter_apply(int64_t *addr, int64_t val,
                                     int reduction_id) {
  switch (reduction_id) {
  case SCATTER_NONE:
    *addr = val;
    return;
  case SCATTER_ADD:
    atomicAdd(reinterpret_cast<unsigned long long *>(addr),
              static_cast<unsigned long long>(val));
    return;
  case SCATTER_MUL:
    cas_scatter_apply<int64_t>(addr, val, ScatterOp::Mul);
    return;
  case SCATTER_MIN:
    cas_scatter_apply<int64_t>(addr, val, ScatterOp::Min);
    return;
  case SCATTER_MAX:
    cas_scatter_apply<int64_t>(addr, val, ScatterOp::Max);
    return;
  }
}

__device__ inline void scatter_apply(float *addr, float val, int reduction_id) {
  switch (reduction_id) {
  case SCATTER_NONE:
    *addr = val;
    return;
  case SCATTER_ADD:
    atomicAdd(addr, val);
    return;
  case SCATTER_MUL:
    cas_scatter_apply<float>(addr, val, ScatterOp::Mul);
    return;
  case SCATTER_MIN:
    cas_scatter_apply<float>(addr, val, ScatterOp::Min);
    return;
  case SCATTER_MAX:
    cas_scatter_apply<float>(addr, val, ScatterOp::Max);
    return;
  }
}

__device__ inline void scatter_apply(__half *addr, __half val, int reduction_id) {
  switch (reduction_id) {
  case SCATTER_NONE:
    *addr = val;
    return;
  case SCATTER_ADD:
    atomic_add_f16(addr, val);
    return;
  case SCATTER_MUL:
    atomic_apply_f16<ScatterOp::Mul>(addr, val);
    return;
  case SCATTER_MIN:
    atomic_apply_f16<ScatterOp::Min>(addr, val);
    return;
  case SCATTER_MAX:
    atomic_apply_f16<ScatterOp::Max>(addr, val);
    return;
  }
}
