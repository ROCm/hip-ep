/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define EIGEN_USE_THREADS
#include "morphizen/transpose.hpp"
#include "cleanup.hpp"
#include "morphizen/env_config.hpp"
#include <glog/logging.h>
#ifndef _WIN32
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
#ifndef _WIN32
#  pragma GCC diagnostic pop
#endif
DEF_ENV_PARAM(NUM_OF_ENGINE_THREAD, "4")
namespace {

struct EigenThreadPool {
public:
  EigenThreadPool(int num_threads)
      : pool_(std::make_unique<Eigen::ThreadPool>(num_threads)),
        device_(std::make_unique<Eigen::ThreadPoolDevice>(pool_.get(),
                                                          num_threads)) {}
  void Clear() {
    device_.reset();
    pool_.reset();
  }
  ~EigenThreadPool() { Clear(); }

  Eigen::ThreadPoolDevice& get_device() { return *device_; }

private:
  std::unique_ptr<Eigen::ThreadPool> pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> device_;
};

static EigenThreadPool& tp() {
  static EigenThreadPool tp(ENV_PARAM(NUM_OF_ENGINE_THREAD));
  static bool init = false;

  if (!init) {
    vaip_core::add_cleanup_function("Eigen pool", []() { tp.Clear(); });
    init = true;
  }
  return tp;
}

// in Eigen convention, the first dimention is continous in memory.
// a(d0, d1, d2, ...., dn), where d0 varis in memory first, i.e. stride = 1
// shape

// float data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
//                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
//                   29};
//   Eigen::TensorMap<Eigen::Tensor<float, 3>> a(data,
//                                               std::array<int, 3>{2, 3, 5});
//   std::cout << "NumRows: " << a.dimension(0) << " NumCols: " <<
//   a.dimension(1)
//             << " NumZ " << a.dimension(2) << std::endl;
//
//   for (auto z = 0u; z < a.dimension(2); ++z) {
//     for (auto y = 0u; y < a.dimension(1); ++y) {
//       for (auto x = 0u; x < a.dimension(0); ++x) {
//         std::cout << a(x, y, z) << " ";
//       }
//     }
//     std::cout << std::endl;
//   }

template <int NDIMS, typename T>
void transpose1(const T* src, T* dst, const std::array<int, NDIMS>& shape_src,
                const std::array<int, NDIMS>& perm) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = (int)(NDIMS - 1 - perm[i]);
  }
  Eigen::TensorMap<Eigen::Tensor<T, NDIMS>, Eigen::Aligned> x(
      const_cast<T*>(src), shape_src);
  Eigen::array<int, NDIMS> shape_dst;
  for (int i = 0; i < NDIMS; ++i) {
    shape_dst[i] = shape_src[p[i]];
  }
  Eigen::TensorMap<Eigen::Tensor<T, NDIMS>, Eigen::Aligned> y(dst, shape_dst);
  y.device(tp().get_device()) = x.shuffle(p);
}

template <int NDIMS, typename C>
static std::array<int, NDIMS> to_array(const C& v) {
  auto ret = std::array<int, NDIMS>();
  DCHECK_EQ(v.size(), (size_t)NDIMS);
  for (auto i = 0u; i < v.size(); ++i) {
    ret[i] = (int)v[i];
  }
  return ret;
}

template <typename T> static std::vector<T> reverse(const std::vector<T>& v) {
  std::vector<T> ret(v);
  std::reverse(ret.begin(), ret.end());
  return ret;
}

template <typename T, typename Shape, typename Perm>
void transpose0(const T* src, T* dst, const Shape& shape, const Perm& perm) {
  CHECK_EQ(shape.size(), perm.size());
  auto size = shape.size();
  switch (size) {
  case 2: {
    transpose1<2, T>(src, dst, to_array<2>(reverse(shape)),
                     to_array<2>(reverse(perm)));
    break;
  }
  case 3: {
    // test case: issue #1143
    transpose1<3, T>(src, dst, to_array<3>(reverse(shape)),
                     to_array<3>(reverse(perm)));
    break;
  }
  case 4: {
    transpose1<4, T>(src, dst, to_array<4>(reverse(shape)),
                     to_array<4>(reverse(perm)));
    break;
  }
  case 5: {
    transpose1<5, T>(src, dst, to_array<5>(reverse(shape)),
                     to_array<5>(reverse(perm)));
    break;
  }
  default:
    LOG(FATAL) << "unsupported rank. rank=" << size;
  }
  return;
}
} // namespace

namespace vaip_core {
void transpose_f(const float* src, float* dst,
                 const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& perm) {
  transpose0<float, std::vector<int64_t>, std::vector<int64_t>>(src, dst, shape,
                                                                perm);
}

void transpose_i8(const int8_t* src, int8_t* dst,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& perm) {
  transpose0<int8_t, std::vector<int64_t>, std::vector<int64_t>>(src, dst,
                                                                 shape, perm);
}

void transpose_ui8(const uint8_t* src, uint8_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<uint8_t, std::vector<int64_t>, std::vector<int64_t>>(src, dst,
                                                                  shape, perm);
}

void transpose_i16(const int16_t* src, int16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<int16_t, std::vector<int64_t>, std::vector<int64_t>>(src, dst,
                                                                  shape, perm);
}
void transpose_u16(const uint16_t* src, uint16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<uint16_t, std::vector<int64_t>, std::vector<int64_t>>(src, dst,
                                                                   shape, perm);
}

void transpose_bf16(const xir::bfloat16_t* src, xir::bfloat16_t* dst,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& perm) {
  transpose0<uint16_t, std::vector<int64_t>, std::vector<int64_t>>(
      (const uint16_t*)src, (uint16_t*)dst, shape, perm);
}

} // namespace vaip_core
