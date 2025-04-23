/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include "morphizen/transpose.hpp"
#include "morphizen/env_config.hpp"
#include <algorithm>
#include <chrono>
#include <glog/logging.h>

namespace {

inline void calc_dst_shape_and_flat_src_step(const int* src_shape,
                                             const int* perm, const int& NDIMS,
                                             int* dst_shape,
                                             int* flat_src_step) {
  for (int i = 0; i < NDIMS; ++i) {
    dst_shape[i] = src_shape[NDIMS - 1 - perm[i]];
  }
  int p_inv[5];
  for (int i = 0; i < NDIMS; ++i) {
    p_inv[i] = NDIMS - 1 - perm[i];
  }
  int src_stride[5];
  src_stride[0] = 1;
  for (int i = 1; i < NDIMS; ++i) {
    src_stride[i] = src_stride[i - 1] * src_shape[i - 1];
  }
  int st[5];
  for (int i = 0; i < NDIMS; i++) {
    st[i] = src_stride[p_inv[i]];
  }
  flat_src_step[0] = st[0];
  for (int i = 1; i < NDIMS; i++) {
    flat_src_step[i] = st[i] - dst_shape[i - 1] * st[i - 1];
  }
}

inline void calc_flat_dst_step(const int* src_shape, const int* perm,
                               const int& NDIMS, const int* dst_shape,
                               int* flat_dst_step) {
  int p_inv[5];
  for (int i = 0; i < NDIMS; ++i) {
    p_inv[NDIMS - 1 - perm[i]] = i;
  }
  int dst_stride[5];
  dst_stride[0] = 1;
  for (int i = 1; i < NDIMS; ++i) {
    dst_stride[i] = dst_stride[i - 1] * dst_shape[i - 1];
  }
  int st[5];
  for (int i = 0; i < NDIMS; i++) {
    st[i] = dst_stride[p_inv[i]];
  }
  flat_dst_step[0] = st[0];
  for (int i = 1; i < NDIMS; i++) {
    flat_dst_step[i] = st[i] - src_shape[i - 1] * st[i - 1];
  }
}
template <typename T>
void transpose1_2(const T* src, T* dst, const int* src_shape, const int* perm) {
  const int NDIMS = 2;
  int dst_shape[NDIMS];
  int flat_src_step[NDIMS];
  calc_dst_shape_and_flat_src_step(src_shape, perm, NDIMS, dst_shape,
                                   flat_src_step);
  if (dst_shape[0] < src_shape[0]) {
    int flat_dst = 0, flat_src = 0;
    for (int idx1 = 0; idx1 < dst_shape[1];
         ++idx1, flat_src += flat_src_step[1]) {
      for (int flat_dst_end = flat_dst + dst_shape[0]; flat_dst < flat_dst_end;
           flat_src += flat_src_step[0], ++flat_dst) {
        dst[flat_dst] = src[flat_src];
      }
    }
  } else {
    int flat_dst_step[NDIMS];
    calc_flat_dst_step(src_shape, perm, NDIMS, dst_shape, flat_dst_step);
    int flat_dst = 0, flat_src = 0;
    for (int idx1 = 0; idx1 < src_shape[1];
         ++idx1, flat_dst += flat_dst_step[1]) {
      for (int flat_src_end = flat_src + src_shape[0]; flat_src < flat_src_end;
           ++flat_src, flat_dst += flat_dst_step[0]) {
        dst[flat_dst] = src[flat_src];
      }
    }
  }
}
template <typename T>
void transpose1_3(const T* src, T* dst, const int* src_shape, const int* perm) {
  const int NDIMS = 3;
  int dst_shape[NDIMS];
  int flat_src_step[NDIMS];
  calc_dst_shape_and_flat_src_step(src_shape, perm, NDIMS, dst_shape,
                                   flat_src_step);
  int flat_dst = 0, flat_src = 0;
  for (int idx2 = 0; idx2 < dst_shape[2];
       ++idx2, flat_src += flat_src_step[2]) {
    for (int idx1 = 0; idx1 < dst_shape[1];
         ++idx1, flat_src += flat_src_step[1]) {
      int flat_dst_end = flat_dst + dst_shape[0];
      for (; flat_dst < flat_dst_end;
           flat_src += flat_src_step[0], ++flat_dst) {
        dst[flat_dst] = src[flat_src];
      }
    }
  }
}
template <typename T>
void transpose1_4(const T* src, T* dst, const int* src_shape, const int* perm) {
  const int NDIMS = 4;
  int dst_shape[NDIMS];
  int flat_src_step[NDIMS];
  calc_dst_shape_and_flat_src_step(src_shape, perm, NDIMS, dst_shape,
                                   flat_src_step);
  int flat_dst = 0, flat_src = 0;
  for (int idx3 = 0; idx3 < dst_shape[3];
       ++idx3, flat_src += flat_src_step[3]) {
    for (int idx2 = 0; idx2 < dst_shape[2];
         ++idx2, flat_src += flat_src_step[2]) {
      for (int idx1 = 0; idx1 < dst_shape[1];
           ++idx1, flat_src += flat_src_step[1]) {
        auto dst_end = flat_dst + dst_shape[0];
        for (; flat_dst < dst_end; flat_src += flat_src_step[0], ++flat_dst) {
          dst[flat_dst] = src[flat_src];
        }
      }
    }
  }
}

template <typename T>
void transpose1_5(const T* src, T* dst, const int* src_shape, const int* perm) {
  const int NDIMS = 5;
  int dst_shape[NDIMS];
  int flat_src_step[NDIMS];
  calc_dst_shape_and_flat_src_step(src_shape, perm, NDIMS, dst_shape,
                                   flat_src_step);
  int flat_dst = 0, flat_src = 0;

  for (int idx4 = 0; idx4 < dst_shape[4];
       ++idx4, flat_src += flat_src_step[4]) {
    for (int idx3 = 0; idx3 < dst_shape[3];
         ++idx3, flat_src += flat_src_step[3]) {
      for (int idx2 = 0; idx2 < dst_shape[2];
           ++idx2, flat_src += flat_src_step[2]) {
        for (int idx1 = 0; idx1 < dst_shape[1];
             ++idx1, flat_src += flat_src_step[1]) {
          auto dst_end = flat_dst + dst_shape[0];
          for (; flat_dst < dst_end; flat_src += flat_src_step[0], ++flat_dst) {
            dst[flat_dst] = src[flat_src];
          }
        }
      }
    }
  }
}
inline void reverse(const std::vector<int64_t>& v, int NDIMS, int* ret) {
  DCHECK_EQ(v.size(), (size_t)NDIMS);
  for (size_t i = 0u, j = v.size() - 1; i < v.size(); ++i, --j) {
    ret[i] = (int)v[j];
  }
}
void optimize_shape(int* shape, int* perm, int& NDIMS) {
  int l_end = NDIMS;
  for (int l = 0; l < l_end; l++) {
    for (int i = 0; i < NDIMS && NDIMS > 1; i++) {
      if (shape[i] == 1) {
        for (int j = i + 1; j < NDIMS; j++) {
          shape[j - 1] = shape[j];
        }
        int delperm = NDIMS - 1 - i;
        int k = 0;
        for (; k < NDIMS; ++k) {
          if (perm[k] == delperm) {
            break;
          }
        }
        for (int j = k; j < NDIMS - 1; ++j) {
          perm[j] = perm[j + 1];
        }
        for (int j = 0; j < NDIMS - 1; j++) {
          if (perm[j] > delperm) {
            perm[j]--;
          }
        }
        NDIMS -= 1;
        break;
      }
    }
  }
  l_end = NDIMS;
  for (int l = 0; l < l_end; l++) {
    for (int i = 0; i < NDIMS - 1; i++) {
      if (perm[i] - 1 == perm[i + 1]) {
        shape[NDIMS - 1 - (perm[i] - 1)] *= shape[NDIMS - 1 - perm[i]];
        for (int j = NDIMS - 1 - perm[i] + 1; j < NDIMS; j++) {
          shape[j - 1] = shape[j];
        }
        int delperm = perm[i];
        int k = 0;
        for (; k < NDIMS; ++k) {
          if (perm[k] == delperm) {
            break;
          }
        }
        for (int j = k; j < NDIMS - 1; ++j) {
          perm[j] = perm[j + 1];
        }
        for (int j = 0; j < NDIMS - 1; j++) {
          if (perm[j] > delperm) {
            perm[j]--;
          }
        }
        NDIMS -= 1;
        break;
      }
    }
  }
}
template <typename T>
void transpose0(const T* src, T* dst, const std::vector<int64_t>& shape,
                const std::vector<int64_t>& perm) {
  CHECK_EQ(shape.size(), perm.size());
  int NDIMS = int(shape.size());
  CHECK_LE(NDIMS, 5) << "unsupported rank. rank=" << NDIMS;
  int shape_reverse[5];
  int perm_reverse[5];
  reverse(shape, NDIMS, shape_reverse);
  reverse(perm, NDIMS, perm_reverse);
  optimize_shape(shape_reverse, perm_reverse, NDIMS);
  switch (NDIMS) {
  case 1:
    memcpy(dst, src, shape_reverse[0] * sizeof(T));
    break;
  case 2:
    transpose1_2(src, dst, shape_reverse, perm_reverse);
    break;
  case 3:
    transpose1_3(src, dst, shape_reverse, perm_reverse);
    break;
  case 4:
    transpose1_4(src, dst, shape_reverse, perm_reverse);
    break;
  case 5:
    transpose1_5(src, dst, shape_reverse, perm_reverse);
    break;
  default:
    LOG(FATAL) << "unsupported rank. rank=" << NDIMS;
    break;
  }
  return;
}
} // namespace

namespace vaip_core {
void transpose_f(const float* src, float* dst,
                 const std::vector<int64_t>& shape,
                 const std::vector<int64_t>& perm) {
  transpose0<float>(src, dst, shape, perm);
}

void transpose_i8(const int8_t* src, int8_t* dst,
                  const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& perm) {
  transpose0<int8_t>(src, dst, shape, perm);
}

void transpose_ui8(const uint8_t* src, uint8_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<uint8_t>(src, dst, shape, perm);
}

void transpose_i16(const int16_t* src, int16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<int16_t>(src, dst, shape, perm);
}
void transpose_u16(const uint16_t* src, uint16_t* dst,
                   const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm) {
  transpose0<uint16_t>(src, dst, shape, perm);
}
void transpose_bf16(const xir::bfloat16_t* src, xir::bfloat16_t* dst,
                    const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& perm) {
  transpose0<uint16_t>((const uint16_t*)src, (uint16_t*)dst, shape, perm);
}

} // namespace vaip_core