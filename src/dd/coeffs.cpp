/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
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
#include "vaip/dd/coeffs.hpp"

namespace vaip::dd::qmatmulcalc {

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

std::pair<int32_t, int16_t> find_closest_shifted_int32(double float_val,
                                                       int32_t max_value) {
  int32_t INT32_MAX_ = max_value; // Typically 2147483647 for int32_t
  double prev_rel_err = 1e9;
  double curr_float_val = float_val;
  // double best_float_val = 0.0;
  int16_t shift_val = 0;
  int32_t best_int = 0;
  int64_t closest_curr_int = 0;
  int16_t best_shift_val = 0;

  while (curr_float_val <= INT32_MAX_) {
    closest_curr_int = static_cast<int64_t>(std::round(curr_float_val));
    double cur_rel_err =
        std::abs((double)float_val -
                 ((double)closest_curr_int /
                  static_cast<double>((uint64_t)1 << shift_val))) /
        (double)float_val;

    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      // best_float_val = static_cast<double>(closest_curr_int >> shift_val);
      best_shift_val = shift_val;
      best_int = (int32_t)closest_curr_int;
    }

    curr_float_val *= 2;
    shift_val++;
  }

  return std::make_pair(best_int, best_shift_val);
}

std::pair<int32_t, int16_t> find_closest_shifted_int32_gap(double float_val,
                                                           int32_t max_value) {
  int32_t INT32_MAX_ = max_value; // Typically 2147483647 for int32_t
  int neg = 1;
  if (float_val < 0.0f)
    neg = -1;
  double prev_rel_err = 1e9;
  float_val = std::abs(float_val);
  double curr_float_val = float_val;
  // double best_float_val = 0.0;
  int16_t shift_val = 0;
  int32_t best_int = 0;
  int64_t closest_curr_int = 0;
  int16_t best_shift_val = 0;

  while (curr_float_val <= INT32_MAX_) {
    closest_curr_int = static_cast<int64_t>(std::round(curr_float_val));

    double cur_rel_err = std::abs((double)float_val - (double)closest_curr_int /
                                                          pow(2, shift_val)) /
                         (double)float_val;

    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      best_shift_val = shift_val;
      best_int = (int32_t)closest_curr_int;
    }

    curr_float_val *= 2;
    shift_val++;
  }
  best_int = best_int * neg;
  return std::make_pair(best_int, best_shift_val);
}

std::pair<int16_t, int16_t> find_closest_shifted_int16(double float_val,
                                                       int32_t max_value) {
  int32_t INT32_MAX_ = max_value; // Typically 2147483647 for int32_t
  double prev_rel_err = 1e9;
  double curr_float_val = float_val;
  // double best_float_val = 0.0;
  int16_t shift_val = 0;
  int16_t best_int = 0;
  int64_t closest_curr_int = 0;
  int16_t best_shift_val = 0;

  while (curr_float_val <= INT32_MAX_) {
    closest_curr_int = static_cast<int64_t>(std::round(curr_float_val));
    double cur_rel_err =
        std::abs((double)float_val -
                 ((double)closest_curr_int /
                  static_cast<double>((uint64_t)1 << shift_val))) /
        (double)float_val;

    if (cur_rel_err < prev_rel_err) {
      prev_rel_err = cur_rel_err;
      // best_float_val = static_cast<double>(closest_curr_int >> shift_val);
      best_shift_val = shift_val;
      best_int = (int16_t)closest_curr_int;
    }

    curr_float_val *= 2;
    shift_val++;
  }

  return std::make_pair(best_int, best_shift_val);
}

MatmulQDQParams calculate_matmuladd_qdq_params_uint16_uint8(
    const std::vector<std::vector<uint8_t>>& weights,
    const std::vector<uint16_t>& bias, float a_sc, uint16_t a_zp, float w_sc,
    uint16_t w_zp, float b_sc, uint16_t b_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t b_zp_int64 = static_cast<int64_t>(b_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = (int64_t)(std::min(
      std::max((int)std::ceil(std::log2(weights_in_ch)) - 7, 0), 7));
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  std::vector<int64_t> bias_min_zp(bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    bias_min_zp[i] = (int64_t)bias[i] - b_zp_int64;
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  double c4_coeff = b_sc / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [_c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);
  int64_t c4_coeff_prime = _c4_coeff_prime;
  if (shft_c2 != shft_c4) {
    auto diff_shft_c2_c4 = shft_c2 - shft_c4;
    auto abs_diff_shft_c2_c4 = std::abs((int64_t)(diff_shft_c2_c4));
    if (diff_shft_c2_c4 > 0)
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    else if (diff_shft_c2_c4 < 0)
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    else
      c4_coeff_prime = c4_coeff_prime;
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }

    c1_coeff[i] = (-a_zp_int64) * c2_coeff_prime * weights_sum +
                  (q_zp_int64 << shft_c2) + bias_min_zp[i] * c4_coeff_prime;
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int32_t c2 = int(c2_coeff_prime << matmul_shift);
  int32_t c1 = int(c3_coeff_scale);

  MatmulQDQParams ret;
  std::vector<int64_t> c0(weights[0].size(), 0);

  int64_t temp2 = static_cast<int64_t>(
      c3_coeff_scale * ((int64_t)c3_coeff_offset << c3_coeff_scale_shift));
  std::transform(c1_coeff.begin(), c1_coeff.end(), c0.begin(),
                 [temp2](int64_t c) { return c + temp2; });
  ret.c0_coeffs = c0;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2;
  // ret.qdq_params[4] = 0;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  ret.qdq_params[10] = 1;
  ret.c1 = c1;
  ret.c2 = c2;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams calculate_matmuladd_qdq_params_uint8_uint8(
    const std::vector<std::vector<uint8_t>>& weights,
    const std::vector<uint16_t>& bias, float a_sc, uint16_t a_zp, float w_sc,
    uint16_t w_zp, float b_sc, uint16_t b_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t b_zp_int64 = static_cast<int64_t>(b_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = 0;
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  std::vector<int64_t> bias_min_zp(bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    bias_min_zp[i] = (int64_t)bias[i] - b_zp_int64;
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  double c4_coeff = b_sc / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);
  if (shft_c2 != shft_c4) {
    auto diff_shft_c2_c4 = shft_c2 - shft_c4;
    auto abs_diff_shft_c2_c4 = std::abs((int64_t)(diff_shft_c2_c4));
    if (diff_shft_c2_c4 > 0)
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    else if (diff_shft_c2_c4 < 0)
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    else
      c4_coeff_prime = c4_coeff_prime;
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_zp_int64) * c2_coeff_prime * weights_sum +
                  (q_zp_int64 << shft_c2) + bias_min_zp[i] * c4_coeff_prime;
  }
  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });
  MatmulQDQParams ret;
  ret.c0_coeffs = c1_coeff;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2_coeff_prime;
  // ret.qdq_params[4] = 0;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  // ret.qdq_params[10] = 0L;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams calculate_matmuladd_qdq_params_uint16_uint8_b32(
    const std::vector<std::vector<uint8_t>>& weights,
    const std::vector<int32_t>& bias, float a_sc, uint16_t a_zp, float w_sc,
    uint16_t w_zp, float b_sc, uint16_t b_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t b_zp_int64 = static_cast<int64_t>(b_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = (int64_t)(std::min(
      std::max((int)std::ceil(std::log2(weights_in_ch)) - 7, 0), 7));
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  std::vector<int64_t> bias_min_zp(bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    bias_min_zp[i] = (int64_t)bias[i] - b_zp_int64;
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  double c4_coeff = b_sc / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [_c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);
  int64_t c4_coeff_prime = _c4_coeff_prime;
  if (shft_c2 != shft_c4) {
    auto diff_shft_c2_c4 = shft_c2 - shft_c4;
    auto abs_diff_shft_c2_c4 = std::abs((int64_t)(diff_shft_c2_c4));
    if (diff_shft_c2_c4 > 0)
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    else if (diff_shft_c2_c4 < 0)
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    else
      c4_coeff_prime = c4_coeff_prime;
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }

    c1_coeff[i] = (-a_zp_int64) * c2_coeff_prime * weights_sum +
                  (q_zp_int64 << shft_c2) + bias_min_zp[i] * c4_coeff_prime;
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int32_t c2 = int(c2_coeff_prime << matmul_shift);
  int32_t c1 = int(c3_coeff_scale);

  MatmulQDQParams ret;
  std::vector<int64_t> c0(weights[0].size(), 0);

  int64_t temp2 = static_cast<int64_t>(
      c3_coeff_scale * ((int64_t)c3_coeff_offset << c3_coeff_scale_shift));
  std::transform(c1_coeff.begin(), c1_coeff.end(), c0.begin(),
                 [temp2](int64_t c) { return c + temp2; });
  ret.c0_coeffs = c0;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2;
  // ret.qdq_params[4] = 0;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  ret.qdq_params[10] = 1;
  ret.c1 = c1;
  ret.c2 = c2;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams calculate_matmuladd_qdq_params_uint8_uint8_b32(
    const std::vector<std::vector<uint8_t>>& weights,
    const std::vector<int32_t>& bias, float a_sc, uint16_t a_zp, float w_sc,
    uint16_t w_zp, float b_sc, uint16_t b_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t b_zp_int64 = static_cast<int64_t>(b_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = 0;
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  std::vector<int64_t> bias_min_zp(bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    bias_min_zp[i] = (int64_t)bias[i] - b_zp_int64;
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  double c4_coeff = b_sc / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);
  if (shft_c2 != shft_c4) {
    auto diff_shft_c2_c4 = shft_c2 - shft_c4;
    auto abs_diff_shft_c2_c4 = std::abs((int64_t)(diff_shft_c2_c4));
    if (diff_shft_c2_c4 > 0)
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    else if (diff_shft_c2_c4 < 0)
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    else
      c4_coeff_prime = c4_coeff_prime;
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_zp_int64) * c2_coeff_prime * weights_sum +
                  (q_zp_int64 << shft_c2) + bias_min_zp[i] * c4_coeff_prime;
  }
  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });
  MatmulQDQParams ret;
  ret.c0_coeffs = c1_coeff;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2_coeff_prime;
  // ret.qdq_params[4] = 0;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  // ret.qdq_params[10] = 0L;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams calculate_matmul_qdq_params_uint8_uint8(
    const std::vector<std::vector<uint8_t>>& weights, float a_sc, uint16_t a_zp,
    float w_sc, uint16_t w_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = 0;
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] =
        (-a_zp_int64) * c2_coeff_prime * weights_sum + (q_zp_int64 << shft_c2);
  }
  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });
  MatmulQDQParams ret;
  ret.c0_coeffs = c1_coeff;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2_coeff_prime;
  ret.qdq_params[4] = c2_coeff_prime;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  // ret.qdq_params[10] = 0L;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams_3d calculate_matmul_3d_qdq_params_uint16_uint8(
    const std::vector<std::vector<std::vector<uint8_t>>>& weights, float a_sc,
    uint16_t a_zp, float w_sc, uint16_t w_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights[0].size());
  int64_t matmul_shift = (int64_t)(std::min(
      std::max(25 + (int)std::ceil(std::log2(weights_in_ch)) - 32, 0), 7));
  // Copy weights to int64_t
  std::vector<std::vector<std::vector<int64_t>>> weights_int64(
      weights.size(),
      std::vector<std::vector<int64_t>>(
          weights[0].size(), std::vector<int64_t>(weights[0][0].size())));

  MatmulQDQParams_3d ret;
  ret.c0_coeffs.resize(weights.size());
  double c2_coeff = (a_sc * w_sc) / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int32_t c2 = (c2_coeff_prime << matmul_shift);
  int64_t temp = c3_coeff_scale * (c3_coeff_offset << c3_coeff_scale_shift);

  for (size_t batch = 0; batch < weights.size(); ++batch) {
    for (size_t i = 0; i < weights[batch].size(); ++i) {
      for (size_t j = 0; j < weights[batch][i].size(); ++j) {
        weights_int64[batch][i][j] = static_cast<int64_t>(weights[batch][i][j]);
      }
    }
    std::vector<int64_t> c1_coeff(weights[0][0].size());
    for (size_t i = 0; i < weights[0][0].size(); ++i) {
      int64_t weights_sum = 0;
      for (size_t j = 0; j < weights[0].size(); ++j) {
        weights_sum += weights_int64[batch][j][i];
      }
      c1_coeff[i] = (-a_zp_int64) * c2_coeff_prime * weights_sum +
                    (q_zp_int64 << shft_c2);
    }
    std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                   [temp](int64_t c) { return c + temp; });
    ret.c0_coeffs[batch] = c1_coeff;
  }
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2;
  ret.qdq_params[4] = c2;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  ret.qdq_params[10] = 1;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

MatmulQDQParams calculate_matmul_qdq_params_uint16_uint8(
    const std::vector<std::vector<uint8_t>>& weights, float a_sc, uint16_t a_zp,
    float w_sc, uint16_t w_zp, float q_sc, uint16_t q_zp) {
  int64_t a_zp_int64 = static_cast<int64_t>(a_zp);
  int64_t w_zp_int64 = static_cast<int64_t>(w_zp);
  int64_t q_zp_int64 = static_cast<int64_t>(q_zp);
  int64_t weights_in_ch = static_cast<int64_t>(weights.size());
  int64_t matmul_shift = (int64_t)(std::min(
      std::max(25 + (int)std::ceil(std::log2(weights_in_ch)) - 32, 0), 7));
  // Copy weights to int64_t
  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }
  double c2_coeff = (a_sc * w_sc) / q_sc;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);
  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] =
        (-a_zp_int64) * c2_coeff_prime * weights_sum + (q_zp_int64 << shft_c2);
  }
  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = (int32_t)(-a_zp_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp_int64;

  int64_t c3_coeff_scale_shift = 0;
  // right shift c3 coeff_scale to ensure fits into int32
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int32_t c2 = (c2_coeff_prime << matmul_shift);
  int64_t temp = c3_coeff_scale * (c3_coeff_offset << c3_coeff_scale_shift);
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });
  MatmulQDQParams ret;
  ret.c0_coeffs = c1_coeff;
  ret.qdq_params.resize(16, 0);
  ret.qdq_params[2] = (int32_t)c3_coeff_scale;
  ret.qdq_params[3] = c2;
  ret.qdq_params[4] = c2;
  ret.qdq_params[5] = 64;
  ret.qdq_params[6] = 64;
  ret.qdq_params[7] = (int32_t)c3_coeff_scale_shift;
  ret.qdq_params[8] = shft_c2;
  ret.qdq_params[9] = (int32_t)matmul_shift;
  ret.qdq_params[10] = 1;
  ret.c3_coeff_scale = (int32_t)c3_coeff_scale;
  ret.c2_coeff_prime = c2_coeff_prime;
  ret.c3_coeff_scale_shift = c3_coeff_scale_shift;
  ret.shft_c2 = shft_c2;
  ret.matmul_shift = matmul_shift;
  // ret.qdq_params.resize(16);

  return ret;
}

// Function to convert float to bfloat16
uint16_t float_to_bfloat16(float value) {
  // Reinterpret the float as an unsigned 32-bit integer
  uint32_t int_value;
  std::memcpy(&int_value, &value, sizeof(value));

  // Extract the least significant bit (lsb) of the original float's lower 16
  // bits
  uint32_t lsb = (int_value >> 16) & 1;

  // Calculate the bias
  uint32_t bias = 0x7FFF + lsb;

  // Add the bias to the original integer value
  uint32_t new_value = int_value + bias;

  // Shift the new value right by 16 bits to get the bfloat16 representation
  uint16_t bfloat16_value = (uint16_t)(new_value >> 16);

  return bfloat16_value;
}

std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>
calc_eltwise_coeff(float a_sc, uint16_t a_zp, float b_sc, uint16_t b_zp) {
  return {float_to_bfloat16(a_sc), a_zp, float_to_bfloat16(b_sc), b_zp};
}
std::tuple<uint16_t, uint16_t> calc_lrn_coeff(float q_sc, uint16_t q_zp) {
  return {float_to_bfloat16(q_sc), q_zp};
}

std::tuple<int64_t, int32_t, int32_t>
global_avg_pool_qdq(double a_sc, uint16_t a_zp, double b_sc, uint16_t b_zp) {
  int32_t ifm_sv_height = 1;
  int32_t ifm_sv_width_eff = 49;
  int32_t k = ifm_sv_height * ifm_sv_width_eff;
  double c0f = ((double)b_sc / (double)(a_sc * k));
  double c1f = (b_zp - ((double)(a_zp * b_sc / a_sc)));

  auto [C0, C0_shift] = find_closest_shifted_int32_gap(c0f, 2147483647);

  auto [C1, C1_shift] = find_closest_shifted_int32_gap(c1f, 2147483647);

  int32_t diffShift = (int32_t)(C1_shift - C0_shift);
  int64_t offset = 0;
  if (diffShift < 0) {
    offset = (int64_t)C1 << -diffShift;
  } else {
    offset = (int64_t)C1 >> diffShift;
  }

  int32_t divFactor = C0;
  int32_t divShift = C0_shift;

  return {offset, divFactor, divShift};
}

std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t, int64_t>
compute_qdq_coeff_matmul_bias(float a_dq_xscale, uint8_t a_dq_xzero_pt,
                              const std::vector<std::vector<uint8_t>>& weights,
                              float w_dq_xscale, uint8_t w_dq_xzero_pt,
                              const std::vector<uint16_t>& bias,
                              float b_dq_xscale, uint8_t b_dq_xzero_pt,
                              float a_q_yscale, uint8_t a_q_yzero_pt) {
  int64_t a_dq_xzero_pt_int64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_int64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_int64 = static_cast<int64_t>(a_q_yzero_pt);

  // assert(weights.size() > 0 && weights[0].size() > 0);  // weights shape
  // should be 2 dims

  int64_t weights_in_ch = static_cast<int64_t>(weights.size());

  int64_t matmul_shift = 0;

  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }

  std::vector<int64_t> bias_min_zp(bias.size());
  std::transform(bias.begin(), bias.end(), bias_min_zp.begin(),
                 [b_dq_xzero_pt](uint16_t b) {
                   return static_cast<int64_t>(b) -
                          static_cast<int64_t>(b_dq_xzero_pt);
                 });

  double c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  double c4_coeff = b_dq_xscale / a_q_yscale;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);

  if (shft_c2 != shft_c4) {
    int64_t diff_shft_c2_c4 = shft_c2 - shft_c4;
    int64_t abs_diff_shft_c2_c4 = std::abs(diff_shft_c2_c4);
    if (diff_shft_c2_c4 > 0) {
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    } else if (diff_shft_c2_c4 < 0) {
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    }
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_dq_xzero_pt_int64) * c2_coeff_prime * weights_sum +
                  (a_q_yzero_pt_int64 << shft_c2) +
                  (bias_min_zp[i] * c4_coeff_prime);
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_int64;
  int64_t c3_coeff_scale_shift = 0;

  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });

  return std::make_tuple(c1_coeff, (int32_t)c3_coeff_scale,
                         (int64_t)c2_coeff_prime, (int64_t)c3_coeff_scale_shift,
                         (int64_t)shft_c2, (int64_t)matmul_shift);
}

std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t, int64_t>
dq_uint16A_uint16W_bias_matmul_q_param_gen(
    float a_dq_xscale, uint16_t a_dq_xzero_pt,
    const std::vector<std::vector<uint16_t>>& weights, float w_dq_xscale,
    uint16_t w_dq_xzero_pt, const std::vector<uint16_t>& bias,
    float b_dq_xscale, uint16_t b_dq_xzero_pt, float a_q_yscale,
    uint16_t a_q_yzero_pt, std::vector<int> shifts) {

  int64_t a_dq_xzero_pt_int64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_int64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_int64 = static_cast<int64_t>(a_q_yzero_pt);

  int64_t weights_in_ch = static_cast<int64_t>(weights.size());

  int64_t matmul_shift = std::min(
      std::max(static_cast<int64_t>(std::ceil(std::log2(weights_in_ch))) - 1,
               int64_t(0)),
      int64_t(15));

  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }

  std::vector<int64_t> bias_min_zp(bias.size());
  std::transform(bias.begin(), bias.end(), bias_min_zp.begin(),
                 [b_dq_xzero_pt](uint16_t b) {
                   return static_cast<int64_t>(b) -
                          static_cast<int64_t>(b_dq_xzero_pt);
                 });

  double c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  double c4_coeff = b_dq_xscale / a_q_yscale;
  auto [c2_coeff_prime, shft_c2] = find_closest_shifted_int16(c2_coeff, 32767);
  auto [_c4_coeff_prime, shft_c4] = find_closest_shifted_int16(c4_coeff, 32767);
  int64_t c4_coeff_prime = _c4_coeff_prime;

  if (shft_c2 != shft_c4) {
    int64_t diff_shft_c2_c4 = shft_c2 - shft_c4;
    int64_t abs_diff_shft_c2_c4 = std::abs(diff_shft_c2_c4);
    if (diff_shft_c2_c4 > 0) {
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    } else if (diff_shft_c2_c4 < 0) {
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    }
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_dq_xzero_pt_int64) * c2_coeff_prime * weights_sum +
                  (a_q_yzero_pt_int64 << shft_c2) +
                  (bias_min_zp[i] * c4_coeff_prime);
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_int64;
  int64_t c3_coeff_scale_shift = 0;

  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);

  int64_t temp = c3_coeff_scale * c3_coeff_offset << c3_coeff_scale_shift;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp, matmul_shift](int64_t c) { return (c + temp); });

  // int32_t C2 = c2_coeff_prime;
  int32_t C2 = c2_coeff_prime << matmul_shift;

  c3_coeff_scale = c3_coeff_scale >> shifts[1];
  C2 = C2 >> shifts[2];
  int sft = shifts[0];
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp, sft](int64_t c) { return (c >> sft); });

  return std::make_tuple(c1_coeff, (int32_t)c3_coeff_scale, C2,
                         (int64_t)c3_coeff_scale_shift, (int64_t)shft_c2,
                         (int64_t)matmul_shift);
}

std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t, int64_t>
dq_uint16A_uint8W_bias_matmul_q_param_gen(
    float a_dq_xscale, uint16_t a_dq_xzero_pt,
    const std::vector<std::vector<uint8_t>>& weights, float w_dq_xscale,
    uint16_t w_dq_xzero_pt, const std::vector<uint16_t>& bias,
    float b_dq_xscale, uint16_t b_dq_xzero_pt, float a_q_yscale,
    uint16_t a_q_yzero_pt) {

  int64_t a_dq_xzero_pt_int64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_int64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_int64 = static_cast<int64_t>(a_q_yzero_pt);

  // assert(weights.size() > 0 && weights[0].size() > 0);  // weights shape
  // should be 2 dims

  int64_t weights_in_ch = static_cast<int64_t>(weights.size());

  int64_t matmul_shift = std::min(
      std::max(static_cast<int64_t>(std::ceil(std::log2(weights_in_ch))) - 7,
               int64_t(0)),
      int64_t(7));

  std::vector<std::vector<int64_t>> weights_int64(
      weights.size(), std::vector<int64_t>(weights[0].size()));
  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      weights_int64[i][j] = static_cast<int64_t>(weights[i][j]);
    }
  }

  std::vector<int64_t> bias_min_zp(bias.size());
  std::transform(bias.begin(), bias.end(), bias_min_zp.begin(),
                 [b_dq_xzero_pt](uint16_t b) {
                   return static_cast<int64_t>(b) -
                          static_cast<int64_t>(b_dq_xzero_pt);
                 });

  double c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  double c4_coeff = b_dq_xscale / a_q_yscale;
  auto [c2_coeff_prime, shft_c2] =
      find_closest_shifted_int32(c2_coeff, 8388607);
  auto [_c4_coeff_prime, shft_c4] =
      find_closest_shifted_int32(c4_coeff, 8388607);
  int64_t c4_coeff_prime = _c4_coeff_prime;

  if (shft_c2 != shft_c4) {
    int64_t diff_shft_c2_c4 = shft_c2 - shft_c4;
    int64_t abs_diff_shft_c2_c4 = std::abs(diff_shft_c2_c4);
    if (diff_shft_c2_c4 > 0) {
      c4_coeff_prime = c4_coeff_prime << abs_diff_shft_c2_c4;
    } else if (diff_shft_c2_c4 < 0) {
      c4_coeff_prime = c4_coeff_prime >> abs_diff_shft_c2_c4;
    }
  }

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  std::vector<int64_t> c1_coeff(weights[0].size());
  for (size_t i = 0; i < weights[0].size(); ++i) {
    int64_t weights_sum = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      weights_sum += weights_int64[j][i];
    }
    c1_coeff[i] = (-a_dq_xzero_pt_int64) * c2_coeff_prime * weights_sum +
                  (a_q_yzero_pt_int64 << shft_c2) +
                  (bias_min_zp[i] * c4_coeff_prime);
  }

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_int64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_int64;
  int64_t c3_coeff_scale_shift = 0;

  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  int64_t temp = c3_coeff_scale * c3_coeff_offset << c3_coeff_scale_shift;
  std::transform(c1_coeff.begin(), c1_coeff.end(), c1_coeff.begin(),
                 [temp](int64_t c) { return c + temp; });

  // std::tuple<
  //     std::vector<int64_t>,
  //     int32_t,
  //     int64_t,
  //     int64_t,
  //     int64_t,
  //     int64_t
  // >

  // std::tuple<std::vector<int64_t>, int32_t, int64_t, int64_t, int64_t,
  // int64_t>

  int32_t C2 = c2_coeff_prime << matmul_shift;
  // int32_t C1 = c3_coeff_scale;
  // int64_t C0 = static_cast<int64_t>(c3_coeff_scale*c3_coeff_offset) <<
  // c3_coeff_scale_shift + c1_coeff;

  return std::make_tuple(c1_coeff, (int32_t)c3_coeff_scale, C2,
                         (int64_t)c3_coeff_scale_shift, (int64_t)shft_c2,
                         (int64_t)matmul_shift);
}

std::vector<int64_t> grpb_qgprb_vec64_fill(std::vector<int64_t> bias,
                                           int64_t qk_qdq_c0,
                                           int64_t smv_qdq_c0) {
  std::vector<int64_t> gprb_vec64(11, 0);

  for (int i = 0; i < 8; i++)
    gprb_vec64[i] = bias[i];

  gprb_vec64[9] = qk_qdq_c0;
  gprb_vec64[10] = smv_qdq_c0;

  return gprb_vec64;
}

std::vector<int32_t>
gprb_vec32_fill(const std::vector<int64_t>& coeff_grpb, float act_scale,
                int32_t act_zero_point, float wgt_scale, int32_t wgt_zero_point,
                const std::vector<uint16_t>& model_a, float model_a_scale,
                int32_t model_a_zp, uint16_t model_b, float model_b_scale,
                int32_t model_b_zp, uint16_t model_c, float model_c_scale,
                int32_t model_c_zp, int32_t is_grpb_int16) {

  std::vector<int32_t> gprb_vec32(32, 0);

  // const int qdq_c0_idx = 0;
  const int qdq_c1_idx = 2;
  const int qdq_c2_idx = 3;
  const int qdq_c3_idx = 4;
  const int qdq_Mv_idx = 5;
  const int qdq_Nv_idx = 6;
  const int qdq_SQb_idx = 7;
  const int qdq_Sout_idx = 8;
  const int qdq_Stdm_idx = 9;

  const int gprb_act_scale_idx = 10;
  const int gprb_act_zero_idx = 11;
  const int gprb_wgt_scale_idx = 12;
  const int gprb_wgt_zero_idx = 13;
  const int gprb_model_a_idx = 14;
  const int gprb_model_b_idx = 26;
  const int gprb_model_c_idx = 27;
  const int gprb_isint16_idx = 28;

  const int num_heads = 12;

  gprb_vec32[qdq_c1_idx] = coeff_grpb[0];
  gprb_vec32[qdq_c2_idx] = coeff_grpb[1];
  gprb_vec32[qdq_c3_idx] = 0;
  gprb_vec32[qdq_Mv_idx] = 32;
  gprb_vec32[qdq_Nv_idx] = 8;
  gprb_vec32[qdq_SQb_idx] = coeff_grpb[2];
  gprb_vec32[qdq_Sout_idx] = coeff_grpb[3];
  gprb_vec32[qdq_Stdm_idx] = coeff_grpb[4];

  gprb_vec32[gprb_act_scale_idx] =
      static_cast<int32_t>(float_to_bfloat16(act_scale));
  gprb_vec32[gprb_act_zero_idx] = act_zero_point;
  gprb_vec32[gprb_wgt_scale_idx] =
      static_cast<int32_t>(float_to_bfloat16(wgt_scale));
  gprb_vec32[gprb_wgt_zero_idx] = wgt_zero_point;
  gprb_vec32[gprb_isint16_idx] = is_grpb_int16;

  std::vector<float> model_a_bf(num_heads);
  for (size_t i = 0; i < num_heads; ++i) {
    model_a_bf[i] = dq<int32_t>(model_a[i], model_a_scale, model_a_zp);
  }

  for (int h = 0; h < num_heads; ++h) {
    gprb_vec32[gprb_model_a_idx + h] =
        static_cast<int32_t>(float_to_bfloat16(model_a_bf[h]));
  }

  gprb_vec32[gprb_model_b_idx] = static_cast<int32_t>(
      float_to_bfloat16(dq<int32_t>(model_b, model_b_scale, model_b_zp)));
  gprb_vec32[gprb_model_c_idx] = static_cast<int32_t>(
      float_to_bfloat16(dq<int32_t>(model_c, model_c_scale, model_c_zp)));

  return gprb_vec32;
}

std::tuple<int64_t, int32_t, int64_t, int32_t, int32_t, int32_t, int32_t>
qdq_act_matmul_uint8_uint8_cstm(float a_dq_xscale, int64_t a_dq_xzero_pt,
                                int64_t weights_in_ch, float w_dq_xscale,
                                int64_t w_dq_xzero_pt, float a_q_yscale,
                                int64_t a_q_yzero_pt) {
  // Ensure the zero points are of type int64_t
  int64_t a_dq_xzero_pt_i64 = static_cast<int64_t>(a_dq_xzero_pt);
  int64_t w_dq_xzero_pt_i64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_i64 = static_cast<int64_t>(a_q_yzero_pt);

  // Calculate the c2 coefficient
  float c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  int64_t c2_coeff_prime;
  int32_t shft_c2;

  std::tie(c2_coeff_prime, shft_c2) =
      find_closest_shifted_int32(c2_coeff, 8388607);

  c2_coeff_prime = static_cast<int64_t>(c2_coeff_prime);

  // Calculate the weight coefficient scale
  int64_t weight_coeff_scale = -c2_coeff_prime * a_dq_xzero_pt_i64;
  int32_t weight_coeff_scale_shift = 0;

  if (std::abs(weight_coeff_scale) > 2147483647) { // Max int32 number
    weight_coeff_scale_shift = static_cast<int32_t>(
        std::ceil(std::log2(std::abs(weight_coeff_scale))) - 31);
  } else {
    weight_coeff_scale_shift = 0;
  }

  weight_coeff_scale =
      static_cast<int32_t>(weight_coeff_scale >> weight_coeff_scale_shift);

  // Calculate c1 coefficient
  int64_t c1_coeff = a_q_yzero_pt_i64 << shft_c2;

  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset =
      static_cast<int32_t>(-a_dq_xzero_pt_i64 * num_weights_unrolled);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_dq_xzero_pt_i64;
  c1_coeff += c3_coeff_scale * static_cast<int64_t>(c3_coeff_offset);

  // Calculate the shift for c3 coefficient scale
  int32_t c3_coeff_scale_shift = 0;
  if (std::abs(c3_coeff_scale) > 2147483647) { // Max int32 number
    c3_coeff_scale_shift = static_cast<int32_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  } else {
    c3_coeff_scale_shift = 0;
  }

  c3_coeff_scale = static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);

  int32_t matmul_shift = 0;

  return std::make_tuple(c1_coeff,                                   // C0
                         static_cast<int32_t>(c3_coeff_scale),       // C1
                         static_cast<int64_t>(c2_coeff_prime),       // C2
                         static_cast<int32_t>(weight_coeff_scale),   // C3
                         static_cast<int32_t>(c3_coeff_scale_shift), // shift_qb
                         static_cast<int32_t>(shft_c2), // shift_out
                         matmul_shift                   // matmul_shift
  );
}

std::tuple<int64_t, int32_t, int64_t, int32_t, int32_t, int32_t, int32_t>
qdq_act_matmul_uint16_uint16_cstm(float a_dq_xscale, int64_t a_dq_xzero_pt,
                                  int64_t in_ch_dim, float w_dq_xscale,
                                  int64_t w_dq_xzero_pt, float a_q_yscale,
                                  int64_t a_q_yzero_pt) {

  if (in_ch_dim % 49 == 0) //  # hacky way for padding windowed attention
    in_ch_dim = (int64_t)(std::ceil((double)in_ch_dim / 49) * 64);

  auto matmul_shift = (int64_t)(std::min(
      std::max((int)std::ceil(std::log2(in_ch_dim)) + 1, 0), 15));
  // Calculate the c2 coefficient
  float c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;
  auto [c2_coeff_prime1, shft] = find_closest_shifted_int16(c2_coeff, 32767);
  int64_t c2_coeff_prime = (int64_t)c2_coeff_prime1;
  auto c3_coeff_scale = (int64_t)(-c2_coeff_prime * w_dq_xzero_pt);
  int64_t c3_coeff_scale_shift = 0;
  if (std::abs(c3_coeff_scale) > 2147483647) {
    c3_coeff_scale_shift = static_cast<int32_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
    throw std::runtime_error("Current AIE uint16A_uint16A qdq implementation "
                             "does not support ifm sum shift");
  }
  auto c3_coeff_scale1 =
      static_cast<int32_t>(c3_coeff_scale >> c3_coeff_scale_shift);
  auto C3 = static_cast<int64_t>(c2_coeff_prime << matmul_shift);
  auto C2 = static_cast<int32_t>(c3_coeff_scale1);
  auto C1 = static_cast<int32_t>((-a_dq_xzero_pt) * c2_coeff_prime);
  if (std::abs(C1) > 2147483647) {
    throw std::runtime_error("Current AIE uint16A_uint16A qdq implementation "
                             "does not support ifm sum shift");
  }
  int64_t C0 = static_cast<int64_t>(
      (a_q_yzero_pt << shft) +
      (a_dq_xzero_pt * (w_dq_xzero_pt) * (in_ch_dim)*c2_coeff_prime));

  auto right_shft_matmul = matmul_shift;

  auto shft_final = shft;

  return std::make_tuple(C0,                                         // C0
                         static_cast<int32_t>(C2),                   // C1
                         static_cast<int64_t>(C3),                   // C2
                         static_cast<int32_t>(C1),                   // C3
                         static_cast<int32_t>(c3_coeff_scale_shift), // shift_qb
                         static_cast<int32_t>(shft_final),       // shift_out
                         static_cast<int32_t>(right_shft_matmul) // matmul_shift
  );
}

std::tuple<int64_t, int32_t, int32_t, int64_t, int64_t, int64_t>
qdq_matmul_uint16_uint8_cstm(std::vector<uint8_t> weights, float a_dq_xscale,
                             int64_t a_dq_xzero_pt, float w_dq_xscale,
                             int64_t w_dq_xzero_pt, float a_q_yscale,
                             int64_t a_q_yzero_pt) {

  int64_t a_dq_xzero_pt_int64 = static_cast<int64_t>(a_dq_xzero_pt);
  // int64_t w_dq_xzero_pt_int64 = static_cast<int64_t>(w_dq_xzero_pt);
  int64_t a_q_yzero_pt_int64 = static_cast<int64_t>(a_q_yzero_pt);

  int64_t weights_in_ch = static_cast<int64_t>(weights.size());

  int64_t matmul_shift = std::min(
      std::max(25 + (int32_t)std::ceil(std::log2(weights_in_ch)) - 32, 0), 7);

  std::vector<int64_t> weights_int64;
  for (const auto& weight : weights) {
    weights_int64.push_back(static_cast<int64_t>(weight));
  }

  double c2_coeff = (a_dq_xscale * w_dq_xscale) / a_q_yscale;

  int64_t c2_coeff_prime;
  int64_t shft_c2;
  std::tie(c2_coeff_prime, shft_c2) =
      find_closest_shifted_int32(c2_coeff, 8388607);

  int64_t c2_coeff_prime_int64 = static_cast<int64_t>(c2_coeff_prime);

  int64_t c1_coeff =
      (-a_dq_xzero_pt_int64) * c2_coeff_prime_int64 *
          std::accumulate(weights_int64.begin(), weights_int64.end(), 0LL) +
      (a_q_yzero_pt_int64 << shft_c2);

  int64_t c1_coeff_int64 = static_cast<int64_t>(c1_coeff);
  int64_t num_weights_unrolled = weights_in_ch;
  int32_t c3_coeff_offset = -a_dq_xzero_pt_int64 * num_weights_unrolled;
  int64_t c3_coeff_scale = -c2_coeff_prime_int64 * w_dq_xzero_pt;

  int64_t c3_coeff_scale_shift = 0;
  if (std::abs(c3_coeff_scale) > 2147483647) {
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
  }

  c3_coeff_scale >>= c3_coeff_scale_shift;
  int32_t c3_coeff_scale_int32 = static_cast<int32_t>(c3_coeff_scale);

  int32_t C2 = c2_coeff_prime_int64 << matmul_shift;
  int32_t C1 = c3_coeff_scale_int32;
  int64_t C0 =
      c3_coeff_scale_int32 * (c3_coeff_offset << c3_coeff_scale_shift) +
      c1_coeff_int64;

  return std::make_tuple(C0, C1, C2, c3_coeff_scale_shift, shft_c2,
                         matmul_shift);
}

// std::vector<int32_t> mha_qdq_params_fill(
//     std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
//         qkt_qdq,
//     std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
//         smv_qdq,
//     std::tuple<int64_t, int64_t> sm_qdq_before,
//     std::tuple<int64_t, int64_t> sm_qdq_after,
//      is_qkt_smv_int16)

// {
//   int64_t qry_subv_rows = 32;
//   // int64_t qry_subv_cols = 96;
//   int64_t key_subv_rows = 64;
//   // int64_t key_subv_rows_int16 = 16;
//   // int64_t key_subv_cols = 96;
//   // int64_t val_subv_rows = 64;
//   int64_t val_subv_cols = 64;
//   // int64_t out_subv_rows = 32;
//   // int64_t out_subv_cols = 64;

//   std::vector<int32_t> qdq_params(96, 0);

//   // QKT
//   qdq_params[0] = std::get<0>(qkt_qdq);            // 64bit

//   qdq_params[(16 * 0) + 2] = std::get<1>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 3] = std::get<2>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 4] = std::get<3>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 5] = qry_subv_rows;        // 32 bit
//   qdq_params[(16 * 0) + 6] = key_subv_rows;        // 32 bit
//   qdq_params[(16 * 0) + 7] = std::get<4>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 8] = std::get<5>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 9] = std::get<6>(qkt_qdq); // 32 bit
//   qdq_params[(16 * 0) + 10] = is_qkt_smv_int16;    // 32 bit

//   // SM *V
//   qdq_params[8] = std::get<0>(smv_qdq);
//   qdq_params[(16 * 1) + 2] = std::get<1>(smv_qdq);
//   qdq_params[(16 * 1) + 3] = std::get<2>(smv_qdq);
//   qdq_params[(16 * 1) + 4] = std::get<3>(smv_qdq);
//   qdq_params[(16 * 1) + 5] = qry_subv_rows;
//   qdq_params[(16 * 1) + 6] = val_subv_cols;
//   qdq_params[(16 * 1) + 7] = std::get<4>(smv_qdq);
//   qdq_params[(16 * 1) + 8] = std::get<5>(smv_qdq);
//   qdq_params[(16 * 1) + 9] = std::get<6>(smv_qdq);
//   qdq_params[(16 * 1) + 10] = is_qkt_smv_int16;

//   // DQ before SM
//   qdq_params[(16 * 2) + 0] = std::get<1>(sm_qdq_before);
//   qdq_params[(16 * 2) + 1] = std::get<0>(sm_qdq_before);

//   // Q after SM
//   qdq_params[(16 * 3) + 0] = std::get<1>(sm_qdq_after);
//   qdq_params[(16 * 3) + 1] = std::get<0>(sm_qdq_after);
//   qdq_params[(16 * 3) + 2] = is_qkt_smv_int16;

//   return qdq_params;
// }

// This utility is for DeMHA QDQ params
// - Set 0 –
// 0, 1 --> DeQuant zp, scale of 2nd input of Add (position encoding addition
// position+context addition (pos_con_add)) which is again coming from another
// node (SkipAdd ) 2, 3 --> DeQuant zp, scale of attention mask input going to
// second Add 4, 5 --> DeQuant zp, scale of 1st input of Add3 which is coming
// from QKt
std::vector<int32_t> DeMHA_qdq_params_fill(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
        qkt_qdq,
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
        smv_qdq,
    std::tuple<int64_t, int64_t> sm_qdq_before,
    std::tuple<int64_t, int64_t> sm_qdq_after,
    std::tuple<int64_t, int64_t, int64_t, int64_t> qdq_pos_con_add_in,
    std::tuple<int64_t, int64_t, int64_t, int64_t> qdq_att_mask,
    int64_t is_qkt_smv_int16, int64_t smv_swap) {
  std::vector<int32_t> qdq_params(96, 0);

  // pos_con_add 2nd input qdq params ( input comming from pos encoding)
  qdq_params[(16 * 0) + 0] = std::get<3>(qdq_pos_con_add_in);
  qdq_params[(16 * 0) + 1] = std::get<2>(qdq_pos_con_add_in);

  // qdq params of input to add of attetion mask ( comming from attention mask)
  qdq_params[(16 * 0) + 2] = std::get<1>(qdq_pos_con_add_in);
  qdq_params[(16 * 0) + 3] = std::get<0>(qdq_pos_con_add_in);

  // qdq params of pos_con_add 1st input comming from div
  // qdq_params[(16 * 0) + 4] = std::get<3>(qdq_pos_con_add_in);
  // qdq_params[(16 * 0) + 5] = std::get<2>(qdq_pos_con_add_in);

  // qdq params of output to add of attetion mask
  qdq_params[(16 * 1) + 0] = std::get<1>(qdq_att_mask);
  qdq_params[(16 * 1) + 1] = std::get<0>(qdq_att_mask);

  // QKT
  reinterpret_cast<int64_t*>(qdq_params.data())[16] = std::get<0>(qkt_qdq);
  qdq_params[(16 * 2) + 2] = std::get<1>(qkt_qdq);
  qdq_params[(16 * 2) + 3] = std::get<2>(qkt_qdq);
  qdq_params[(16 * 2) + 4] = std::get<3>(qkt_qdq);
  qdq_params[(16 * 2) + 5] = 16;
  qdq_params[(16 * 2) + 6] = 64;
  qdq_params[(16 * 2) + 7] = std::get<4>(qkt_qdq);
  qdq_params[(16 * 2) + 8] = std::get<5>(qkt_qdq);
  qdq_params[(16 * 2) + 9] = std::get<6>(qkt_qdq);

  // SM *V
  reinterpret_cast<int64_t*>(qdq_params.data())[24] = std::get<0>(smv_qdq);
  if (smv_swap != 0) {
    qdq_params[(16 * 3) + 2] = std::get<3>(smv_qdq); // swapping
    qdq_params[(16 * 3) + 4] = std::get<1>(smv_qdq);
  } else {
    qdq_params[(16 * 3) + 2] = std::get<1>(smv_qdq);
    qdq_params[(16 * 3) + 4] = std::get<3>(smv_qdq);
  }

  qdq_params[(16 * 3) + 3] = std::get<2>(smv_qdq);
  qdq_params[(16 * 3) + 5] = 16;
  qdq_params[(16 * 3) + 6] = 64;
  qdq_params[(16 * 3) + 7] = std::get<4>(smv_qdq);
  qdq_params[(16 * 3) + 8] = std::get<5>(smv_qdq);
  qdq_params[(16 * 3) + 9] = std::get<6>(smv_qdq);

  // DQ before SM
  qdq_params[(16 * 4) + 0] = std::get<1>(sm_qdq_before);
  qdq_params[(16 * 4) + 1] = std::get<0>(sm_qdq_before);

  // Q after SM
  qdq_params[(16 * 5) + 0] = std::get<1>(sm_qdq_after);
  qdq_params[(16 * 5) + 1] = std::get<0>(sm_qdq_after);

  return qdq_params;
}

std::vector<int32_t> mha_channel_qdq_params_fill(
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
        qkt_qdq,
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
        smv_qdq,
    std::tuple<int64_t, int64_t> sm_qdq_before,
    std::tuple<int64_t, int64_t> sm_qdq_after,
    std::tuple<int64_t, int64_t> qdq_mul_in,
    std::tuple<int64_t, int64_t> qdq_mul_out, int64_t is_qkt_smv_int16,
    int64_t smv_swap) {
  std::vector<int32_t> qdq_params(96, 0);

  // int64_t qry_subv_rows = 32;
  // int64_t qry_subv_cols = 96;
  // int64_t key_subv_rows = 64;
  // int64_t key_subv_rows_int16 = 16;
  // int64_t key_subv_cols = 96;
  // int64_t val_subv_rows = 64;
  // int64_t val_subv_cols = 64;
  // int64_t out_subv_rows = 32;
  // int64_t out_subv_cols = 64;

  qdq_params[(16 * 0) + 0] = std::get<1>(qdq_mul_in);
  qdq_params[(16 * 0) + 1] = std::get<0>(qdq_mul_in);

  qdq_params[(16 * 1) + 0] = std::get<1>(qdq_mul_out);
  qdq_params[(16 * 1) + 1] = std::get<0>(qdq_mul_out);

  // QKT
  reinterpret_cast<int64_t*>(qdq_params.data())[16] = std::get<0>(qkt_qdq);
  qdq_params[(16 * 2) + 2] = std::get<1>(qkt_qdq);
  qdq_params[(16 * 2) + 3] = std::get<2>(qkt_qdq);
  qdq_params[(16 * 2) + 4] = std::get<3>(qkt_qdq);
  qdq_params[(16 * 2) + 5] = 49;
  qdq_params[(16 * 2) + 6] = 49;
  qdq_params[(16 * 2) + 7] = std::get<4>(qkt_qdq);
  qdq_params[(16 * 2) + 8] = std::get<5>(qkt_qdq);
  qdq_params[(16 * 2) + 9] = std::get<6>(qkt_qdq);

  // SM *V
  reinterpret_cast<int64_t*>(qdq_params.data())[24] = std::get<0>(smv_qdq);
  if (smv_swap != 0) {
    qdq_params[(16 * 3) + 2] = std::get<3>(smv_qdq); // swapping
    qdq_params[(16 * 3) + 4] = std::get<1>(smv_qdq);
  } else {
    qdq_params[(16 * 3) + 2] = std::get<1>(smv_qdq);
    qdq_params[(16 * 3) + 4] = std::get<3>(smv_qdq);
  }

  qdq_params[(16 * 3) + 3] = std::get<2>(smv_qdq);
  qdq_params[(16 * 3) + 5] = 49;
  qdq_params[(16 * 3) + 6] = 32;
  qdq_params[(16 * 3) + 7] = std::get<4>(smv_qdq);
  qdq_params[(16 * 3) + 8] = std::get<5>(smv_qdq);
  qdq_params[(16 * 3) + 9] = std::get<6>(smv_qdq);

  // DQ before SM
  qdq_params[(16 * 4) + 0] = std::get<1>(sm_qdq_before);
  qdq_params[(16 * 4) + 1] = std::get<0>(sm_qdq_before);

  // Q after SM
  qdq_params[(16 * 5) + 0] = std::get<1>(sm_qdq_after);
  qdq_params[(16 * 5) + 1] = std::get<0>(sm_qdq_after);

  return qdq_params;
}

std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t>
dq_uint16A_uint8W_conv_q_param_gen(float in_s, uint16_t in_zp,
                                   gsl::span<const uint8_t> w, float w_s,
                                   uint8_t w_zp,
                                   const std::vector<int64_t>& w_shape,
                                   gsl::span<const int32_t> b, float b_s,
                                   int32_t b_zp, float o_s, uint16_t o_zp) {
  // not a must, may need to relax
  CHECK(w_shape.size() == 4);
  int64_t num_weights_unrolled = w_shape[1] * w_shape[2] * w_shape[3];
  int64_t num_weights_unrolled_origin = num_weights_unrolled;
  // hardcode for m3uec? may need to change later
  int64_t num_weight_zp_padded = 0;
  if (num_weights_unrolled == 3 * 7 * 7) {
    int64_t num_weights_unrolled_new = 4 * 7 * 8;
    num_weight_zp_padded = num_weights_unrolled_new - num_weights_unrolled;
    num_weights_unrolled = num_weights_unrolled_new;
  }
  int64_t conv_shift = (int64_t)(std::min(
      std::max((int)std::ceil(std::log2(num_weights_unrolled)) - 7, 0), 7));
  float c2_coeff = (in_s * w_s) / o_s;

  int64_t c2_coeff_prime;
  int32_t shft_c2;
  std::tie(c2_coeff_prime, shft_c2) =
      find_closest_shifted_int32(c2_coeff, 8388607);

  int64_t o_zp_int64 = static_cast<int64_t>(o_zp);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp;
  int32_t c3_coeff_offset = -in_zp * num_weights_unrolled;
  int64_t c3_coeff_scale_shift = 0;
  CHECK(std::abs(c3_coeff_scale) <= std::numeric_limits<int32_t>::max())
      << "Current AIE uint16A_uint8W qdq implementation does not support ifm "
         "sum shift";
  int32_t C2 = static_cast<int32_t>(c2_coeff_prime << conv_shift);
  int32_t C1 = static_cast<int32_t>(c3_coeff_scale);

  std::vector<int64_t> C0(w_shape[0], 0);
  for (size_t i = 0; i < C0.size(); ++i) {
    int range_start = i * num_weights_unrolled_origin;
    int range_end = (i + 1) * num_weights_unrolled_origin;
    C0[i] = (-in_zp * c2_coeff_prime) *
                (std::accumulate(w.begin() + range_start, w.begin() + range_end,
                                 0LL) +
                 num_weight_zp_padded * w_zp) +
            (b.size() > 0 ? b[i] * c2_coeff_prime : 0) +
            (o_zp_int64 << shft_c2) +
            c3_coeff_scale * (c3_coeff_offset << c3_coeff_scale_shift);
  }

  return {C0, C1, C2, conv_shift, (int64_t)shft_c2};
}
std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t>
dq_uint16A_uint16W_conv_q_param_gen(float in_s, uint16_t in_zp,
                                    gsl::span<const uint16_t> w, float w_s,
                                    uint16_t w_zp,
                                    const std::vector<int64_t>& w_shape,
                                    gsl::span<const int32_t> b, float b_s,
                                    int32_t b_zp, float o_s, uint16_t o_zp) {
  // not a must, may need to relax
  CHECK(w_shape.size() == 4);
  int64_t num_weights_unrolled = w_shape[1] * w_shape[2] * w_shape[3];
  int64_t num_weights_unrolled_origin = num_weights_unrolled;
  int64_t conv_shift = (int64_t)(std::min(
      std::max((int)std::ceil(std::log2(num_weights_unrolled)) + 1, 0), 15));
  float c2_coeff = (float)((in_s * w_s) / o_s);

  int64_t c2_coeff_prime;
  int32_t shft_c2;
  std::tie(c2_coeff_prime, shft_c2) =
      find_closest_shifted_int16(c2_coeff, 32767);

  int64_t o_zp_int64 = static_cast<int64_t>(o_zp);
  int64_t c3_coeff_scale = -c2_coeff_prime * w_zp;
  int32_t c3_coeff_offset = -in_zp * num_weights_unrolled;
  int64_t c3_coeff_scale_shift = 0;

  if (std::abs(c3_coeff_scale) > 2147483647) {
    c3_coeff_scale_shift = static_cast<int64_t>(
        std::ceil(std::log2(std::abs(c3_coeff_scale))) - 31);
    std::cout << "Current AIE uint16A_uint16W qdq implementation does not "
                 "support ifm "
                 "sum shift"
              << std::endl;
  }

  c3_coeff_scale >>= c3_coeff_scale_shift;
  int32_t c3_coeff_scale_int32 = static_cast<int32_t>(c3_coeff_scale);

  int32_t C2 = static_cast<int32_t>(c2_coeff_prime << conv_shift);
  int32_t C1 = c3_coeff_scale_int32;

  std::vector<int64_t> C0(w_shape[0], 0);
  for (size_t i = 0; i < C0.size(); ++i) {
    int range_start = i * num_weights_unrolled_origin;
    int range_end = (i + 1) * num_weights_unrolled_origin;
    C0[i] = (-in_zp * c2_coeff_prime) *
                (std::accumulate(w.begin() + range_start, w.begin() + range_end,
                                 0LL)) +
            (b.size() > 0 ? b[i] * c2_coeff_prime : 0) +
            (o_zp_int64 << shft_c2) +
            c3_coeff_scale * (c3_coeff_offset << c3_coeff_scale_shift);
  }

  return {C0, C1, C2, conv_shift, (int64_t)shft_c2};
}

std::vector<int32_t>
mha_qdq_params_fill(const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t>& coeff_qkt,
                    const std::tuple<int64_t, int32_t, int64_t, int32_t,
                                     int32_t, int32_t, int32_t>& coeff_smv,
                    const std::tuple<uint16_t, int>& sm_qdq_before,
                    const std::tuple<uint16_t, int>& sm_qdq_after,
                    int32_t is_qkt_smv_int16) {
  std::vector<int32_t> qdq_params(96, 0);

  constexpr int32_t qry_subv_rows = 32;
  //   constexpr int32_t qry_subv_cols = 96;
  constexpr int32_t key_subv_rows = 64;
  //   constexpr int32_t key_subv_rows_int16 = 16;
  //   constexpr int32_t key_subv_cols = 96;
  //   constexpr int32_t val_subv_rows = 64;
  constexpr int32_t val_subv_cols = 64;
  //   constexpr int32_t out_subv_rows = 32;
  //   constexpr int32_t out_subv_cols = 64;

  // QKT
  reinterpret_cast<int64_t*>(qdq_params.data())[0] = std::get<0>(coeff_qkt);
  qdq_params[(16 * 0) + 2] = std::get<1>(coeff_qkt);
  qdq_params[(16 * 0) + 3] = static_cast<int32_t>(std::get<2>(coeff_qkt));
  qdq_params[(16 * 0) + 4] = std::get<3>(coeff_qkt);
  qdq_params[(16 * 0) + 5] = qry_subv_rows;
  qdq_params[(16 * 0) + 6] = key_subv_rows;
  qdq_params[(16 * 0) + 7] = std::get<4>(coeff_qkt);
  qdq_params[(16 * 0) + 8] = std::get<5>(coeff_qkt);
  qdq_params[(16 * 0) + 9] = std::get<6>(coeff_qkt);
  qdq_params[(16 * 0) + 10] = is_qkt_smv_int16;

  // SM *V
  reinterpret_cast<int64_t*>(qdq_params.data())[8] = std::get<0>(coeff_smv);
  qdq_params[(16 * 1) + 2] = std::get<1>(coeff_smv);
  qdq_params[(16 * 1) + 3] = std::get<2>(coeff_smv);
  qdq_params[(16 * 1) + 4] = std::get<3>(coeff_smv);
  qdq_params[(16 * 1) + 5] = qry_subv_rows;
  qdq_params[(16 * 1) + 6] = val_subv_cols;
  qdq_params[(16 * 1) + 7] = std::get<4>(coeff_smv);
  qdq_params[(16 * 1) + 8] = std::get<5>(coeff_smv);
  qdq_params[(16 * 1) + 9] = std::get<6>(coeff_smv);
  qdq_params[(16 * 1) + 10] = is_qkt_smv_int16;

  // DQ before SM
  qdq_params[(16 * 2) + 0] = std::get<1>(sm_qdq_before);
  qdq_params[(16 * 2) + 1] = std::get<0>(sm_qdq_before);

  // Q after SM
  qdq_params[(16 * 3) + 0] = std::get<1>(sm_qdq_after);
  qdq_params[(16 * 3) + 1] = std::get<0>(sm_qdq_after);
  qdq_params[(16 * 3) + 2] = is_qkt_smv_int16;

  return qdq_params;
}

static std::uint32_t convert_float_to_qint(float in_f) {
  std::uint32_t ret{0};
  std::memcpy(&ret, &in_f, sizeof(in_f));
  ret &= 0x7fffffff;
  return ret;
}

std::vector<uint8_t>
mladfelwmul_qdq_param_gen(float ifm1_scale, float ifm2_scale, float ofm_scale,
                          uint16_t ifm1_zp, uint16_t ifm2_zp, uint16_t ofm_zp,
                          int64_t tensor_sz) {
  float C0 = ifm1_scale * ifm2_scale / ofm_scale;
  double C0_d = double(ifm1_scale) * ifm2_scale / ofm_scale;
  uint32_t c0_qint = convert_float_to_qint(C0);
  uint8_t c0_shift =
      127 - (((c0_qint >> 23) & 255) + 1) + (8 * sizeof(int) - 2);
  // use C0_d to align with python accuracy
  int32_t coeff0 = int32_t(C0_d * std::pow(2, c0_shift));
  float C1 = C0 * ifm1_zp * ifm2_zp + ofm_zp;
  uint32_t c1_qint = convert_float_to_qint(C1);
  uint8_t c1_shift =
      127 - (((c1_qint >> 23) & 255) + 1) + (8 * sizeof(int) - 2);
  int32_t coeff1 = int32_t(C1 * std::pow(2, c1_shift));
  std::vector<uint8_t> qdq_param(64, 0);
  *(int32_t*)(&(qdq_param[0])) = 4096;
  *(int32_t*)(&(qdq_param[4])) = coeff0;
  *(int32_t*)(&(qdq_param[8])) = coeff1;
  qdq_param[12] = ifm1_zp & 0xFF;
  qdq_param[13] = (ifm1_zp >> 8) & 0xFF;
  qdq_param[14] = ifm2_zp & 0xFF;
  qdq_param[15] = (ifm2_zp >> 8) & 0xFF;
  qdq_param[16] = c0_shift;
  qdq_param[17] = c1_shift;
  qdq_param[20] = (tensor_sz / (4096 * 8)) & 0xFF;
  qdq_param[21] = ((tensor_sz / (4096 * 8)) >> 8) & 0xFF;
  qdq_param[62] = 133;
  qdq_param[63] = 201;
  return qdq_param;
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif
} // namespace vaip::dd::qmatmulcalc