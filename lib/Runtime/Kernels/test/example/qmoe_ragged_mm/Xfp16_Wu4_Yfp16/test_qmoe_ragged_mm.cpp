/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the device-driven ragged W4A16 MatMul primitive.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t error_ = (call);                                           \
    if (error_ != hipSuccess) {                                                 \
      std::cerr << "HIP failure " << hipGetErrorString(error_) << " at "       \
                << __FILE__ << ':' << __LINE__ << '\n';                        \
      std::exit(2);                                                             \
    }                                                                          \
  } while (false)

constexpr int kMaxRowsPerGroup = 64;
constexpr int kTaskGemv = 0;
constexpr int kTaskWmma16 = 1;
constexpr int kTaskWmma64 = 2;
constexpr int kGuardElements = 32;
constexpr uint16_t kGuardBits = 0x3555;

struct Shape {
  const char* name;
  int rows;
  int experts;
  int n;
  int k;
  int block_size;
  int min_coverage;
};

// Coverage tiers thin only this continuous-shape list. Every selected shape
// still crosses all routing, input-layout, and bias categories below.
constexpr Shape kShapes[] = {
    {"tiny", 20, 4, 17, 32, 32, 1},
    {"multi_group", 17, 6, 33, 64, 32, 2},
    {"tail_k_and_n", 23, 8, 65, 48, 32, 3},
    {"wider", 19, 5, 64, 96, 32, 3},
};

constexpr Shape kAdaptiveShapes[] = {
    {"task_m1", 1, 1, 17, 32, 32, 3},
    {"task_m2", 2, 1, 17, 32, 32, 3},
    {"task_m7", 7, 1, 17, 32, 32, 3},
    {"task_m8", 8, 1, 17, 32, 32, 3},
    {"task_m15", 15, 1, 17, 32, 32, 3},
    {"task_m16", 16, 1, 17, 32, 32, 3},
    {"task_m31", 31, 1, 17, 32, 32, 3},
    {"task_m32", 32, 1, 17, 32, 32, 3},
    {"task_m63", 63, 1, 17, 32, 32, 3},
    {"task_m64", 64, 1, 17, 32, 32, 3},
    {"task_m65", 65, 1, 17, 32, 32, 3},
};

enum class Routing { Balanced, EmptyExperts, Skewed };

struct QueuePolicyExpectation {
  const char* name;
  uint32_t chunk;
  bool grid_stride;
};

struct TaskPolicyExpectation {
  const char* name;
  int id;
  int gemv_max_rows;
  int wmma16_max_rows;
};

TaskPolicyExpectation expected_task_policy(bool fc1) {
  const char* policy = std::getenv(
      fc1 ? "HIPDNN_QMOE_RAGGED_FC1_TASK_POLICY"
          : "HIPDNN_QMOE_RAGGED_FC2_TASK_POLICY");
  if (!policy || std::strcmp(policy, "p0-current") == 0) {
    return {"p0-current", 0, 7, 31};
  }
  if (std::strcmp(policy, "p1-early-wmma16") == 0) {
    return {"p1-early-wmma16", 1, 3, 31};
  }
  if (std::strcmp(policy, "p2-early-wmma64") == 0) {
    return {"p2-early-wmma64", 2, 7, 15};
  }
  if (std::strcmp(policy, "p3-early-both") == 0) {
    return {"p3-early-both", 3, 3, 15};
  }
  return {"p0-current", 0, 7, 31};
}

QueuePolicyExpectation expected_queue_policy() {
  const char* policy = std::getenv("HIPDNN_QMOE_RAGGED_QUEUE_POLICY");
  if (!policy || std::strcmp(policy, "atomic1") == 0) {
    return {"atomic1", 1, false};
  }
  if (std::strcmp(policy, "chunk2") == 0) {
    return {"chunk2", 2, false};
  }
  if (std::strcmp(policy, "chunk4") == 0) {
    return {"chunk4", 4, false};
  }
  if (std::strcmp(policy, "chunk8") == 0) {
    return {"chunk8", 8, false};
  }
  if (std::strcmp(policy, "grid-stride") == 0) {
    return {"grid-stride", 0, true};
  }
  return {"atomic1", 1, false};
}

const char* routing_name(Routing routing) {
  switch (routing) {
  case Routing::Balanced:
    return "balanced";
  case Routing::EmptyExperts:
    return "empty";
  case Routing::Skewed:
    return "skewed";
  }
  return "unknown";
}

float half_to_float(__half value) {
  uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x3ffu;
  uint32_t result_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result_bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffu;
      result_bits = sign | ((127u - 15u + 1u - shift) << 23) |
                    (mantissa << 13);
    }
  } else if (exponent == 31) {
    result_bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result_bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &result_bits, sizeof(result));
  return result;
}

__half float_to_half(float value) { return __float2half(value); }

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count_ != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }
  }
  ~DeviceBuffer() {
    if (data_) {
      hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }
  const T* get() const { return data_; }
  size_t size() const { return count_; }

private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source) {
  if (destination.size() != source.size()) {
    std::cerr << "internal test error: upload size mismatch\n";
    std::exit(2);
  }
  HIP_CHECK(hipMemcpy(destination.get(), source.data(),
                      source.size() * sizeof(T), hipMemcpyHostToDevice));
}

std::vector<int> make_counts(const Shape& shape, Routing routing) {
  std::vector<int> counts(shape.experts, 0);
  if (routing == Routing::Balanced) {
    for (int row = 0; row < shape.rows; ++row) {
      ++counts[row % shape.experts];
    }
  } else if (routing == Routing::EmptyExperts) {
    const int first = shape.experts > 1 ? 1 : 0;
    const int second = shape.experts > 2 ? shape.experts - 1 : first;
    counts[first] = shape.rows / 2;
    counts[second] += shape.rows - counts[first];
  } else {
    counts[0] = shape.rows - (shape.experts - 1);
    for (int expert = 1; expert < shape.experts; ++expert) {
      counts[expert] = 1;
    }
  }
  return counts;
}

bool build_groups(const Shape& shape, const std::vector<int>& counts,
                  std::vector<int32_t>& groups,
                  std::vector<int>& expert_for_row) {
  groups.clear();
  expert_for_row.assign(shape.rows, -1);
  std::vector<int> visits(shape.rows, 0);
  int row_begin = 0;
  for (int expert = 0; expert < shape.experts; ++expert) {
    const int expert_begin = row_begin;
    const int expert_end = expert_begin + counts[expert];
    for (int row = expert_begin; row < expert_end; ++row) {
      expert_for_row[row] = expert;
    }
    for (int group_begin = expert_begin; group_begin < expert_end;
         group_begin += kMaxRowsPerGroup) {
      const int group_rows =
          std::min(kMaxRowsPerGroup, expert_end - group_begin);
      groups.push_back(expert);
      groups.push_back(group_begin);
      groups.push_back(group_rows);
      groups.push_back(0);
      for (int row = group_begin; row < group_begin + group_rows; ++row) {
        ++visits[row];
      }
    }
    row_begin = expert_end;
  }
  if (row_begin != shape.rows) {
    return false;
  }
  for (int row = 0; row < shape.rows; ++row) {
    if (visits[row] != 1 || expert_for_row[row] < 0) {
      return false;
    }
  }
  for (int expert = 0; expert < shape.experts; ++expert) {
    if (counts[expert] != 0) {
      continue;
    }
    for (size_t group = 0; group < groups.size() / 4; ++group) {
      if (groups[group * 4] == expert) {
        return false;
      }
    }
  }
  return true;
}

uint8_t quant_value(int expert, int n, int k) {
  return static_cast<uint8_t>((expert * 11 + n * 7 + k * 5 + 3) & 0x0f);
}

float input_value(int row, int k) {
  return static_cast<float>(((row * 13 + k * 3 + 5) % 23) - 11) / 16.0f;
}

float scale_value(int expert, int n, int group) {
  return 0.015625f * static_cast<float>(1 + ((expert + n + group) % 4));
}

float bias_value(int expert, int n) {
  return static_cast<float>(((expert * 5 + n * 3) % 11) - 5) / 32.0f;
}

struct Metrics {
  double rel_l2 = std::numeric_limits<double>::infinity();
  double max_abs = std::numeric_limits<double>::infinity();
  bool finite = false;
};

Metrics compare(const std::vector<__half>& actual,
                const std::vector<__half>& expected) {
  double error_sq = 0.0;
  double reference_sq = 0.0;
  double max_abs = 0.0;
  bool finite = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double got = half_to_float(actual[i]);
    const double want = half_to_float(expected[i]);
    finite = finite && std::isfinite(got) && std::isfinite(want);
    const double delta = got - want;
    error_sq += delta * delta;
    reference_sq += want * want;
    max_abs = std::max(max_abs, std::abs(delta));
  }
  return {std::sqrt(error_sq / std::max(reference_sq, 1.0e-20)), max_abs,
          finite};
}

bool run_case(const Shape& shape, Routing routing, bool gather, bool use_bias,
              bool use_zero_points, hipStream_t stream) {
  const std::vector<int> counts = make_counts(shape, routing);
  std::vector<int32_t> groups;
  std::vector<int> expert_for_row;
  if (!build_groups(shape, counts, groups, expert_for_row)) {
    std::cerr << "FAIL descriptor coverage shape=" << shape.name
              << " routing=" << routing_name(routing) << '\n';
    return false;
  }

  const int input_rows = gather ? std::max(2, shape.rows / 2) : shape.rows;
  std::vector<__half> input(static_cast<size_t>(input_rows) * shape.k);
  for (int row = 0; row < input_rows; ++row) {
    for (int k = 0; k < shape.k; ++k) {
      input[static_cast<size_t>(row) * shape.k + k] =
          float_to_half(input_value(row, k));
    }
  }

  std::vector<int32_t> input_row_ids(shape.rows);
  for (int row = 0; row < shape.rows; ++row) {
    input_row_ids[row] = gather ? ((row * 3 + 1) % input_rows) : row;
  }

  const int k_blocks = (shape.k + shape.block_size - 1) / shape.block_size;
  const int blob_size = shape.block_size / 2;
  const int weight_row_bytes = k_blocks * blob_size;
  std::vector<uint8_t> weights(
      static_cast<size_t>(shape.experts) * shape.n * weight_row_bytes, 0);
  std::vector<__half> scales(
      static_cast<size_t>(shape.experts) * shape.n * k_blocks);
  std::vector<uint8_t> zero_points(
      static_cast<size_t>(shape.experts) * shape.n * k_blocks);
  std::vector<__half> bias(static_cast<size_t>(shape.experts) * shape.n);

  for (int expert = 0; expert < shape.experts; ++expert) {
    for (int n = 0; n < shape.n; ++n) {
      const size_t expert_n = static_cast<size_t>(expert) * shape.n + n;
      for (int group = 0; group < k_blocks; ++group) {
        scales[expert_n * k_blocks + group] =
            float_to_half(scale_value(expert, n, group));
        zero_points[expert_n * k_blocks + group] = static_cast<uint8_t>(
            3 + ((expert * 5 + n * 3 + group) % 10));
        for (int offset = 0; offset < shape.block_size; offset += 2) {
          const int k0 = group * shape.block_size + offset;
          const uint8_t low = k0 < shape.k ? quant_value(expert, n, k0) : 8;
          const uint8_t high = k0 + 1 < shape.k
                                   ? quant_value(expert, n, k0 + 1)
                                   : 8;
          weights[expert_n * weight_row_bytes + group * blob_size + offset / 2] =
              static_cast<uint8_t>(low | (high << 4));
        }
      }
      bias[expert_n] = float_to_half(bias_value(expert, n));
    }
  }

  std::vector<__half> expected(static_cast<size_t>(shape.rows) * shape.n);
  for (int row = 0; row < shape.rows; ++row) {
    const int expert = expert_for_row[row];
    const int source_row = input_row_ids[row];
    for (int n = 0; n < shape.n; ++n) {
      float accumulator = 0.0f;
      const size_t expert_n = static_cast<size_t>(expert) * shape.n + n;
      for (int k = 0; k < shape.k; ++k) {
        const int group = k / shape.block_size;
        const int offset = k - group * shape.block_size;
        const uint8_t packed = weights[expert_n * weight_row_bytes +
                                       group * blob_size + offset / 2];
        const float q = (offset & 1) ? static_cast<float>(packed >> 4)
                                     : static_cast<float>(packed & 0x0f);
        const float zero_point =
            use_zero_points
                ? static_cast<float>(zero_points[expert_n * k_blocks + group])
                : 8.0f;
        const float weight =
            (q - zero_point) *
            half_to_float(scales[expert_n * k_blocks + group]);
        accumulator +=
            half_to_float(input[static_cast<size_t>(source_row) * shape.k + k]) *
            weight;
      }
      if (use_bias) {
        accumulator += half_to_float(bias[expert_n]);
      }
      expected[static_cast<size_t>(row) * shape.n + n] =
          float_to_half(accumulator);
    }
  }

  DeviceBuffer<__half> d_input(input.size());
  DeviceBuffer<int32_t> d_input_row_ids(input_row_ids.size());
  DeviceBuffer<int32_t> d_groups(groups.size());
  DeviceBuffer<int32_t> d_group_count(1);
  DeviceBuffer<uint32_t> d_queue_head(1);
  DeviceBuffer<uint8_t> d_weights(weights.size());
  DeviceBuffer<__half> d_scales(scales.size());
  DeviceBuffer<uint8_t> d_zero_points(zero_points.size());
  DeviceBuffer<__half> d_bias(bias.size());
  const size_t output_elements = static_cast<size_t>(shape.rows) * shape.n;
  DeviceBuffer<__half> d_guarded_output(output_elements + 2 * kGuardElements);

  upload(d_input, input);
  upload(d_input_row_ids, input_row_ids);
  upload(d_groups, groups);
  const std::vector<int32_t> group_count = {
      static_cast<int32_t>(groups.size() / 4)};
  upload(d_group_count, group_count);
  upload(d_weights, weights);
  upload(d_scales, scales);
  upload(d_zero_points, zero_points);
  upload(d_bias, bias);
  const uint32_t zero = 0;
  HIP_CHECK(hipMemcpy(d_queue_head.get(), &zero, sizeof(zero),
                      hipMemcpyHostToDevice));

  std::vector<uint16_t> guard_bits(output_elements + 2 * kGuardElements,
                                   kGuardBits);
  HIP_CHECK(hipMemcpy(d_guarded_output.get(), guard_bits.data(),
                      guard_bits.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));

  __half* output = d_guarded_output.get() + kGuardElements;
  const int status = hip_qmoe_ragged_matmul_nbits(
      reinterpret_cast<void*>(stream), d_input.get(),
      gather ? static_cast<const void*>(d_input_row_ids.get()) : nullptr,
      d_groups.get(), d_group_count.get(), d_queue_head.get(), d_weights.get(),
      d_scales.get(),
      use_zero_points ? static_cast<const void*>(d_zero_points.get()) : nullptr,
      use_bias ? static_cast<const void*>(d_bias.get()) : nullptr,
      output, shape.rows, shape.experts, shape.n, shape.k, shape.block_size,
      sizeof(__half));
  if (status != static_cast<int>(hipSuccess)) {
    std::cerr << "FAIL launch status=" << status << " shape=" << shape.name
              << '\n';
    return false;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<__half> actual(output_elements);
  HIP_CHECK(hipMemcpy(actual.data(), output, output_elements * sizeof(__half),
                      hipMemcpyDeviceToHost));
  std::vector<uint16_t> guarded_result(guard_bits.size());
  HIP_CHECK(hipMemcpy(guarded_result.data(), d_guarded_output.get(),
                      guarded_result.size() * sizeof(uint16_t),
                      hipMemcpyDeviceToHost));

  bool guards_ok = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards_ok = guards_ok && guarded_result[i] == kGuardBits;
    guards_ok = guards_ok &&
                guarded_result[kGuardElements + output_elements + i] ==
                    kGuardBits;
  }

  const Metrics metrics = compare(actual, expected);
  const bool pass = guards_ok && metrics.finite && metrics.max_abs <= 0.02 &&
                    metrics.rel_l2 <= 0.01;
  std::cout << (pass ? "PASS" : "FAIL") << " relL2=" << metrics.rel_l2
            << " n=" << output_elements << " shape=" << shape.name
            << " routing=" << routing_name(routing)
            << " input=" << (gather ? "gather" : "contiguous")
            << " bias=" << (use_bias ? "on" : "off")
            << " zp=" << (use_zero_points ? "on" : "off")
            << " maxAbs=" << metrics.max_abs
            << " guards=" << (guards_ok ? "PASS" : "FAIL") << '\n';
  return pass;
}

bool run_bucket_descriptor_case(hipStream_t stream) {
  constexpr int counts[] = {1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 65};
  constexpr int experts = sizeof(counts) / sizeof(counts[0]);
  int pairs = 0;
  for (int count : counts) pairs += count;

  std::vector<int32_t> expert_indices;
  std::vector<__half> expert_weights;
  std::vector<int32_t> expected_counts(experts);
  std::vector<int32_t> expected_offsets(experts + 1, 0);
  std::vector<int32_t> expected_groups;
  for (int expert = 0; expert < experts; ++expert) {
    expected_counts[expert] = counts[expert];
    expected_offsets[expert + 1] = expected_offsets[expert] + counts[expert];
    for (int i = 0; i < counts[expert]; ++i) {
      expert_indices.push_back(expert);
      expert_weights.push_back(float_to_half(1.0f));
    }
    for (int row = expected_offsets[expert]; row < expected_offsets[expert + 1];
         row += kMaxRowsPerGroup) {
      const int rows = std::min(kMaxRowsPerGroup,
                                expected_offsets[expert + 1] - row);
      expected_groups.push_back(expert);
      expected_groups.push_back(row);
      expected_groups.push_back(rows);
      expected_groups.push_back(0);
    }
  }

  DeviceBuffer<int32_t> d_indices(expert_indices.size());
  DeviceBuffer<__half> d_weights(expert_weights.size());
  DeviceBuffer<int32_t> d_counts(experts);
  DeviceBuffer<int32_t> d_offsets(experts + 1);
  DeviceBuffer<int32_t> d_sorted_ids(pairs);
  DeviceBuffer<__half> d_sorted_weights(pairs);
  DeviceBuffer<int32_t> d_pair_to_sorted(pairs);
  DeviceBuffer<int32_t> d_groups(static_cast<size_t>(pairs) * 4);
  DeviceBuffer<int32_t> d_group_count(1);
  DeviceBuffer<uint32_t> d_fc1_queue(1);
  DeviceBuffer<uint32_t> d_fc2_queue(1);
  upload(d_indices, expert_indices);
  upload(d_weights, expert_weights);
  const uint32_t dirty_queue = 0x12345678u;
  HIP_CHECK(hipMemcpy(d_fc1_queue.get(), &dirty_queue, sizeof(dirty_queue),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_fc2_queue.get(), &dirty_queue, sizeof(dirty_queue),
                      hipMemcpyHostToDevice));

  const int status = hip_qmoe_bucket_tokens_ragged(
      reinterpret_cast<void*>(stream), d_indices.get(), d_weights.get(),
      d_counts.get(), d_offsets.get(), d_sorted_ids.get(),
      d_sorted_weights.get(), d_pair_to_sorted.get(), d_groups.get(),
      d_group_count.get(), d_fc1_queue.get(), d_fc2_queue.get(), pairs,
      experts, /*k=*/1, kMaxRowsPerGroup, sizeof(__half));
  if (status != static_cast<int>(hipSuccess)) {
    std::cout << "FAIL relL2=inf n=" << pairs
              << " shape=adaptive_descriptors launch=" << status << '\n';
    return false;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int32_t> got_counts(experts);
  std::vector<int32_t> got_offsets(experts + 1);
  std::vector<int32_t> got_groups(expected_groups.size());
  int32_t got_group_count = 0;
  uint32_t got_fc1_queue = dirty_queue;
  uint32_t got_fc2_queue = dirty_queue;
  HIP_CHECK(hipMemcpy(got_counts.data(), d_counts.get(),
                      got_counts.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_offsets.data(), d_offsets.get(),
                      got_offsets.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_group_count, d_group_count.get(), sizeof(int32_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_groups.data(), d_groups.get(),
                      got_groups.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_fc1_queue, d_fc1_queue.get(), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_fc2_queue, d_fc2_queue.get(), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  bool reserved_zero = true;
  for (size_t group = 0; group < got_groups.size() / 4; ++group) {
    reserved_zero = reserved_zero && got_groups[group * 4 + 3] == 0;
  }
  const bool pass = got_counts == expected_counts &&
                    got_offsets == expected_offsets &&
                    got_group_count ==
                        static_cast<int32_t>(expected_groups.size() / 4) &&
                    got_groups == expected_groups && got_fc1_queue == 0 &&
                    got_fc2_queue == 0 && reserved_zero;
  std::cout << (pass ? "PASS" : "FAIL") << " relL2=0 n=" << pairs
            << " shape=adaptive_descriptors maxAbs=0 metadata="
            << (pass ? "PASS" : "FAIL")
            << " descriptor_field=reserved_zero\n";
  return pass;
}

bool run_task_policy_case(bool fc1) {
  const TaskPolicyExpectation expected = expected_task_policy(fc1);
  bool pass = true;
  bool saw_kind[3] = {false, false, false};
  for (int rows = 1; rows <= kMaxRowsPerGroup; ++rows) {
    int32_t policy_id = -1;
    int32_t task_kind = -1;
    const int status = hip_qmoe_ragged_matmul_nbits_get_task_kind(
        fc1 ? 1 : 0, rows, &policy_id, &task_kind);
    const int expected_kind =
        rows <= expected.gemv_max_rows
            ? kTaskGemv
            : rows <= expected.wmma16_max_rows ? kTaskWmma16 : kTaskWmma64;
    pass = pass && status == static_cast<int>(hipSuccess) &&
           policy_id == expected.id && task_kind == expected_kind;
    if (task_kind >= kTaskGemv && task_kind <= kTaskWmma64) {
      saw_kind[task_kind] = true;
    }
  }
  pass = pass && saw_kind[kTaskGemv] && saw_kind[kTaskWmma16] &&
         saw_kind[kTaskWmma64];
  std::cout << (pass ? "PASS" : "FAIL") << " relL2=0 n=64"
            << " shape=task_policy stage=" << (fc1 ? "fc1" : "fc2")
            << " policy=" << expected.name
            << " boundaries=" << expected.gemv_max_rows << ','
            << expected.wmma16_max_rows << ",64\n";
  return pass;
}

bool run_launcher_grid_case() {
  struct LaunchShape {
    int64_t rows;
    int64_t n;
  };
  constexpr LaunchShape shapes[] = {
      {1, 17}, {20, 17}, {64, 5760}, {512, 5760}, {2048, 2880}};

  bool pass = true;
  bool saw_task_bound = false;
  bool saw_occupancy_bound = false;
  int32_t expected_cu_count = 0;
  int32_t expected_resident_blocks = 0;
  const char* block_override =
      std::getenv("HIPDNN_QMOE_RAGGED_GRID_BLOCKS");
  const char* factor_override =
      std::getenv("HIPDNN_QMOE_RAGGED_GRID_FACTOR");
  const int valid_blocks[] = {20, 40, 80, 160, 320, 640, 4096};
  const int valid_factors[] = {1, 2, 4, 8, 16};
  auto parse_candidate = [](const char* text, const int* candidates,
                            size_t candidate_count) {
    if (!text || !*text) {
      return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return 0;
    }
    const int value = static_cast<int>(parsed);
    return std::find(candidates, candidates + candidate_count, value) !=
                   candidates + candidate_count
               ? value
               : 0;
  };
  const int expected_blocks =
      parse_candidate(block_override, valid_blocks,
                      sizeof(valid_blocks) / sizeof(valid_blocks[0]));
  const int expected_factor =
      block_override
          ? 0
          : parse_candidate(factor_override, valid_factors,
                            sizeof(valid_factors) / sizeof(valid_factors[0]));
  for (const LaunchShape& shape : shapes) {
    int32_t cu_count = 0;
    int32_t resident_blocks = 0;
    int32_t grid_blocks = 0;
    int64_t upper_tasks = 0;
    const int status = hip_qmoe_ragged_matmul_nbits_get_launch_config(
        shape.rows, shape.n, &cu_count, &resident_blocks, &grid_blocks,
        &upper_tasks);
    const int64_t expected_tasks = shape.rows * ((shape.n + 63) / 64);
    const int64_t resident_capacity =
        static_cast<int64_t>(cu_count) * resident_blocks;
    const int production_factor = shape.rows >= 512 ? 2 : 1;
    int64_t expected_limit = resident_capacity * production_factor;
    if (expected_blocks > 0) {
      expected_limit = expected_blocks;
    } else if (expected_factor > 0) {
      expected_limit *= expected_factor;
    }
    const int64_t expected_grid = std::min(expected_tasks, expected_limit);
    const bool shape_pass = status == static_cast<int>(hipSuccess) &&
                            cu_count > 0 && resident_blocks > 0 &&
                            upper_tasks == expected_tasks &&
                            grid_blocks == expected_grid &&
                            grid_blocks <= upper_tasks;
    if (expected_cu_count == 0) {
      expected_cu_count = cu_count;
      expected_resident_blocks = resident_blocks;
    } else {
      pass = pass && cu_count == expected_cu_count &&
             resident_blocks == expected_resident_blocks;
    }
    saw_task_bound = saw_task_bound || expected_tasks <= resident_capacity;
    saw_occupancy_bound =
        saw_occupancy_bound || expected_tasks > resident_capacity;
    pass = pass && shape_pass;
    std::cout << (shape_pass ? "PASS" : "FAIL")
              << " launch_config rows=" << shape.rows << " N=" << shape.n
              << " cu_count=" << cu_count
              << " resident_blocks_per_cu=" << resident_blocks
              << " occupancy_cap=" << resident_capacity
              << " expected_grid_limit=" << expected_limit
              << " grid_blocks=" << grid_blocks
              << " upper_tasks=" << upper_tasks << '\n';
  }
  pass = pass && saw_task_bound && saw_occupancy_bound;
  std::cout << (pass ? "PASS" : "FAIL")
            << " relL2=0 n=5 shape=launcher_grid maxAbs=0"
            << " task_bound=" << (saw_task_bound ? "PASS" : "FAIL")
            << " occupancy_bound="
            << (saw_occupancy_bound ? "PASS" : "FAIL") << '\n';
  return pass;
}

struct PackedMatrix {
  int experts = 0;
  int n = 0;
  int k = 0;
  int block_size = 0;
  int k_blocks = 0;
  int row_bytes = 0;
  std::vector<uint8_t> weights;
  std::vector<__half> scales;
  std::vector<uint8_t> zero_points;
  std::vector<__half> bias;
};

PackedMatrix make_matrix(int experts, int n, int k, int block_size,
                         int seed) {
  PackedMatrix matrix;
  matrix.experts = experts;
  matrix.n = n;
  matrix.k = k;
  matrix.block_size = block_size;
  matrix.k_blocks = (k + block_size - 1) / block_size;
  matrix.row_bytes = matrix.k_blocks * (block_size / 2);
  matrix.weights.resize(static_cast<size_t>(experts) * n * matrix.row_bytes, 0);
  matrix.scales.resize(static_cast<size_t>(experts) * n * matrix.k_blocks);
  matrix.zero_points.resize(
      static_cast<size_t>(experts) * n * matrix.k_blocks);
  matrix.bias.resize(static_cast<size_t>(experts) * n);

  for (int expert = 0; expert < experts; ++expert) {
    for (int col = 0; col < n; ++col) {
      const size_t expert_col = static_cast<size_t>(expert) * n + col;
      for (int group = 0; group < matrix.k_blocks; ++group) {
        matrix.scales[expert_col * matrix.k_blocks + group] = float_to_half(
            0.0078125f * static_cast<float>(1 +
                ((expert * 3 + col + group + seed) % 5)));
        matrix.zero_points[expert_col * matrix.k_blocks + group] =
            static_cast<uint8_t>(2 + ((expert + col * 3 + group + seed) % 12));
        for (int offset = 0; offset < block_size; offset += 2) {
          const int k0 = group * block_size + offset;
          const uint8_t low = k0 < k
                                  ? static_cast<uint8_t>(
                                        (expert * 7 + col * 5 + k0 * 3 + seed) &
                                        0x0f)
                                  : 8;
          const uint8_t high = k0 + 1 < k
                                   ? static_cast<uint8_t>(
                                         (expert * 7 + col * 5 +
                                          (k0 + 1) * 3 + seed) &
                                         0x0f)
                                   : 8;
          matrix.weights[expert_col * matrix.row_bytes +
                         group * (block_size / 2) + offset / 2] =
              static_cast<uint8_t>(low | (high << 4));
        }
      }
      matrix.bias[expert_col] = float_to_half(
          static_cast<float>(((expert * 3 + col + seed) % 9) - 4) / 64.0f);
    }
  }
  return matrix;
}

void cpu_matrix_row(const __half* input, int expert,
                    const PackedMatrix& matrix, __half* output) {
  const int blob_size = matrix.block_size / 2;
  for (int col = 0; col < matrix.n; ++col) {
    const size_t expert_col = static_cast<size_t>(expert) * matrix.n + col;
    float accumulator = 0.0f;
    for (int k = 0; k < matrix.k; ++k) {
      const int group = k / matrix.block_size;
      const int offset = k - group * matrix.block_size;
      const uint8_t packed =
          matrix.weights[expert_col * matrix.row_bytes + group * blob_size +
                         offset / 2];
      const float q = (offset & 1) ? static_cast<float>(packed >> 4)
                                   : static_cast<float>(packed & 0x0f);
      const float weight =
          (q - static_cast<float>(
                   matrix.zero_points[expert_col * matrix.k_blocks + group])) *
          half_to_float(matrix.scales[expert_col * matrix.k_blocks + group]);
      accumulator += half_to_float(input[k]) * weight;
    }
    output[col] = float_to_half(accumulator + half_to_float(matrix.bias[expert_col]));
  }
}

bool run_pipeline_case(hipStream_t stream) {
  constexpr int tokens = 20;
  constexpr int experts = 5;
  constexpr int top_k = 2;
  constexpr int hidden = 32;
  constexpr int inter = 16;
  constexpr int fusion_inter = 2 * inter;
  constexpr int block_size = 32;
  constexpr int pairs = tokens * top_k;
  constexpr float alpha = 1.702f;
  constexpr float beta = 1.0f;
  constexpr float limit = 7.0f;

  std::vector<__half> input(static_cast<size_t>(tokens) * hidden);
  for (int token = 0; token < tokens; ++token) {
    for (int col = 0; col < hidden; ++col) {
      input[static_cast<size_t>(token) * hidden + col] =
          float_to_half(input_value(token + 3, col));
    }
  }

  std::vector<int32_t> expert_indices(pairs);
  std::vector<__half> expert_weights(pairs);
  for (int token = 0; token < tokens; ++token) {
    expert_indices[token * top_k] = 0;
    expert_indices[token * top_k + 1] = 1 + token % (experts - 1);
    expert_weights[token * top_k] = float_to_half(0.625f);
    expert_weights[token * top_k + 1] = float_to_half(0.375f);
  }

  std::vector<int32_t> expected_counts(experts, 0);
  for (int32_t expert : expert_indices) {
    ++expected_counts[expert];
  }
  std::vector<int32_t> expected_offsets(experts + 1, 0);
  for (int expert = 0; expert < experts; ++expert) {
    expected_offsets[expert + 1] =
        expected_offsets[expert] + expected_counts[expert];
  }
  std::vector<int32_t> expected_sorted_ids(pairs);
  std::vector<__half> expected_sorted_weights(pairs);
  std::vector<int32_t> expected_pair_to_sorted(pairs, -1);
  for (int expert = 0; expert < experts; ++expert) {
    int write = expected_offsets[expert];
    for (int pair = 0; pair < pairs; ++pair) {
      if (expert_indices[pair] == expert) {
        expected_sorted_ids[write] = pair / top_k;
        expected_sorted_weights[write] = expert_weights[pair];
        expected_pair_to_sorted[pair] = write++;
      }
    }
  }
  std::vector<int32_t> expected_groups;
  for (int expert = 0; expert < experts; ++expert) {
    for (int row = expected_offsets[expert]; row < expected_offsets[expert + 1];
         row += kMaxRowsPerGroup) {
      const int rows = std::min(kMaxRowsPerGroup,
                                expected_offsets[expert + 1] - row);
      expected_groups.push_back(expert);
      expected_groups.push_back(row);
      expected_groups.push_back(rows);
      expected_groups.push_back(0);
    }
  }

  const PackedMatrix fc1 =
      make_matrix(experts, fusion_inter, hidden, block_size, 1);
  const PackedMatrix fc2 = make_matrix(experts, hidden, inter, block_size, 9);

  std::vector<__half> reference_fc1(static_cast<size_t>(pairs) * fusion_inter);
  std::vector<__half> reference_act(static_cast<size_t>(pairs) * inter);
  std::vector<__half> reference_fc2(static_cast<size_t>(pairs) * hidden);
  for (int row = 0; row < pairs; ++row) {
    const int token = expected_sorted_ids[row];
    int expert = -1;
    for (int candidate = 0; candidate < experts; ++candidate) {
      if (row >= expected_offsets[candidate] &&
          row < expected_offsets[candidate + 1]) {
        expert = candidate;
        break;
      }
    }
    cpu_matrix_row(input.data() + static_cast<size_t>(token) * hidden, expert,
                   fc1,
                   reference_fc1.data() + static_cast<size_t>(row) * fusion_inter);
    for (int col = 0; col < inter; ++col) {
      const float gate = half_to_float(
          reference_fc1[static_cast<size_t>(row) * fusion_inter + 2 * col]);
      const float linear = half_to_float(
          reference_fc1[static_cast<size_t>(row) * fusion_inter + 2 * col + 1]);
      const float g = std::min(gate, limit);
      const float l = std::max(-limit, std::min(linear, limit));
      reference_act[static_cast<size_t>(row) * inter + col] =
          float_to_half(g * (1.0f / (1.0f + std::exp(-alpha * g))) *
                        (l + beta));
    }
    cpu_matrix_row(reference_act.data() + static_cast<size_t>(row) * inter,
                   expert, fc2,
                   reference_fc2.data() + static_cast<size_t>(row) * hidden);
  }

  std::vector<__half> expected_output(static_cast<size_t>(tokens) * hidden,
                                      float_to_half(0.0f));
  for (int token = 0; token < tokens; ++token) {
    for (int col = 0; col < hidden; ++col) {
      uint32_t used = 0;
      __half accumulator = float_to_half(0.0f);
      for (int rank = 0; rank < top_k; ++rank) {
        int best_slot = -1;
        int best_expert = std::numeric_limits<int>::max();
        for (int slot = 0; slot < top_k; ++slot) {
          if ((used & (1u << slot)) == 0 &&
              expert_indices[token * top_k + slot] < best_expert) {
            best_expert = expert_indices[token * top_k + slot];
            best_slot = slot;
          }
        }
        used |= 1u << best_slot;
        const int pair = token * top_k + best_slot;
        const int sorted_row = expected_pair_to_sorted[pair];
        accumulator = float_to_half(
            half_to_float(accumulator) + half_to_float(expert_weights[pair]) *
                half_to_float(reference_fc2[
                    static_cast<size_t>(sorted_row) * hidden + col]));
      }
      expected_output[static_cast<size_t>(token) * hidden + col] = accumulator;
    }
  }

  DeviceBuffer<__half> d_input(input.size());
  DeviceBuffer<int32_t> d_expert_indices(expert_indices.size());
  DeviceBuffer<__half> d_expert_weights(expert_weights.size());
  DeviceBuffer<int32_t> d_counts(experts);
  DeviceBuffer<int32_t> d_offsets(experts + 1);
  DeviceBuffer<int32_t> d_sorted_ids(pairs);
  DeviceBuffer<__half> d_sorted_weights(pairs);
  DeviceBuffer<int32_t> d_pair_to_sorted(pairs);
  DeviceBuffer<int32_t> d_groups(static_cast<size_t>(pairs) * 4);
  DeviceBuffer<int32_t> d_group_count(1);
  DeviceBuffer<uint32_t> d_fc1_queue(1);
  DeviceBuffer<uint32_t> d_fc2_queue(1);
  DeviceBuffer<uint8_t> d_fc1_weights(fc1.weights.size());
  DeviceBuffer<__half> d_fc1_scales(fc1.scales.size());
  DeviceBuffer<uint8_t> d_fc1_zero_points(fc1.zero_points.size());
  DeviceBuffer<__half> d_fc1_bias(fc1.bias.size());
  DeviceBuffer<uint8_t> d_fc2_weights(fc2.weights.size());
  DeviceBuffer<__half> d_fc2_scales(fc2.scales.size());
  DeviceBuffer<uint8_t> d_fc2_zero_points(fc2.zero_points.size());
  DeviceBuffer<__half> d_fc2_bias(fc2.bias.size());
  DeviceBuffer<__half> d_fc1_output(static_cast<size_t>(pairs) * fusion_inter);
  DeviceBuffer<__half> d_act(static_cast<size_t>(pairs) * inter);
  DeviceBuffer<__half> d_fc2_output(static_cast<size_t>(pairs) * hidden);
  DeviceBuffer<__half> d_output(static_cast<size_t>(tokens) * hidden);

  upload(d_input, input);
  upload(d_expert_indices, expert_indices);
  upload(d_expert_weights, expert_weights);
  upload(d_fc1_weights, fc1.weights);
  upload(d_fc1_scales, fc1.scales);
  upload(d_fc1_zero_points, fc1.zero_points);
  upload(d_fc1_bias, fc1.bias);
  upload(d_fc2_weights, fc2.weights);
  upload(d_fc2_scales, fc2.scales);
  upload(d_fc2_zero_points, fc2.zero_points);
  upload(d_fc2_bias, fc2.bias);

  std::vector<__half> first_output;
  bool deterministic = true;
  for (int repeat = 0; repeat < 20; ++repeat) {
    int status = hip_qmoe_bucket_tokens_ragged(
        reinterpret_cast<void*>(stream), d_expert_indices.get(),
        d_expert_weights.get(), d_counts.get(), d_offsets.get(),
        d_sorted_ids.get(), d_sorted_weights.get(), d_pair_to_sorted.get(),
        d_groups.get(), d_group_count.get(), d_fc1_queue.get(),
        d_fc2_queue.get(), tokens, experts, top_k, kMaxRowsPerGroup,
        sizeof(__half));
    if (status != static_cast<int>(hipSuccess)) return false;
    status = hip_qmoe_ragged_matmul_nbits(
        reinterpret_cast<void*>(stream), d_input.get(), d_sorted_ids.get(),
        d_groups.get(), d_group_count.get(), d_fc1_queue.get(),
        d_fc1_weights.get(), d_fc1_scales.get(), d_fc1_zero_points.get(),
        d_fc1_bias.get(), d_fc1_output.get(), pairs, experts, fusion_inter,
        hidden, block_size, sizeof(__half));
    if (status != static_cast<int>(hipSuccess)) return false;
    status = hip_qmoe_swiglu(reinterpret_cast<void*>(stream), d_fc1_output.get(),
                             d_act.get(), pairs, inter, alpha, beta, limit,
                             sizeof(__half));
    if (status != static_cast<int>(hipSuccess)) return false;
    status = hip_qmoe_ragged_matmul_nbits(
        reinterpret_cast<void*>(stream), d_act.get(), nullptr, d_groups.get(),
        d_group_count.get(), d_fc2_queue.get(), d_fc2_weights.get(),
        d_fc2_scales.get(), d_fc2_zero_points.get(), d_fc2_bias.get(),
        d_fc2_output.get(), pairs, experts, hidden, inter, block_size,
        sizeof(__half));
    if (status != static_cast<int>(hipSuccess)) return false;
    status = hip_qmoe_reduce_sorted_pairs(
        reinterpret_cast<void*>(stream), d_fc2_output.get(),
        d_pair_to_sorted.get(), d_expert_indices.get(), d_expert_weights.get(),
        d_output.get(), tokens, hidden, top_k, sizeof(__half));
    if (status != static_cast<int>(hipSuccess)) return false;
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<__half> current(static_cast<size_t>(tokens) * hidden);
    HIP_CHECK(hipMemcpy(current.data(), d_output.get(),
                        current.size() * sizeof(__half), hipMemcpyDeviceToHost));
    if (repeat == 0) {
      first_output = current;
    } else {
      deterministic =
          deterministic && std::memcmp(first_output.data(), current.data(),
                                       current.size() * sizeof(__half)) == 0;
    }
  }

  std::vector<int32_t> got_counts(experts);
  std::vector<int32_t> got_offsets(experts + 1);
  std::vector<int32_t> got_sorted_ids(pairs);
  std::vector<__half> got_sorted_weights(pairs);
  std::vector<int32_t> got_pair_to_sorted(pairs);
  std::vector<int32_t> got_groups(expected_groups.size());
  int32_t got_group_count = 0;
  uint32_t got_fc1_queue = 0;
  uint32_t got_fc2_queue = 0;
  HIP_CHECK(hipMemcpy(got_counts.data(), d_counts.get(),
                      got_counts.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_offsets.data(), d_offsets.get(),
                      got_offsets.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_sorted_ids.data(), d_sorted_ids.get(),
                      got_sorted_ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_sorted_weights.data(), d_sorted_weights.get(),
                      got_sorted_weights.size() * sizeof(__half),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_pair_to_sorted.data(), d_pair_to_sorted.get(),
                      got_pair_to_sorted.size() * sizeof(int32_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_group_count, d_group_count.get(), sizeof(int32_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_groups.data(), d_groups.get(),
                      got_groups.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_fc1_queue, d_fc1_queue.get(), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&got_fc2_queue, d_fc2_queue.get(), sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  const bool metadata_ok =
      got_counts == expected_counts && got_offsets == expected_offsets &&
      got_sorted_ids == expected_sorted_ids &&
      std::memcmp(got_sorted_weights.data(), expected_sorted_weights.data(),
                  pairs * sizeof(__half)) == 0 &&
      got_pair_to_sorted == expected_pair_to_sorted &&
      got_group_count == static_cast<int32_t>(expected_groups.size() / 4) &&
      got_groups == expected_groups;
  const QueuePolicyExpectation queue_policy = expected_queue_policy();
  // Atomic policies advance each independently reset queue in exact chunk
  // multiples. Grid-stride deliberately leaves both queue heads at zero.
  const auto queue_ok = [&queue_policy](uint32_t value) {
    return queue_policy.grid_stride
               ? value == 0
               : value > 0 && value % queue_policy.chunk == 0;
  };
  const bool queues_ok = queue_ok(got_fc1_queue) && queue_ok(got_fc2_queue);
  const Metrics metrics = compare(first_output, expected_output);
  const bool pass = metadata_ok && queues_ok && deterministic && metrics.finite &&
                    metrics.max_abs <= 0.05 && metrics.rel_l2 <= 0.01;
  std::cout << (pass ? "PASS" : "FAIL") << " relL2=" << metrics.rel_l2
            << " n=" << first_output.size() << " shape=pipeline"
            << " maxAbs=" << metrics.max_abs
            << " metadata=" << (metadata_ok ? "PASS" : "FAIL")
            << " deterministic=" << (deterministic ? "PASS" : "FAIL")
            << " queues=" << (queues_ok ? "PASS" : "FAIL")
            << " queue_policy=" << queue_policy.name
            << " fc1_queue_head=" << got_fc1_queue
            << " fc2_queue_head=" << got_fc2_queue
            << " launches=fc1:1,fc2:1\n";
  return pass;
}

struct Options {
  int coverage = 3;
  bool launcher_grid_only = false;
};

bool parse_options(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--coverage" && i + 1 < argc) {
      options->coverage = std::atoi(argv[++i]);
    } else if (argument == "--launcher-grid-only") {
      options->launcher_grid_only = true;
    } else {
      std::cerr << "usage: " << argv[0]
                << " [--coverage 1|2|3] [--launcher-grid-only]\n";
      return false;
    }
  }
  return options->coverage >= 1 && options->coverage <= 3;
}

} // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    return 2;
  }
  if (options.launcher_grid_only) {
    const bool pass = run_launcher_grid_case();
    std::cout << (pass ? "ALL PASS" : "SOME FAIL") << '\n';
    return pass ? 0 : 1;
  }
  const int coverage = options.coverage;

  int shape_count = 0;
  for (const Shape& shape : kShapes) {
    shape_count += shape.min_coverage <= coverage ? 1 : 0;
  }
  constexpr int category_count = 3 * 2 * 2 * 2;
  const int model_shape_cases = coverage == 3 ? 2 : 0;
  const int adaptive_shape_cases =
      coverage == 3
          ? static_cast<int>(sizeof(kAdaptiveShapes) / sizeof(kAdaptiveShapes[0]))
          : 0;
  const int descriptor_cases = coverage == 3 ? 1 : 0;
  constexpr int task_policy_cases = 2;
  constexpr int launcher_cases = 1;
  std::cout << "qmoe_ragged_mm coverage=" << coverage
            << " cases=" << shape_count * category_count + model_shape_cases +
                               adaptive_shape_cases + descriptor_cases +
                               task_policy_cases + launcher_cases
            << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  all_pass = run_launcher_grid_case() && all_pass;
  all_pass = run_task_policy_case(/*fc1=*/true) && all_pass;
  all_pass = run_task_policy_case(/*fc1=*/false) && all_pass;
  for (const Shape& shape : kShapes) {
    if (shape.min_coverage > coverage) {
      continue;
    }
    for (Routing routing : {Routing::Balanced, Routing::EmptyExperts,
                            Routing::Skewed}) {
      for (bool gather : {false, true}) {
        for (bool use_bias : {false, true}) {
          for (bool use_zero_points : {false, true}) {
            all_pass = run_case(shape, routing, gather, use_bias,
                                use_zero_points, stream) &&
                       all_pass;
          }
        }
      }
    }
  }
  if (coverage == 3) {
    all_pass = run_bucket_descriptor_case(stream) && all_pass;
    for (const Shape& shape : kAdaptiveShapes) {
      all_pass = run_case(shape, Routing::Balanced,
                          /*gather=*/(shape.rows & 1) != 0,
                          /*use_bias=*/true, /*use_zero_points=*/true, stream) &&
                 all_pass;
    }
    const Shape gpt_oss_fc1 =
        {"gpt_oss_20b_fc1", 2, 1, 5760, 2880, 32, 3};
    const Shape gpt_oss_fc2 =
        {"gpt_oss_20b_fc2", 2, 1, 2880, 2880, 32, 3};
    all_pass = run_case(gpt_oss_fc1, Routing::Balanced,
                        /*gather=*/true, /*use_bias=*/false,
                        /*use_zero_points=*/false, stream) &&
               all_pass;
    all_pass = run_case(gpt_oss_fc2, Routing::Balanced,
                        /*gather=*/false, /*use_bias=*/false,
                        /*use_zero_points=*/false, stream) &&
               all_pass;
  }
  all_pass = run_pipeline_case(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
