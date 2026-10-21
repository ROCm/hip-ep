// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "routed_experts.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include <algorithm>
#include <climits>
#include <limits>

extern "C" int wrap_routed_experts(RuntimeState *state, const void *input,
                                   const int64_t *indices, const float *gates,
                                   const void *const *parameters, void *output,
                                   int64_t tokens, int64_t hidden,
                                   int64_t intermediate, int64_t experts,
                                   int64_t k, int64_t block) {
  OP_PROFILE(
      "routed_experts",
      [&] {
        return std::to_string(tokens) + "x" + std::to_string(hidden) +
               " I=" + std::to_string(intermediate) +
               " E=" + std::to_string(experts) + " k=" + std::to_string(k);
      },
      state);
  if (!state || !parameters || tokens < 0 || experts < 1 || experts > 128 ||
      k < 1 || k > experts || tokens > 65535 / k || hidden < 1 ||
      intermediate < 1 || hidden > INT_MAX || intermediate > INT_MAX ||
      (block != 32 && block != 64 && block != 128))
    return -1;
  if (tokens == 0)
    return 0;
  size_t pairs = static_cast<size_t>(tokens * k);
  size_t width = static_cast<size_t>(hidden + intermediate);
  if (width > (std::numeric_limits<size_t>::max() - 64) / 2 / pairs)
    return -1;
  size_t activationBytes = (pairs * intermediate * 2 + 63) & ~size_t(63);
  size_t quantOffset =
      (activationBytes + pairs * hidden * 2 + 63) & ~size_t(63);
  int mode = 0;
  if (hipdnn_ep_matmul_dp4a_enabled() &&
      tokens <= hipdnn_ep_matmul_dp4a_max_m() && hidden % 32 == 0 &&
      intermediate % 32 == 0)
    mode = 1;
  else if (tokens >= 8 && hidden % 32 == 0 && intermediate % 32 == 0)
    mode = 3;
  else if (tokens >= 8)
    mode = 2;
  size_t quantBytes = std::max(size_t(tokens) * hidden, pairs * intermediate);
  size_t scaleOffset = (quantOffset + quantBytes + 63) & ~size_t(63);
  size_t scaleCount = std::max(size_t(tokens) * ((hidden + block - 1) / block),
                               pairs * ((intermediate + block - 1) / block));
  size_t bytes =
      mode == 1 ? scaleOffset + scaleCount * sizeof(float) : quantOffset;
  if (mode == 3)
    bytes += routed_grouped_scratch_bytes(tokens, k, experts);
  // The state owns this storage. Operations use it in the same stream.
  if (hipdnn_ep_state_ensure_qmoe_scratch(state, bytes) != 0)
    return -1;
  auto *scratch = static_cast<char *>(hipdnn_ep_state_get_qmoe_scratch(state));
  RoutedMatrixTable up = {}, down = {};
  for (int64_t e = 0; e < experts; ++e) {
    up.weights[e] = static_cast<const uint8_t *>(parameters[e * 6]);
    up.scales[e] = parameters[e * 6 + 1];
    up.zero_points[e] = static_cast<const uint8_t *>(parameters[e * 6 + 2]);
    down.weights[e] = static_cast<const uint8_t *>(parameters[e * 6 + 3]);
    down.scales[e] = parameters[e * 6 + 4];
    down.zero_points[e] = static_cast<const uint8_t *>(parameters[e * 6 + 5]);
  }
  // Kernel argument copies keep the host pointer tables valid after return.
  return hip_routed_experts(
      hipdnn_ep_state_get_stream(state), input, indices, gates, &up, &down,
      scratch, scratch + activationBytes, output, tokens, hidden, intermediate,
      experts, k, block, mode,
      mode == 1 || mode == 3 ? scratch + quantOffset : nullptr,
      mode == 1 ? reinterpret_cast<float *>(scratch + scaleOffset) : nullptr);
}
