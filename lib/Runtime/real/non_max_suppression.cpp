/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <string>

// wrap_non_max_suppression(state, boxes, scores, selected, count_ptr,
//                          B, C, N, capacity, max_output_boxes_per_class,
//                          iou_threshold, score_threshold,
//                          has_score_threshold, center_point_box)

HIPDNN_EP_RT_EXPORT int wrap_non_max_suppression(
    RuntimeState *state, void *boxes, void *scores, void *selected,
    int32_t *count_ptr, int64_t b, int64_t c, int64_t n, int64_t capacity,
    int64_t max_output_boxes_per_class, float iou_threshold,
    float score_threshold, int64_t has_score_threshold,
    int64_t center_point_box) {
  OP_PROFILE(
      "nonmaxsuppression",
      [&] {
        char buf[128];
        snprintf(buf, sizeof(buf), "B=%lld C=%lld N=%lld max=%lld",
                 (long long)b, (long long)c, (long long)n,
                 (long long)max_output_boxes_per_class);
        return std::string(buf);
      },
      state);

  if (!state || !count_ptr) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  size_t ws_bytes = hip_non_max_suppression_workspace_bytes(
      b, c, n, max_output_boxes_per_class);
  void *workspace = nullptr;
  if (ws_bytes > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, ws_bytes) != 0) {
      fprintf(stderr,
              "[REAL] wrap_non_max_suppression: workspace of %zu bytes "
              "unavailable\n",
              ws_bytes);
      return -1;
    }
    workspace = hipdnn_ep_state_get_workspace(state);
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_non_max_suppression: B=%lld C=%lld N=%lld capacity=%lld "
      "max=%lld iou=%f score=%f has_score=%lld center=%lld ws=%zu\n",
      (long long)b, (long long)c, (long long)n, (long long)capacity,
      (long long)max_output_boxes_per_class, (double)iou_threshold,
      (double)score_threshold, (long long)has_score_threshold,
      (long long)center_point_box, ws_bytes);

  int rc = hip_non_max_suppression(
      stream, boxes, scores, selected, count_ptr, b, c, n, capacity,
      max_output_boxes_per_class, iou_threshold, score_threshold,
      static_cast<int>(has_score_threshold), static_cast<int>(center_point_box),
      workspace, ws_bytes);
  if (rc != 0) {
    fprintf(stderr,
            "[REAL] wrap_non_max_suppression: kernel launch failed (%d)\n", rc);
    return -1;
  }
  return 0;
}
