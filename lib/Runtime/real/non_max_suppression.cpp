/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX NonMaxSuppression (opset 11+), selected on the host.
//
// Selection is a sequential greedy scan per (batch, class), and the result
// count is data-dependent, so the inputs are copied to the host, selected
// there with the same ordering and IoU rules as the ONNX Runtime CPU kernel,
// and the [N, 3] triples plus N are copied back.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"

#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

namespace {

struct Corners {
  float yMin, xMin, yMax, xMax;
};

Corners toCorners(const float *box, int64_t centerPointBox) {
  if (centerPointBox) {
    float halfWidth = box[2] / 2.0f;
    float halfHeight = box[3] / 2.0f;
    return {box[1] - halfHeight, box[0] - halfWidth, box[1] + halfHeight,
            box[0] + halfWidth};
  }
  return {std::min(box[0], box[2]), std::min(box[1], box[3]),
          std::max(box[0], box[2]), std::max(box[1], box[3])};
}

/// Degenerate boxes and empty intersections never suppress.
bool suppresses(const Corners &a, const Corners &b, float iouThreshold) {
  float xMin = std::max(a.xMin, b.xMin);
  float xMax = std::min(a.xMax, b.xMax);
  if (xMax <= xMin)
    return false;
  float yMin = std::max(a.yMin, b.yMin);
  float yMax = std::min(a.yMax, b.yMax);
  if (yMax <= yMin)
    return false;
  float intersection = (xMax - xMin) * (yMax - yMin);
  float areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin);
  float areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin);
  float unionArea = areaA + areaB - intersection;
  if (intersection <= 0.0f || areaA <= 0.0f || areaB <= 0.0f ||
      unionArea <= 0.0f)
    return false;
  return intersection / unionArea > iouThreshold;
}

bool copyToHost(void *dst, const void *src, size_t bytes, hipStream_t stream,
                const char *what) {
  if (bytes == 0)
    return true;
  // hipMemcpyDefault, not DeviceToHost: a host-scalar input lives in mapped
  // host scratch, and an explicit device-to-host copy of that pointer fails
  // with `invalid argument`.
  hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: %s D2H failed: %s\n",
            what, hipGetErrorString(err));
    return false;
  }
  return true;
}

} // namespace

int wrap_non_max_suppression(
    RuntimeState *state, const float *boxes, const float *scores,
    const int64_t *max_output_boxes_per_class, const float *iou_threshold,
    const float *score_threshold, int64_t *selected, int32_t *count_ptr,
    int64_t num_batches, int64_t num_classes, int64_t num_boxes,
    int64_t capacity, int64_t center_point_box, int64_t has_score_threshold) {
  OP_PROFILE(
      "non_max_suppression",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "B%lld:C%lld:S%lld", (long long)num_batches,
                 (long long)num_classes, (long long)num_boxes);
        return std::string(b);
      },
      state);

  if (!state || !count_ptr || !max_output_boxes_per_class || !iou_threshold ||
      (has_score_threshold && !score_threshold)) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_non_max_suppression: null required argument\n");
    return -1;
  }
  if (num_batches < 0 || num_classes < 0 || num_boxes < 0 || capacity < 0) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: negative extent\n");
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  int64_t maxOutput = 0;
  float iou = 0.0f;
  float scoreThreshold = 0.0f;
  size_t boxCount = static_cast<size_t>(num_batches * num_boxes);
  size_t scoreCount =
      static_cast<size_t>(num_batches * num_classes * num_boxes);
  std::vector<float> hostBoxes(boxCount * 4);
  std::vector<float> hostScores(scoreCount);
  if (!copyToHost(&maxOutput, max_output_boxes_per_class, sizeof(int64_t),
                  stream, "max_output_boxes_per_class") ||
      !copyToHost(&iou, iou_threshold, sizeof(float), stream,
                  "iou_threshold") ||
      (has_score_threshold &&
       !copyToHost(&scoreThreshold, score_threshold, sizeof(float), stream,
                   "score_threshold")) ||
      (boxCount &&
       !copyToHost(hostBoxes.data(), boxes, hostBoxes.size() * sizeof(float),
                   stream, "boxes")) ||
      (scoreCount &&
       !copyToHost(hostScores.data(), scores, hostScores.size() * sizeof(float),
                   stream, "scores")))
    return -1;
  hipError_t err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: sync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  std::vector<int64_t> triples;
  std::vector<int64_t> order;
  std::vector<Corners> kept;
  // num_boxes == 0 leaves hostScores empty. Indexing it for a class pointer
  // is out of bounds; the count written below is already zero.
  if (maxOutput > 0 && num_boxes > 0) {
    for (int64_t batch = 0; batch < num_batches; ++batch) {
      std::vector<Corners> corners(static_cast<size_t>(num_boxes));
      for (int64_t box = 0; box < num_boxes; ++box)
        corners[box] = toCorners(&hostBoxes[(batch * num_boxes + box) * 4],
                                 center_point_box);
      for (int64_t cls = 0; cls < num_classes; ++cls) {
        const float *classScores =
            &hostScores[(batch * num_classes + cls) * num_boxes];
        order.clear();
        for (int64_t box = 0; box < num_boxes; ++box)
          if (!has_score_threshold || classScores[box] > scoreThreshold)
            order.push_back(box);
        // Ties keep the lower box index first. std::stable_sort would pull in
        // CRT helpers (nothrow new) the model JIT does not resolve.
        std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
          if (classScores[lhs] != classScores[rhs])
            return classScores[lhs] > classScores[rhs];
          return lhs < rhs;
        });
        kept.clear();
        for (int64_t box : order) {
          if (static_cast<int64_t>(kept.size()) >= maxOutput)
            break;
          const Corners &candidate = corners[box];
          bool suppressed =
              std::any_of(kept.begin(), kept.end(), [&](const Corners &other) {
                return suppresses(other, candidate, iou);
              });
          if (suppressed)
            continue;
          kept.push_back(candidate);
          triples.insert(triples.end(), {batch, cls, box});
        }
      }
    }
  }

  int64_t total = static_cast<int64_t>(triples.size() / 3);
  if (total > capacity) {
    fprintf(stderr,
            "[REAL] wrap_non_max_suppression: %lld selections exceed "
            "capacity %lld\n",
            (long long)total, (long long)capacity);
    return -1;
  }
  int32_t hostCount = static_cast<int32_t>(total);
  if (total > 0) {
    if (!selected)
      return -1;
    err = hipMemcpyAsync(selected, triples.data(),
                         triples.size() * sizeof(int64_t),
                         hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_non_max_suppression: selected H2D failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
  }
  err = hipMemcpyAsync(count_ptr, &hostCount, sizeof(int32_t),
                       hipMemcpyHostToDevice, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: count H2D failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  // The host staging buffers die on return.
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_non_max_suppression: sync failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_non_max_suppression: B=%lld C=%lld S=%lld "
                    "selected=%lld\n",
                    (long long)num_batches, (long long)num_classes,
                    (long long)num_boxes, (long long)total);
  return 0;
}
