/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX NonMaxSuppression. Selection and IoU rules follow the ONNX Runtime
 * CPU kernel:
 *   * candidates are visited by descending score, ties by lower box index
 *   * with a score threshold, only scores strictly above it compete
 *   * a candidate is dropped when IoU with a kept box is > iou_threshold;
 *     empty intersections and non-positive areas never suppress
 *   * corner boxes may list either corner first (min/max normalisation)
 *
 * Three launches on one stream:
 *   1. nms_select_kernel   one block per (batch, class): bitonic sort of
 *                          the candidates in the workspace, then a greedy
 *                          scan that tests each candidate against the kept
 *                          boxes in parallel.
 *   2. nms_offsets_kernel  exclusive prefix over the per-pair counts and the
 *                          total into *count.
 *   3. nms_compact_kernel  one block per pair writes its (b, c, box) rows.
 *
 * Workspace layout (BC = B * C, P = next_pow2(N), S = min(N, max_out)):
 *   int64 offsets[BC] | float keys[BC * P] | int32 ids[BC * P] |
 *   int32 kept[BC * S] | int32 kept_count[BC]
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kNmsBlockSize = 512;

struct Workspace {
  int64_t *offsets;
  float *keys;
  int32_t *ids;
  int32_t *kept;
  int32_t *kept_count;
  size_t bytes;
};

int64_t next_pow2(int64_t v) {
  int64_t p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

Workspace layout(void *base, int64_t bc, int64_t p, int64_t s) {
  Workspace ws{};
  char *cursor = static_cast<char *>(base);
  auto take = [&](size_t bytes) {
    char *at = cursor;
    if (cursor)
      cursor += bytes;
    ws.bytes += bytes;
    return at;
  };
  ws.offsets = reinterpret_cast<int64_t *>(take(bc * sizeof(int64_t)));
  ws.keys = reinterpret_cast<float *>(take(bc * p * sizeof(float)));
  ws.ids = reinterpret_cast<int32_t *>(take(bc * p * sizeof(int32_t)));
  ws.kept = reinterpret_cast<int32_t *>(take(bc * s * sizeof(int32_t)));
  ws.kept_count = reinterpret_cast<int32_t *>(take(bc * sizeof(int32_t)));
  return ws;
}

struct Box {
  float x_min, y_min, x_max, y_max;
};

__device__ inline Box load_box(const float *__restrict__ box, int center) {
  Box r;
  if (center == 0) {
    r.y_min = fminf(box[0], box[2]);
    r.y_max = fmaxf(box[0], box[2]);
    r.x_min = fminf(box[1], box[3]);
    r.x_max = fmaxf(box[1], box[3]);
  } else {
    float half_w = box[2] / 2.0f;
    float half_h = box[3] / 2.0f;
    r.x_min = box[0] - half_w;
    r.x_max = box[0] + half_w;
    r.y_min = box[1] - half_h;
    r.y_max = box[1] + half_h;
  }
  return r;
}

__device__ inline bool suppresses(const Box &a, const Box &b,
                                  float iou_threshold) {
  float ix_min = fmaxf(a.x_min, b.x_min);
  float ix_max = fminf(a.x_max, b.x_max);
  if (ix_max <= ix_min)
    return false;
  float iy_min = fmaxf(a.y_min, b.y_min);
  float iy_max = fminf(a.y_max, b.y_max);
  if (iy_max <= iy_min)
    return false;
  float inter = (ix_max - ix_min) * (iy_max - iy_min);
  if (inter <= 0.0f)
    return false;
  float area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
  float area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
  float uni = area_a + area_b - inter;
  if (area_a <= 0.0f || area_b <= 0.0f || uni <= 0.0f)
    return false;
  return inter / uni > iou_threshold;
}

// True when (ka, ia) must precede (kb, ib). Invalid slots (id < 0) sort last.
__device__ inline bool precedes(float ka, int32_t ia, float kb, int32_t ib) {
  if (ia < 0)
    return false;
  if (ib < 0)
    return true;
  return ka > kb || (ka == kb && ia < ib);
}

__global__ void nms_select_kernel(const float *__restrict__ boxes,
                                  const float *__restrict__ scores,
                                  Workspace ws, int64_t C, int64_t N,
                                  int64_t P, int64_t S, float iou_threshold,
                                  float score_threshold,
                                  int has_score_threshold, int center) {
  __shared__ int s_valid;
  __shared__ int s_num_kept;
  __shared__ int s_suppressed;

  const int tid = threadIdx.x;
  const int64_t bc = blockIdx.x;
  const int64_t b = bc / C;
  const float *class_scores = scores + bc * N;
  const float *batch_boxes = boxes + b * N * 4;
  float *keys = ws.keys + bc * P;
  int32_t *ids = ws.ids + bc * P;
  int32_t *kept = ws.kept + bc * S;

  if (tid == 0) {
    s_valid = 0;
    s_num_kept = 0;
  }
  __syncthreads();

  int local_valid = 0;
  for (int64_t i = tid; i < P; i += blockDim.x) {
    bool valid = false;
    float score = 0.0f;
    if (i < N) {
      score = class_scores[i];
      valid = !isnan(score) && (!has_score_threshold || score > score_threshold);
    }
    keys[i] = valid ? score : -INFINITY;
    ids[i] = valid ? static_cast<int32_t>(i) : -1;
    local_valid += valid ? 1 : 0;
  }
  atomicAdd(&s_valid, local_valid);
  __syncthreads();

  for (int64_t k = 2; k <= P; k <<= 1) {
    for (int64_t j = k >> 1; j > 0; j >>= 1) {
      for (int64_t i = tid; i < P; i += blockDim.x) {
        int64_t partner = i ^ j;
        if (partner <= i)
          continue;
        float ka = keys[i], kb = keys[partner];
        int32_t ia = ids[i], ib = ids[partner];
        bool leading = (i & k) == 0;
        bool swap = leading ? precedes(kb, ib, ka, ia)
                            : precedes(ka, ia, kb, ib);
        if (swap) {
          keys[i] = kb;
          keys[partner] = ka;
          ids[i] = ib;
          ids[partner] = ia;
        }
      }
      __syncthreads();
    }
  }

  const int valid = s_valid;
  for (int r = 0; r < valid; ++r) {
    if (s_num_kept >= S)
      break;
    const int32_t cand = ids[r];
    const Box cand_box = load_box(batch_boxes + static_cast<int64_t>(cand) * 4,
                                  center);
    if (tid == 0)
      s_suppressed = 0;
    __syncthreads();
    const int num_kept = s_num_kept;
    for (int j = tid; j < num_kept; j += blockDim.x) {
      Box kept_box =
          load_box(batch_boxes + static_cast<int64_t>(kept[j]) * 4, center);
      if (suppresses(kept_box, cand_box, iou_threshold))
        s_suppressed = 1;
    }
    __syncthreads();
    if (tid == 0 && !s_suppressed) {
      kept[num_kept] = cand;
      s_num_kept = num_kept + 1;
    }
    __syncthreads();
  }

  if (tid == 0)
    ws.kept_count[bc] = s_num_kept;
}

__global__ void nms_offsets_kernel(Workspace ws, int64_t BC,
                                   int32_t *__restrict__ count) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  int64_t sum = 0;
  for (int64_t i = 0; i < BC; ++i) {
    ws.offsets[i] = sum;
    sum += ws.kept_count[i];
  }
  *count = static_cast<int32_t>(sum);
}

__global__ void nms_compact_kernel(Workspace ws, int64_t C, int64_t S,
                                   int64_t *__restrict__ selected) {
  const int64_t bc = blockIdx.x;
  const int64_t b = bc / C;
  const int64_t c = bc - b * C;
  const int32_t n = ws.kept_count[bc];
  const int64_t base = ws.offsets[bc];
  const int32_t *kept = ws.kept + bc * S;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t *row = selected + (base + i) * 3;
    row[0] = b;
    row[1] = c;
    row[2] = kept[i];
  }
}

} // namespace

extern "C" size_t hip_non_max_suppression_workspace_bytes(
    int64_t b, int64_t c, int64_t n, int64_t max_output_boxes_per_class) {
  int64_t s = n < max_output_boxes_per_class ? n : max_output_boxes_per_class;
  if (b <= 0 || c <= 0 || n <= 0 || s <= 0)
    return 0;
  return layout(nullptr, b * c, next_pow2(n), s).bytes;
}

extern "C" int hip_non_max_suppression(
    void *stream, const void *boxes, const void *scores, void *selected,
    int32_t *count, int64_t b, int64_t c, int64_t n, int64_t capacity,
    int64_t max_output_boxes_per_class, float iou_threshold,
    float score_threshold, int has_score_threshold, int center_point_box,
    void *workspace, size_t workspace_bytes) {
  if (!count)
    return -1;
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();

  int64_t per_class =
      n < max_output_boxes_per_class ? n : max_output_boxes_per_class;
  if (b <= 0 || c <= 0 || n <= 0 || per_class <= 0) {
    hipMemsetAsync(count, 0, sizeof(int32_t), s);
    return static_cast<int>(hipGetLastError());
  }
  if (!boxes || !scores || !selected || !workspace) {
    fprintf(stderr, "[custom_kernels] hip_non_max_suppression: null buffer\n");
    return -1;
  }
  if (n > INT32_MAX) {
    fprintf(stderr,
            "[custom_kernels] hip_non_max_suppression: N=%lld exceeds int32\n",
            (long long)n);
    return -1;
  }
  int64_t bc = b * c;
  if (bc * per_class > capacity) {
    fprintf(stderr,
            "[custom_kernels] hip_non_max_suppression: capacity %lld < "
            "%lld\n",
            (long long)capacity, (long long)(bc * per_class));
    return -1;
  }
  int64_t p = next_pow2(n);
  Workspace ws = layout(workspace, bc, p, per_class);
  if (ws.bytes > workspace_bytes) {
    fprintf(stderr,
            "[custom_kernels] hip_non_max_suppression: workspace %zu < %zu\n",
            workspace_bytes, ws.bytes);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_non_max_suppression: B=%lld C=%lld N=%lld P=%lld "
      "S=%lld capacity=%lld iou=%f score=%f has_score=%d center=%d\n",
      (long long)b, (long long)c, (long long)n, (long long)p,
      (long long)per_class, (long long)capacity, (double)iou_threshold,
      (double)score_threshold, has_score_threshold, center_point_box);

  hipMemsetAsync(selected, 0, static_cast<size_t>(capacity) * 3 * sizeof(int64_t),
                 s);
  dim3 pairs(static_cast<unsigned int>(bc));
  hipLaunchKernelGGL(nms_select_kernel, pairs, dim3(kNmsBlockSize), 0, s,
                     static_cast<const float *>(boxes),
                     static_cast<const float *>(scores), ws, c, n, p,
                     per_class, iou_threshold, score_threshold,
                     has_score_threshold, center_point_box);
  hipLaunchKernelGGL(nms_offsets_kernel, dim3(1), dim3(1), 0, s, ws, bc,
                     count);
  hipLaunchKernelGGL(nms_compact_kernel, pairs, dim3(256), 0, s, ws, c,
                     per_class, static_cast<int64_t *>(selected));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_non_max_suppression launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
