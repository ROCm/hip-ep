/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for ONNX STFT (opset 17), real-to-complex onesided.
//
// Pipeline:
//   1. hip_stft_frame_window  -- carve overlapping frames out of `signal`
//                                  and multiply by the window in fp32.
//   2. rocfft_execute (R2C)    -- batched onesided real DFT,
//                                  hipfftReal -> hipfftComplex.
//   3. hip_stft_split_complex -- copy interleaved (re, im) pairs into the
//                                  ONNX-shaped destination buffer.
//
// All temp buffers (the framing buffer, the rocFFT output, and the
// rocFFT work buffer) live inside the shared runtime workspace, so we
// pay a single hipMalloc the first time the wrapper is called and
// reuse forever after.
//
// rocFFT plan creation is non-trivial -- we cache the plan + execution
// info + work buffer size keyed by (frame_length, num_transforms).
// Kokoro only ever calls us with a single (20, 76) tuple so the cache
// holds at most a couple of entries; the linear search is fine.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <rocfft/rocfft.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <new>

namespace {

constexpr int kMaxStftPlanCache = 8;

struct StftPlanEntry {
  bool valid;
  int64_t frame_length;
  int64_t num_transforms;
  rocfft_plan plan;
  rocfft_execution_info info;
  size_t work_size;
};

// Per-DLL plan cache.  RuntimeState is single-threaded, and each model
// DLL gets its own copy of this static via the bitcode link, so no
// further synchronization is needed.
static StftPlanEntry g_stft_plan_cache[kMaxStftPlanCache] = {};
static bool g_rocfft_setup_done = false;

static int ensure_rocfft_setup() {
  if (g_rocfft_setup_done)
    return 0;
  rocfft_status st = rocfft_setup();
  if (st != rocfft_status_success) {
    fprintf(stderr, "wrap_stft: rocfft_setup failed (status=%d)\n", (int)st);
    return -1;
  }
  g_rocfft_setup_done = true;
  return 0;
}

static StftPlanEntry *lookup_or_create_plan(int64_t frame_length,
                                            int64_t num_transforms) {
  for (auto &e : g_stft_plan_cache) {
    if (e.valid && e.frame_length == frame_length &&
        e.num_transforms == num_transforms)
      return &e;
  }
  StftPlanEntry *slot = nullptr;
  for (auto &e : g_stft_plan_cache) {
    if (!e.valid) {
      slot = &e;
      break;
    }
  }
  if (!slot) {
    fprintf(stderr,
            "wrap_stft: rocFFT plan cache full (%d entries) -- "
            "consider raising kMaxStftPlanCache\n",
            kMaxStftPlanCache);
    return nullptr;
  }

  // rocFFT takes an array of lengths in column-major order; for a 1-D
  // transform of length N this is just {N}.  number_of_transforms is the
  // batch count -- we set it to batch * n_frames so rocFFT processes
  // every frame in one launch.
  size_t length = static_cast<size_t>(frame_length);
  rocfft_plan plan = nullptr;
  rocfft_status st = rocfft_plan_create(
      &plan, rocfft_placement_notinplace,
      rocfft_transform_type_real_forward, rocfft_precision_single,
      /*dimensions=*/1, &length,
      /*number_of_transforms=*/static_cast<size_t>(num_transforms),
      /*description=*/nullptr);
  if (st != rocfft_status_success) {
    fprintf(stderr,
            "wrap_stft: rocfft_plan_create failed (status=%d, "
            "frame_length=%lld, num_transforms=%lld)\n",
            (int)st, (long long)frame_length, (long long)num_transforms);
    return nullptr;
  }

  size_t work_size = 0;
  st = rocfft_plan_get_work_buffer_size(plan, &work_size);
  if (st != rocfft_status_success) {
    fprintf(stderr,
            "wrap_stft: rocfft_plan_get_work_buffer_size failed (status=%d)\n",
            (int)st);
    rocfft_plan_destroy(plan);
    return nullptr;
  }

  rocfft_execution_info info = nullptr;
  st = rocfft_execution_info_create(&info);
  if (st != rocfft_status_success) {
    fprintf(stderr,
            "wrap_stft: rocfft_execution_info_create failed (status=%d)\n",
            (int)st);
    rocfft_plan_destroy(plan);
    return nullptr;
  }

  slot->valid = true;
  slot->frame_length = frame_length;
  slot->num_transforms = num_transforms;
  slot->plan = plan;
  slot->info = info;
  slot->work_size = work_size;
  return slot;
}

static int hipdnn_ep_to_hip_dtype_stft(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

static inline size_t align_up(size_t v, size_t a) {
  return (v + (a - 1)) & ~(a - 1);
}

} // namespace

extern "C" int wrap_stft(RuntimeState *state, void *signal, void *window,
                         void *output, int64_t batch, int64_t signal_len,
                         int64_t frame_step, int64_t frame_length,
                         int64_t n_frames, int64_t n_freqs, int64_t onesided,
                         int64_t data_type) {
  if (!state || !signal || !output) {
    fprintf(stderr, "wrap_stft: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_stft(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_stft: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  if (batch <= 0 || signal_len <= 0 || frame_step <= 0 || frame_length <= 0 ||
      n_frames <= 0 || n_freqs <= 0)
    return 0;

  int64_t expected_freqs =
      onesided ? frame_length / 2 + 1 : frame_length;
  if (n_freqs != expected_freqs) {
    fprintf(stderr,
            "wrap_stft: n_freqs %lld doesn't match expected %lld for "
            "frame_length=%lld onesided=%lld\n",
            (long long)n_freqs, (long long)expected_freqs,
            (long long)frame_length, (long long)onesided);
    return -1;
  }
  if (!onesided) {
    fprintf(stderr,
            "wrap_stft: onesided=0 is not supported yet (Kokoro uses "
            "onesided=1)\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  if (ensure_rocfft_setup() != 0)
    return -1;

  int64_t num_transforms = batch * n_frames;
  StftPlanEntry *entry = lookup_or_create_plan(frame_length, num_transforms);
  if (!entry)
    return -1;

  // Workspace layout:
  //   [frames | complex_out | rocfft_work]
  // Each segment is 64-byte aligned for safety with vectorized loads.
  constexpr size_t kAlign = 64;
  size_t frames_bytes = align_up(static_cast<size_t>(batch) *
                                     static_cast<size_t>(n_frames) *
                                     static_cast<size_t>(frame_length) *
                                     sizeof(float),
                                 kAlign);
  size_t complex_bytes = align_up(static_cast<size_t>(batch) *
                                      static_cast<size_t>(n_frames) *
                                      static_cast<size_t>(n_freqs) * 2 *
                                      sizeof(float),
                                  kAlign);
  size_t work_bytes = align_up(entry->work_size, kAlign);
  size_t total_bytes = frames_bytes + complex_bytes + work_bytes;

  if (hipdnn_ep_state_ensure_workspace(state, total_bytes) != 0) {
    fprintf(stderr,
            "wrap_stft: failed to ensure %zu-byte workspace\n", total_bytes);
    return -1;
  }
  char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
  void *frames = ws;
  void *complex_out = ws + frames_bytes;
  void *work_buf = entry->work_size > 0 ? (ws + frames_bytes + complex_bytes)
                                        : nullptr;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_stft: batch=%lld, signal_len=%lld, frame_step=%lld, "
      "frame_length=%lld, n_frames=%lld, n_freqs=%lld, onesided=%lld, "
      "dtype=%s, ws=%zuB (frames=%zu, complex=%zu, work=%zu)\n",
      (long long)batch, (long long)signal_len, (long long)frame_step,
      (long long)frame_length, (long long)n_frames, (long long)n_freqs,
      (long long)onesided, hipdnn_ep_datatype_name(data_type), total_bytes,
      frames_bytes, complex_bytes, work_bytes);

  // Step 1: framing + window multiply.
  int rc = hip_stft_frame_window(stream, signal, window, frames, batch,
                                 signal_len, frame_length, frame_step,
                                 n_frames, hip_dtype);
  if (rc != 0)
    return rc;

  // Step 2: bind stream + work buffer to the plan and execute.
  rocfft_status st = rocfft_execution_info_set_stream(entry->info, stream);
  if (st != rocfft_status_success) {
    fprintf(stderr,
            "wrap_stft: rocfft_execution_info_set_stream failed (status=%d)\n",
            (int)st);
    return -1;
  }
  if (entry->work_size > 0 && work_buf) {
    st = rocfft_execution_info_set_work_buffer(entry->info, work_buf,
                                               entry->work_size);
    if (st != rocfft_status_success) {
      fprintf(stderr,
              "wrap_stft: rocfft_execution_info_set_work_buffer failed "
              "(status=%d, work_size=%zu)\n",
              (int)st, entry->work_size);
      return -1;
    }
  }
  void *in_arr[1] = {frames};
  void *out_arr[1] = {complex_out};
  st = rocfft_execute(entry->plan, in_arr, out_arr, entry->info);
  if (st != rocfft_status_success) {
    fprintf(stderr, "wrap_stft: rocfft_execute failed (status=%d)\n",
            (int)st);
    return -1;
  }

  // Step 3: repack interleaved complex into the (..., 2) destination.
  rc = hip_stft_split_complex(stream, complex_out, output, batch, n_frames,
                              n_freqs, hip_dtype);
  if (rc != 0)
    return rc;

  // Debug: check output values.
  {
    hipStreamSynchronize(static_cast<hipStream_t>(stream));
    (void)hipGetLastError();
    int64_t out_count = batch * n_frames * n_freqs * 2;
    float first4[4] = {0};
    hipMemcpy(first4, output, sizeof(first4), hipMemcpyDeviceToHost);
    float maxv = 0;
    std::vector<float> h(out_count);
    hipMemcpy(h.data(), output, out_count * sizeof(float), hipMemcpyDeviceToHost);
    for (auto v : h) if (std::abs(v) > maxv) maxv = std::abs(v);
    fprintf(stderr,
            "[stft_dbg] output: n=%lld first=[%.4f,%.4f,%.4f,%.4f] max=%.4f\n",
            (long long)out_count, first4[0], first4[1], first4[2], first4[3],
            maxv);
    fflush(stderr);
  }

  return 0;
}
