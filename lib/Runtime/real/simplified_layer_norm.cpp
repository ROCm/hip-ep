/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// SimplifiedLayerNormalization / RMSNormalization (RMS Norm):
//
//   rms    = sqrt(mean(input^2, axis) + epsilon)
//   output = (input / rms) * scale
//
// One block per row HIP kernel (see rms_norm_kernel.hip). Internal math is
// FP32 regardless of I/O dtype, and fp16 rows of even width take a packed
// __half2 path.
//
// Tensor layout (row-major): input is viewed as [num_rows, norm_size], where
// norm_size is the reduction width the lowering derived from the ONNX `axis`
// (the product of the input dims from `axis` on) and num_rows =
// input_num_elements / norm_size. `scale` carries scale_num_elements /
// norm_size gain vectors of that width, applied per row group -- one vector
// shared by all rows in the ordinary case, one per group for a grouped norm.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_rms_norm(RuntimeState *state, void *input, void *scale, void *output,
                  int64_t input_num_elements, int64_t scale_num_elements,
                  int64_t norm_num_elements, int64_t element_size_bytes,
                  int64_t axis, float epsilon, int64_t stash_type) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(norm_num_elements > 0
                                 ? input_num_elements / norm_num_elements
                                 : 0),
                 (long long)norm_num_elements);
        return std::string(b);
      },
      state);

  // `axis` arrives already resolved as norm_num_elements (the lowering has the
  // input shape and folds the two together). stash_type needs no handling: the
  // kernel always accumulates in FP32, which is what stash_type = 1 asks for.
  (void)axis;
  (void)stash_type;

  if (!state || !input || !scale || !output) {
    fprintf(stderr, "Invalid arguments to wrap_rms_norm\n");
    return -1;
  }
  if (scale_num_elements <= 0 || norm_num_elements <= 0) {
    fprintf(stderr,
            "wrap_rms_norm: scale_num_elements=%lld norm_num_elements=%lld\n",
            (long long)scale_num_elements, (long long)norm_num_elements);
    return -1;
  }

  // ONNX reduces over the axes `axis` selects and only then multiplies by
  // scale, so scale may span whole rows instead of widening the reduction.
  // Row tiling covers every scale whose shape is a suffix of the input's; a
  // scale carrying an interior broadcast dim (e.g. [G, 1] against [N, G, D])
  // cannot be expressed that way, so refuse it rather than pair rows with the
  // wrong gains.
  if (scale_num_elements % norm_num_elements != 0) {
    fprintf(stderr,
            "wrap_rms_norm: scale_num_elements=%lld is not a multiple of the "
            "normalized width %lld; a scale that broadcasts along the reduced "
            "axes is unsupported\n",
            (long long)scale_num_elements, (long long)norm_num_elements);
    return -1;
  }

  // ONNX only broadcasts Scale onto X, never the reverse. For that
  // unidirectional broadcast, |X| must be a multiple of |Scale|. Together
  // with the check above this also implies |X| is a multiple of the
  // normalized width, so the integer row split below does not truncate.
  if (input_num_elements % scale_num_elements != 0) {
    fprintf(stderr,
            "wrap_rms_norm: input_num_elements=%lld is not a multiple of "
            "scale_num_elements=%lld; Scale is not unidirectional-"
            "broadcastable onto X under row-tiled packing\n",
            (long long)input_num_elements, (long long)scale_num_elements);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2)
    hip_dtype = HIP_DTYPE_FLOAT16;
  else if (element_size_bytes == 4)
    hip_dtype = HIP_DTYPE_FLOAT32;
  else {
    fprintf(stderr,
            "wrap_rms_norm: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  const int64_t norm_size = norm_num_elements;
  const int64_t num_rows = input_num_elements / norm_size;
  const int64_t scale_rows = scale_num_elements / norm_size;

  RUNTIME_DEBUG_LOG("[REAL] wrap_rms_norm: num_rows=%lld, norm_size=%lld, "
                    "scale_rows=%lld, data_type=%s, epsilon=%e, "
                    "total_bytes=%lld\n",
                    (long long)num_rows, (long long)norm_size,
                    (long long)scale_rows,
                    (element_size_bytes == 2) ? "f16" : "f32", (double)epsilon,
                    (long long)(input_num_elements * element_size_bytes));

  return hip_rms_norm(hipdnn_ep_state_get_stream(state), input, scale, output,
                      num_rows, norm_size, scale_rows, epsilon, hip_dtype);
}

int wrap_gated_rms_norm(RuntimeState *state, void *input, void *scale,
                        void *gate, void *output, int64_t input_num_elements,
                        int64_t norm_size, int64_t element_size_bytes,
                        int64_t activation, float epsilon) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(
            b, sizeof(b), "gated %lldx%lld",
            (long long)(norm_size > 0 ? input_num_elements / norm_size : 0),
            (long long)norm_size);
        return std::string(b);
      },
      state);

  if (!state || !input || !scale || !gate || !output) {
    fprintf(stderr, "Invalid arguments to wrap_gated_rms_norm\n");
    return -1;
  }
  if (norm_size <= 0 || input_num_elements % norm_size != 0) {
    fprintf(stderr,
            "wrap_gated_rms_norm: input_num_elements=%lld is not a multiple "
            "of norm_size=%lld\n",
            (long long)input_num_elements, (long long)norm_size);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "wrap_gated_rms_norm: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  const int64_t num_rows = input_num_elements / norm_size;
  RUNTIME_DEBUG_LOG("[REAL] wrap_gated_rms_norm: num_rows=%lld, "
                    "norm_size=%lld, activation=%lld, data_type=%s, "
                    "epsilon=%e\n",
                    (long long)num_rows, (long long)norm_size,
                    (long long)activation,
                    (element_size_bytes == 2) ? "f16" : "f32", (double)epsilon);

  return hip_gated_rms_norm(hipdnn_ep_state_get_stream(state), input, scale,
                            gate, output, num_rows, norm_size,
                            static_cast<int>(activation), epsilon, hip_dtype);
}

int wrap_engram_gate(RuntimeState *state, void *key, void *query, void *value,
                     void *key_scale, void *query_scale, void *conv_scale,
                     void *output, void *normed, int64_t key_num_elements,
                     int64_t hc, int64_t hidden, int64_t element_size_bytes,
                     float epsilon) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "engram_gate %lldx%lld",
                 (long long)(hidden > 0 ? key_num_elements / hidden : 0),
                 (long long)hidden);
        return std::string(b);
      },
      state);

  if (!state || !key || !query || !value || !key_scale || !query_scale ||
      !output || (normed && !conv_scale)) {
    fprintf(stderr, "Invalid arguments to wrap_engram_gate\n");
    return -1;
  }
  if (hc <= 0 || hidden <= 0 || key_num_elements % (hc * hidden) != 0) {
    fprintf(stderr,
            "wrap_engram_gate: key_num_elements=%lld hc=%lld hidden=%lld\n",
            (long long)key_num_elements, (long long)hc, (long long)hidden);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "wrap_engram_gate: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  const int64_t rows = key_num_elements / hidden;
  RUNTIME_DEBUG_LOG("[REAL] wrap_engram_gate: rows=%lld, hc=%lld, "
                    "hidden=%lld, normed=%s, data_type=%s\n",
                    (long long)rows, (long long)hc, (long long)hidden,
                    normed ? "yes" : "no",
                    (element_size_bytes == 2) ? "f16" : "f32");

  return hip_engram_gate(hipdnn_ep_state_get_stream(state), key, query, value,
                         key_scale, query_scale, conv_scale, output, normed,
                         rows, hc, hidden, epsilon, hip_dtype);
}

int wrap_hyper_connection_pre_mix(RuntimeState *state, void *streams,
                                  void *pre_mix, void *output,
                                  int64_t stream_num_elements, int64_t branches,
                                  int64_t hidden, int64_t feature_gate,
                                  int64_t gate_sigmoid,
                                  int64_t element_size_bytes,
                                  float reduction_scale) {
  OP_PROFILE(
      "hyper_connection",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "pre %lldx%lld", (long long)branches,
                 (long long)hidden);
        return std::string(b);
      },
      state);
  if (!state || !streams || !pre_mix || !output) {
    fprintf(stderr, "Invalid arguments to wrap_hyper_connection_pre_mix\n");
    return -1;
  }
  int64_t span = branches * hidden;
  if (branches <= 0 || hidden <= 0 || span <= 0 ||
      stream_num_elements % span != 0) {
    fprintf(stderr,
            "wrap_hyper_connection_pre_mix: elements=%lld branches=%lld "
            "hidden=%lld\n",
            (long long)stream_num_elements, (long long)branches,
            (long long)hidden);
    return -1;
  }
  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "wrap_hyper_connection_pre_mix: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }
  return hip_hyper_connection_pre_mix(
      hipdnn_ep_state_get_stream(state), streams, pre_mix, output,
      stream_num_elements / span, branches, hidden,
      static_cast<int>(feature_gate), static_cast<int>(gate_sigmoid),
      reduction_scale, hip_dtype);
}

int wrap_hyper_connection_post_mix(RuntimeState *state, void *streams,
                                   void *block_output, void *post_mix,
                                   void *stream_mix, void *output,
                                   int64_t stream_num_elements,
                                   int64_t branches, int64_t hidden,
                                   int64_t feature_gate,
                                   int64_t element_size_bytes) {
  OP_PROFILE(
      "hyper_connection",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "post %lldx%lld", (long long)branches,
                 (long long)hidden);
        return std::string(b);
      },
      state);
  if (!state || !streams || !block_output || !post_mix || !output) {
    fprintf(stderr, "Invalid arguments to wrap_hyper_connection_post_mix\n");
    return -1;
  }
  int64_t span = branches * hidden;
  if (branches <= 0 || hidden <= 0 || span <= 0 ||
      stream_num_elements % span != 0) {
    fprintf(stderr,
            "wrap_hyper_connection_post_mix: elements=%lld branches=%lld "
            "hidden=%lld\n",
            (long long)stream_num_elements, (long long)branches,
            (long long)hidden);
    return -1;
  }
  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "wrap_hyper_connection_post_mix: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }
  return hip_hyper_connection_post_mix(
      hipdnn_ep_state_get_stream(state), streams, block_output, post_mix,
      stream_mix, output, stream_num_elements / span, branches, hidden,
      static_cast<int>(feature_gate), hip_dtype);
}

int wrap_hyper_connection_post_mix_norm(
    RuntimeState *state, void *streams, void *block_output, void *gate,
    void *stream_mix, void *norm_scale, void *residual, void *normed,
    int64_t stream_num_elements, int64_t branches, int64_t hidden,
    int64_t feature_gate, int64_t gate_activation, int64_t element_size_bytes,
    float epsilon) {
  OP_PROFILE(
      "hyper_connection",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "post_norm %lldx%lld", (long long)branches,
                 (long long)hidden);
        return std::string(b);
      },
      state);
  if (!state || !streams || !block_output || !gate || !norm_scale ||
      !residual || !normed) {
    fprintf(stderr,
            "Invalid arguments to wrap_hyper_connection_post_mix_norm\n");
    return -1;
  }
  int64_t span = branches * hidden;
  if (branches <= 0 || hidden <= 0 || span <= 0 ||
      stream_num_elements % span != 0) {
    fprintf(stderr,
            "wrap_hyper_connection_post_mix_norm: elements=%lld branches=%lld "
            "hidden=%lld\n",
            (long long)stream_num_elements, (long long)branches,
            (long long)hidden);
    return -1;
  }
  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "wrap_hyper_connection_post_mix_norm: unsupported element_size "
            "%lld (supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }
  return hip_hyper_connection_post_mix_norm(
      hipdnn_ep_state_get_stream(state), streams, block_output, gate,
      stream_mix, norm_scale, residual, normed, stream_num_elements / span,
      branches, hidden, static_cast<int>(feature_gate),
      static_cast<int>(gate_activation), epsilon, hip_dtype);
}
