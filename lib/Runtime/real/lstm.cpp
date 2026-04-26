/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip.lstm -- thin shim around miopenRNNForwardInference.
//
// Design choices:
//
// 1. Layout translation
//    ONNX gate order is IOFG, MIOpen LSTM gate order is IFOG (paramID/biasID
//    semantics from the MIOpen docs):
//        paramID 0/4 -> input gate    (i)
//        paramID 1/5 -> forget gate   (f)
//        paramID 2/6 -> output gate   (o)
//        paramID 3/7 -> cell gate     (g/c)
//    We translate per call by allocating a scratch weight buffer of the size
//    MIOpen reports via miopenGetRNNParamsSize, then issuing one
//    hipMemcpyAsync per (direction, gate) slab from the ONNX W/R/B tensors
//    into the offsets MIOpen reports via miopenGetRNNLayerParamOffset /
//    miopenGetRNNLayerBiasOffset.
//
//    We intentionally bypass miopenSetRNNLayerParam: its CopyTensor path
//    transposes each slab to column-major for non-zero offsets (via a
//    strided GPU kernel) but does a raw memcpy for offset-zero.  MIOpen's
//    forward GEMM reads the packed buffer as row-major, so the column-
//    major slabs at paramID > 0 produce wrong results.  Direct memcpy at
//    the correct byte offset keeps every slab in ONNX's native row-major
//    order, which is exactly what the GEMM expects.
//
//    Caching the permuted buffer across calls is left as a TODO -- Kokoro
//    hits 6 LSTMs per invocation so the per-call overhead (one hipMalloc +
//    ~64 hipMemcpyAsync + one hipFree) is on the order of tens of
//    microseconds, dominated by the MIOpen call itself.
//
// 2. Y output layout
//    ONNX Y is (seq_len, num_dir, batch, hidden).  MIOpen writes
//    (seq_len, batch, num_dir*hidden).  When batch == 1 these byte layouts
//    are identical and we let MIOpen write directly into y.  For batch > 1
//    we'd need a permute kernel; this wrapper rejects batch > 1 with a
//    clear error message until that path lands.
//
// 3. State buffers
//    initial_h / initial_c may be NULL (MIOpen treats NULL as zero
//    initialisation, matching ONNX's default).  hy / cy are always
//    populated since the op definition makes Y_h / Y_c required outputs.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)
#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

namespace {

// Map HIPDNN_EP_DATATYPE_* -> miopenDataType_t.  We only support the float
// flavours; integer LSTM is not a thing in MIOpen.
miopenDataType_t to_miopen_dtype(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    return static_cast<miopenDataType_t>(-1);
  }
}

int64_t element_size(int64_t data_type) {
  return hipdnn_ep_datatype_size(data_type);
}

// Build a packed 3D tensor descriptor with row-major strides.
miopenStatus_t build_3d_descriptor(miopenTensorDescriptor_t desc,
                                   miopenDataType_t dt, int d0, int d1,
                                   int d2) {
  int dims[3] = {d0, d1, d2};
  int strides[3] = {d1 * d2, d2, 1};
  return miopenSetTensorDescriptor(desc, dt, /*nbDims=*/3, dims, strides);
}

// ONNX gate index -> MIOpen gate index.
// ONNX order:   0=I, 1=O, 2=F, 3=G(cell)
// MIOpen order: 0=I, 1=F, 2=O, 3=G(cell)
constexpr int kOnnxToMiopenGate[4] = {0, 2, 1, 3};

} // anonymous namespace

extern "C" int wrap_miopenRNNForwardInference(
    RuntimeState *state, void *x, int64_t seq_len, int64_t batch,
    int64_t input_size, void *w, void *r, void *b, void *hx, void *cx,
    void *y, void *hy, void *cy, int64_t hidden_size, int64_t direction,
    int64_t data_type) {
  if (!state || !x || !w || !r || !y || !hy || !cy) {
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: required pointer is null\n");
    return -1;
  }
  if (seq_len <= 0 || batch <= 0 || input_size <= 0 || hidden_size <= 0) {
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: bad shape "
            "(seq=%lld batch=%lld in=%lld hidden=%lld)\n",
            (long long)seq_len, (long long)batch, (long long)input_size,
            (long long)hidden_size);
    return -1;
  }
  if (direction < 0 || direction > 2) {
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: invalid direction enum %lld "
            "(expected 0=forward, 1=reverse, 2=bidirectional)\n",
            (long long)direction);
    return -1;
  }

  miopenDataType_t mio_dtype = to_miopen_dtype(data_type);
  if (static_cast<int>(mio_dtype) < 0) {
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  if (batch > 1) {
    // The MIOpen Y layout is (seq, batch, num_dir*hidden); ONNX is
    // (seq, num_dir, batch, hidden).  Bytes match when batch == 1 only.
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: batch=%lld not yet supported "
            "(needs (s,b,d,h) -> (s,d,b,h) permute kernel)\n",
            (long long)batch);
    return -1;
  }

  const int num_dir = (direction == 2) ? 2 : 1;
  miopenRNNDirectionMode_t mio_dir =
      (direction == 2) ? miopenRNNbidirection : miopenRNNunidirection;
  const int num_layers = 1;
  const int total_layers = num_dir * num_layers;

  // Bias mode: ONNX semantics expect both Wb and Rb to be present (or the
  // single "no bias" case).  We always register withBias so MIOpen's
  // weight buffer reserves the bias section; if the caller passed b==NULL
  // we'll memset the bias slabs to zero.
  miopenRNNBiasMode_t bias_mode = miopenRNNwithBias;

  miopenHandle_t mio =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Always-on diagnostic line so we can see we're hitting the wrapper.
  fprintf(stderr,
          "[lstm] wrap_miopenRNNForwardInference: dtype=%s seq=%lld batch=%lld "
          "in=%lld hidden=%lld dir=%lld bias=%d hx=%p cx=%p\n",
          hipdnn_ep_datatype_name(data_type), (long long)seq_len,
          (long long)batch, (long long)input_size, (long long)hidden_size,
          (long long)direction, b ? 1 : 0, hx, cx);

  // Resource handles: zero-initialise so cleanup is safe on early bail-out.
  miopenRNNDescriptor_t rnn_desc = nullptr;
  miopenTensorDescriptor_t hx_desc = nullptr;
  miopenTensorDescriptor_t cx_desc = nullptr;
  miopenTensorDescriptor_t hy_desc = nullptr;
  miopenTensorDescriptor_t cy_desc = nullptr;
  miopenTensorDescriptor_t w_desc = nullptr;
  miopenTensorDescriptor_t *x_descs = nullptr;
  miopenTensorDescriptor_t *y_descs = nullptr;
  void *w_buf = nullptr;
  void *workspace = nullptr;
  int result = 0;
  int x_descs_created = 0;
  int y_descs_created = 0;
  size_t workspace_bytes = 0;
  size_t param_bytes = 0;

  MIOPEN_CHECK(miopenCreateRNNDescriptor(&rnn_desc));
  MIOPEN_CHECK(miopenSetRNNDescriptor(
      rnn_desc, /*hsize=*/static_cast<int>(hidden_size),
      /*nlayers=*/num_layers, miopenRNNlinear, mio_dir, miopenLSTM, bias_mode,
      miopenRNNdefault, mio_dtype));

  // hx/cx/hy/cy descriptors: (total_layers, batch, hidden_size).
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&hx_desc));
  MIOPEN_CHECK(build_3d_descriptor(hx_desc, mio_dtype, total_layers,
                                   static_cast<int>(batch),
                                   static_cast<int>(hidden_size)));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&cx_desc));
  MIOPEN_CHECK(build_3d_descriptor(cx_desc, mio_dtype, total_layers,
                                   static_cast<int>(batch),
                                   static_cast<int>(hidden_size)));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&hy_desc));
  MIOPEN_CHECK(build_3d_descriptor(hy_desc, mio_dtype, total_layers,
                                   static_cast<int>(batch),
                                   static_cast<int>(hidden_size)));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&cy_desc));
  MIOPEN_CHECK(build_3d_descriptor(cy_desc, mio_dtype, total_layers,
                                   static_cast<int>(batch),
                                   static_cast<int>(hidden_size)));

  // Per-time-step input/output descriptors.  All entries are identical:
  //   x: (batch, input_size, 1)
  //   y: (batch, num_dir*hidden_size, 1)
  // Allocated as raw arrays so we can pass the C-style pointer that
  // miopenRNNForwardInference expects.
  x_descs = static_cast<miopenTensorDescriptor_t *>(
      std::calloc(seq_len, sizeof(miopenTensorDescriptor_t)));
  y_descs = static_cast<miopenTensorDescriptor_t *>(
      std::calloc(seq_len, sizeof(miopenTensorDescriptor_t)));
  if (!x_descs || !y_descs) {
    fprintf(stderr, "wrap_miopenRNNForwardInference: oom for desc arrays\n");
    result = -1;
    goto cleanup;
  }
  for (int64_t s = 0; s < seq_len; ++s) {
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&x_descs[s]));
    ++x_descs_created;
    MIOPEN_CHECK(build_3d_descriptor(x_descs[s], mio_dtype,
                                     static_cast<int>(batch),
                                     static_cast<int>(input_size), 1));
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&y_descs[s]));
    ++y_descs_created;
    MIOPEN_CHECK(build_3d_descriptor(
        y_descs[s], mio_dtype, static_cast<int>(batch),
        static_cast<int>(num_dir * hidden_size), 1));
  }

  // Weight descriptor + buffer.
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&w_desc));
  MIOPEN_CHECK(miopenGetRNNParamsSize(mio, rnn_desc, x_descs[0], &param_bytes,
                                      mio_dtype));
  if (param_bytes == 0) {
    fprintf(stderr,
            "wrap_miopenRNNForwardInference: miopenGetRNNParamsSize returned "
            "0 -- impossible for an LSTM\n");
    result = -1;
    goto cleanup;
  }
  MIOPEN_CHECK(miopenGetRNNParamsDescriptor(mio, rnn_desc, x_descs[0], w_desc,
                                            mio_dtype));
  HIP_CHECK(hipMalloc(&w_buf, param_bytes));
  HIP_CHECK(hipMemsetAsync(w_buf, 0, param_bytes, stream));

  // Permute ONNX W (input GEMM), R (hidden GEMM), B (bias) into the MIOpen
  // weight blob.
  //
  // We use miopenGetRNNLayerParamOffset / miopenGetRNNLayerBiasOffset to
  // find each gate's byte position inside the packed buffer, then
  // hipMemcpyAsync the ONNX slab directly.  This avoids
  // miopenSetRNNLayerParam whose internal CopyTensor path transposes the
  // matrix for non-zero destination offsets (column-major storage) but not
  // for offset-zero (raw memcpy), producing an inconsistent layout that
  // the downstream GEMM (which reads row-major) cannot tolerate.
  {
    const int64_t elem = element_size(data_type);
    const int64_t w_slab_bytes = hidden_size * input_size * elem;
    const int64_t r_slab_bytes = hidden_size * hidden_size * elem;
    const int64_t b_slab_bytes = hidden_size * elem;
    const int64_t per_dir_w_bytes = 4 * w_slab_bytes;
    const int64_t per_dir_r_bytes = 4 * r_slab_bytes;
    const int64_t per_dir_b_bytes = 8 * b_slab_bytes;

    miopenTensorDescriptor_t param_desc = nullptr;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&param_desc));

    auto set_gate = [&](int layer, int onnx_gate, bool is_recurrent,
                        const void *src_dir_base) -> int {
      const int paramID =
          kOnnxToMiopenGate[onnx_gate] + (is_recurrent ? 4 : 0);
      const int64_t slab = is_recurrent ? r_slab_bytes : w_slab_bytes;
      const char *src = static_cast<const char *>(src_dir_base) +
                        onnx_gate * slab;

      size_t param_offset = 0;
      miopenStatus_t st = miopenGetRNNLayerParamOffset(
          rnn_desc, layer, x_descs[0], paramID, param_desc, &param_offset);
      if (st != miopenStatusSuccess) {
        fprintf(stderr,
                "miopenGetRNNLayerParamOffset failed: layer=%d paramID=%d "
                "status=%d\n",
                layer, paramID, st);
        return -1;
      }
      const size_t byte_offset = param_offset * static_cast<size_t>(elem);
      fprintf(stderr,
              "[lstm] set_gate layer=%d paramID=%d offset_elems=%zu "
              "slab_bytes=%lld\n",
              layer, paramID, param_offset, (long long)slab);
      hipError_t herr = hipMemcpyAsync(
          static_cast<char *>(w_buf) + byte_offset, src,
          static_cast<size_t>(slab), hipMemcpyDeviceToDevice, stream);
      if (herr != hipSuccess) {
        fprintf(stderr,
                "hipMemcpyAsync(param) failed: layer=%d paramID=%d err=%s\n",
                layer, paramID, hipGetErrorString(herr));
        return -1;
      }
      return 0;
    };

    auto set_bias = [&](int layer, int onnx_gate, bool is_recurrent,
                        const void *src_dir_base) -> int {
      const int biasID =
          kOnnxToMiopenGate[onnx_gate] + (is_recurrent ? 4 : 0);
      int onnx_bias_slot = onnx_gate + (is_recurrent ? 4 : 0);
      const char *src = static_cast<const char *>(src_dir_base) +
                        onnx_bias_slot * b_slab_bytes;

      size_t bias_offset = 0;
      miopenStatus_t st = miopenGetRNNLayerBiasOffset(
          rnn_desc, layer, x_descs[0], biasID, param_desc, &bias_offset);
      if (st != miopenStatusSuccess) {
        fprintf(stderr,
                "miopenGetRNNLayerBiasOffset failed: layer=%d biasID=%d "
                "status=%d\n",
                layer, biasID, st);
        return -1;
      }
      const size_t byte_offset = bias_offset * static_cast<size_t>(elem);
      hipError_t herr = hipMemcpyAsync(
          static_cast<char *>(w_buf) + byte_offset, src,
          static_cast<size_t>(b_slab_bytes), hipMemcpyDeviceToDevice, stream);
      if (herr != hipSuccess) {
        fprintf(stderr,
                "hipMemcpyAsync(bias) failed: layer=%d biasID=%d err=%s\n",
                layer, biasID, hipGetErrorString(herr));
        return -1;
      }
      return 0;
    };

    for (int d = 0; d < num_dir; ++d) {
      const char *w_dir =
          static_cast<const char *>(w) + d * per_dir_w_bytes;
      const char *r_dir =
          static_cast<const char *>(r) + d * per_dir_r_bytes;
      for (int g = 0; g < 4; ++g) {
        if (set_gate(d, g, /*is_recurrent=*/false, w_dir) != 0) {
          miopenDestroyTensorDescriptor(param_desc);
          result = -1;
          goto cleanup;
        }
        if (set_gate(d, g, /*is_recurrent=*/true, r_dir) != 0) {
          miopenDestroyTensorDescriptor(param_desc);
          result = -1;
          goto cleanup;
        }
      }
      if (b) {
        const char *b_dir =
            static_cast<const char *>(b) + d * per_dir_b_bytes;
        for (int g = 0; g < 4; ++g) {
          if (set_bias(d, g, /*is_recurrent=*/false, b_dir) != 0) {
            miopenDestroyTensorDescriptor(param_desc);
            result = -1;
            goto cleanup;
          }
          if (set_bias(d, g, /*is_recurrent=*/true, b_dir) != 0) {
            miopenDestroyTensorDescriptor(param_desc);
            result = -1;
            goto cleanup;
          }
        }
      }
    }
    miopenDestroyTensorDescriptor(param_desc);
  }

  // Workspace.
  MIOPEN_CHECK(miopenGetRNNWorkspaceSize(mio, rnn_desc,
                                         static_cast<int>(seq_len), x_descs,
                                         &workspace_bytes));
  if (workspace_bytes > 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
  }

  // Run inference.  hx / cx may legitimately be NULL (zero init).  hy / cy
  // are always populated (DPS init buffers from the caller).
  MIOPEN_CHECK(miopenRNNForwardInference(
      mio, rnn_desc, static_cast<int>(seq_len), x_descs, x, hx_desc, hx,
      cx_desc, cx, w_desc, w_buf, y_descs, y, hy_desc, hy, cy_desc, cy,
      workspace, workspace_bytes));

  int64_t y_count = seq_len * num_dir * batch * hidden_size;

  // No CPU-side post-processing; MIOpen's LSTM output is used as-is.

  nan_trace_check("lstm", y, y_count);

cleanup: {
  hipError_t err;
  if (workspace) {
    err = hipFree(workspace);
    if (err != hipSuccess)
      fprintf(stderr, "wrap_miopenRNNForwardInference: hipFree(workspace) "
                      "failed: %s\n",
              hipGetErrorString(err));
  }
  if (w_buf) {
    err = hipFree(w_buf);
    if (err != hipSuccess)
      fprintf(stderr,
              "wrap_miopenRNNForwardInference: hipFree(w_buf) failed: %s\n",
              hipGetErrorString(err));
  }
  for (int s = 0; s < x_descs_created; ++s)
    miopenDestroyTensorDescriptor(x_descs[s]);
  for (int s = 0; s < y_descs_created; ++s)
    miopenDestroyTensorDescriptor(y_descs[s]);
  std::free(x_descs);
  std::free(y_descs);
  if (w_desc)
    miopenDestroyTensorDescriptor(w_desc);
  if (hx_desc)
    miopenDestroyTensorDescriptor(hx_desc);
  if (cx_desc)
    miopenDestroyTensorDescriptor(cx_desc);
  if (hy_desc)
    miopenDestroyTensorDescriptor(hy_desc);
  if (cy_desc)
    miopenDestroyTensorDescriptor(cy_desc);
  if (rnn_desc)
    miopenDestroyRNNDescriptor(rnn_desc);
}
  return result;
}
