/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Stub implementations for runtime symbols whose dialect lowerings exist
// but whose real GPU kernels haven't been implemented yet.  Without
// these stubs the model DLL fails to link with "undefined symbol", even
// though the surviving dataflow paths through these ops are
// dead/unreachable for the unit-test inputs we care about.
//
// Each stub:
//   - logs a one-time warning (so we know if a model actually needs the
//     missing path),
//   - touches the output buffer in a benign way (zero / pass-through),
//   - returns 0 to satisfy the runtime-driver contract.
//
// Replace each stub with a real implementation as it becomes a
// correctness blocker for a model we want to ship.

#include "../hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <hip/hip_runtime.h>

namespace {

void warn_once(const char *name) {
  static const char *seen[32] = {nullptr};
  for (int i = 0; i < 32; ++i) {
    if (seen[i] == nullptr) {
      seen[i] = name;
      fprintf(stderr,
              "[hipdnn_ep] WARNING: stub runtime symbol called: %s "
              "(returns zero / pass-through)\n",
              name);
      return;
    }
    if (seen[i] == name)
      return;
  }
}

int zero_fill_dev(void *stream, void *ptr, size_t bytes) {
  if (!ptr || bytes == 0)
    return 0;
  return static_cast<int>(hipMemsetAsync(
      ptr, 0, bytes, static_cast<hipStream_t>(stream)));
}

} // namespace

extern "C" {

// MIOpen softmax (used by ActivationLowering for hip.miopen_softmax).
// Signature per ActivationLowering: (handle, input, output, rows, cols).
// Real impl would call miopenSoftmaxForward; for now we zero-fill the
// output and warn.
__declspec(dllexport) int hip_miopen_softmax(void *handle, const void *input,
                                              void *output, int64_t rows,
                                              int64_t cols) {
  (void)handle;
  (void)input;
  warn_once("hip_miopen_softmax");
  // Without an MIOpen-real path, we still need the output buffer in a
  // defined state.  Assume f32 element size for the dead-code path.
  return zero_fill_dev(/*stream=*/nullptr, output,
                        static_cast<size_t>(rows * cols) * sizeof(float));
}

// hip.transpose legacy rank<=3 kernel -- never had an implementation.
// New TransposeOpLowering routes everything through hip_transpose_nd
// (in transpose_kernel.hip), so this stub is only here as a link-time
// fallback in case any not-yet-rebuilt model DLL still references it.
__declspec(dllexport) void hip_transpose(void *handle, const void *input,
                                          void *output, intptr_t rank,
                                          intptr_t dim0, intptr_t dim1,
                                          intptr_t s0, intptr_t s1,
                                          intptr_t s2) {
  (void)handle;
  (void)input;
  (void)dim0;
  (void)dim1;
  warn_once("hip_transpose (legacy; rebuild the EP)");
  size_t total = static_cast<size_t>(s0) * static_cast<size_t>(s1) *
                  static_cast<size_t>(s2);
  if (rank < 3)
    total = 1;
  // Best-effort element-size guess: 4 bytes (f32).
  zero_fill_dev(nullptr, output, total * 4);
}

// STFT helper kernels.  StftLowering emits two device-side helper
// calls (frame_window and split_complex); both will be replaced when
// the real rocFFT path is wired up.
__declspec(dllexport) int hip_stft_frame_window(void *stream, const void *input,
                                                 const void *window,
                                                 void *output,
                                                 int64_t batch,
                                                 int64_t n_frames,
                                                 int64_t frame_length,
                                                 int64_t frame_step,
                                                 int hip_dtype) {
  (void)input;
  (void)window;
  (void)hip_dtype;
  warn_once("hip_stft_frame_window");
  size_t bytes = static_cast<size_t>(batch * n_frames * frame_length) * 4;
  return zero_fill_dev(stream, output, bytes);
}

__declspec(dllexport) int hip_stft_split_complex(void *stream,
                                                  const void *complex_in,
                                                  void *real_out,
                                                  void *imag_out,
                                                  int64_t total_complex,
                                                  int hip_dtype) {
  (void)complex_in;
  (void)hip_dtype;
  warn_once("hip_stft_split_complex");
  size_t bytes = static_cast<size_t>(total_complex) * 4;
  int rc1 = zero_fill_dev(stream, real_out, bytes);
  int rc2 = zero_fill_dev(stream, imag_out, bytes);
  return rc1 != 0 ? rc1 : rc2;
}

} // extern "C"
