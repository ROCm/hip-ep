/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdio>

extern "C" {

void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state) {
  return state ? static_cast<void *>(state->device_error_flag) : nullptr;
}

int hipdnn_ep_state_reset_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    std::fprintf(stderr, "hipdnn_ep_state_reset_error_flag: invalid state\n");
    return -1;
  }
  state->host_error_status = 0;
  hipError_t err =
      hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  return err == hipSuccess ? 0 : -1;
}

int hipdnn_ep_state_set_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    std::fprintf(stderr, "hipdnn_ep_state_set_error_flag: invalid state\n");
    return -1;
  }
  if (state->host_error_status == 0)
    state->host_error_status = -1;
  hipError_t err = hipMemsetAsync(state->device_error_flag, 0xff, sizeof(int),
                                  state->stream);
  return err == hipSuccess ? 0 : -1;
}

int hipdnn_ep_state_record_status(RuntimeState *state, int status) {
  if (status != 0 && state && state->host_error_status == 0)
    state->host_error_status = status;
  return status;
}

int hipdnn_ep_state_read_and_clear_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    std::fprintf(stderr,
                 "hipdnn_ep_state_read_and_clear_error_flag: invalid state\n");
    return -1;
  }

  int deviceError = 0;
  hipError_t err =
      hipMemcpyAsync(&deviceError, state->device_error_flag, sizeof(int),
                     hipMemcpyDeviceToHost, state->stream);
  if (err != hipSuccess)
    return -1;
  err = hipStreamSynchronize(state->stream);
  if (err != hipSuccess)
    return -1;
  err = hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  if (err != hipSuccess)
    return -1;

  int result =
      state->host_error_status != 0 ? state->host_error_status : deviceError;
  state->host_error_status = 0;
  return result;
}

} // extern "C"
