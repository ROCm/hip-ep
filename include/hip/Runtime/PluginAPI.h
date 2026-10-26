/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_RUNTIME_PLUGIN_API_H
#define HIP_RUNTIME_PLUGIN_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque per-session execution state owned by the HIP-EP runtime.
typedef struct RuntimeState RuntimeState;

/// Returns the borrowed session stream as an opaque pointer, or NULL.
void *hipdnn_ep_state_get_stream(RuntimeState *state);

/// Returns the borrowed general workspace pointer and its current capacity.
/// A successful ensure call guarantees at least `needed_size` bytes. Storage
/// never shrinks, but a later growing ensure call may invalidate the pointer.
void *hipdnn_ep_state_get_workspace(RuntimeState *state);
size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state);
int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size);

/// Returns the borrowed DP4A matmul scratch pointer and its current capacity.
/// Its allocation and lifetime rules match the general workspace contract.
void *hipdnn_ep_state_get_matmul_dp4a_scratch(RuntimeState *state);
size_t hipdnn_ep_state_get_matmul_dp4a_scratch_size(RuntimeState *state);
int hipdnn_ep_state_ensure_matmul_dp4a_scratch(RuntimeState *state,
                                               size_t needed_size);

/// Accesses the session-owned device error flag. Reset is stream-ordered.
/// Read-and-clear synchronizes the session stream and returns the observed
/// device value: 0 after clearing the flag, or the nonzero error code, which
/// it leaves set until the next reset. Returns -1 when the state or a device
/// operation is invalid.
void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state);
int hipdnn_ep_state_reset_error_flag(RuntimeState *state);
int hipdnn_ep_state_read_and_clear_error_flag(RuntimeState *state);

#ifdef __cplusplus
}
#endif

#endif // HIP_RUNTIME_PLUGIN_API_H
