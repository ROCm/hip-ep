/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* hip/Runtime/PluginAPI.h is a C header: plugins written in C include it too.
 * Compiling this file as C checks that; it takes the address of every entry
 * point so a declaration that drifts from its definition fails to link. */

#include "hip/Runtime/PluginAPI.h"

typedef void (*entry_point)(void);

int plugin_api_c_entry_points(void) {
  const entry_point all[] = {
      (entry_point)hipdnn_ep_state_get_stream,
      (entry_point)hipdnn_ep_state_get_workspace,
      (entry_point)hipdnn_ep_state_get_workspace_size,
      (entry_point)hipdnn_ep_state_ensure_workspace,
      (entry_point)hipdnn_ep_state_get_matmul_dp4a_scratch,
      (entry_point)hipdnn_ep_state_get_matmul_dp4a_scratch_size,
      (entry_point)hipdnn_ep_state_ensure_matmul_dp4a_scratch,
      (entry_point)hipdnn_ep_state_get_error_flag_device_ptr,
      (entry_point)hipdnn_ep_state_reset_error_flag,
      (entry_point)hipdnn_ep_state_read_and_clear_error_flag,
  };
  return (int)(sizeof(all) / sizeof(all[0]));
}
