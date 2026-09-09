/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

// RocMLIR dispatch runtime wrapper (hip.rocmlir).
//
// The generated IR embeds a pre-compiled GPU kernel (ELF/HSACO blob) in
// `kernel_binary`, stages the operand data pointers (inputs first, then output)
// into `kernargs`, and passes the launch geometry (`block_size`, `grid_size`)
// derived from the tuned perfConfig.
//
// Empty for now: a real implementation would hipModuleLoadData(kernel_binary),
// hipModuleGetFunction(func_name), and hipModuleLaunchKernel with the kernargs
// buffer. Left as a stub (mirrors the mock wrapper) until the launch path is
// wired up.
int wrap_rocmlir(RuntimeState *state, const char *kernel_binary,
                 char *func_name, int64_t block_size, int64_t grid_size,
                 void *kernargs, size_t size) {
  (void)kernel_binary;
  (void)func_name;
  (void)block_size;
  (void)grid_size;
  (void)kernargs;
  (void)size;
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_rocmlir\n");
    return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_rocmlir(func=%s, block_size=%lld, "
                    "grid_size=%lld, kernargs_size=%zu)\n",
                    func_name ? func_name : "(null)", (long long)block_size,
                    (long long)grid_size, size);
  return 0;
}
