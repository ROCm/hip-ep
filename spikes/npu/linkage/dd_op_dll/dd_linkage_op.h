/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// T0.2 spike (Phase 0, throwaway -- see docs/design/hybrid-npu-gpu-tasks.md).
// This header is the ONLY file the harness and this DLL both see --
// deliberately narrow, no C++ types, mirroring the production shim boundary
// described in docs/design/hybrid-npu-gpu-design.md's "Packaging and the shim
// boundary". This spike does not implement that boundary; it only smoke-tests
// that a boundary shaped like it can coexist with HIP + ORT + this repo's EP
// DLL in one process without heap corruption.

#include <stddef.h>

#if defined(_WIN32)
#if defined(DD_LINKAGE_OP_BUILD_DLL)
#define DD_LINKAGE_EXPORT __declspec(dllexport)
#else
#define DD_LINKAGE_EXPORT __declspec(dllimport)
#endif
#else
#define DD_LINKAGE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Shape of the one DD operator this spike runs: ryzenai::elw_mul<uint16_t,
// uint16_t, uint16_t> (DD's bfloat16 elementwise multiply). Picked over the
// originally-tried ryzenai::elw_add<uint16_t, ...> because a `dumpbin
// /exports` check found this particular prebuilt dyn_dispatch_core.dll does
// not export elw_add's explicit template instantiation (present in DD's
// source tree, absent from this binary) -- see ../README.md. Not chosen for
// relevance to the production op set: T0.3 recovers the real quantized-matmul
// call sequence this project actually needs.
#define DD_LINKAGE_ROWS 49
#define DD_LINKAGE_COLS 1024
#define DD_LINKAGE_ELEM_BYTES 2
#define DD_LINKAGE_OUTPUT_BYTES                                                \
  ((size_t)DD_LINKAGE_ROWS * (size_t)DD_LINKAGE_COLS *                         \
   (size_t)DD_LINKAGE_ELEM_BYTES)

enum dd_linkage_status {
  DD_LINKAGE_OK = 0,
  DD_LINKAGE_ERR_BAD_ARGS = 1,
  DD_LINKAGE_ERR_EXCEPTION = 2,
  DD_LINKAGE_ERR_UNKNOWN = 3,
};

// Runs one DD operator (construct -> set_params -> initialize_const_params ->
// execute) writing its output into `output_buffer` (caller-owned; must be at
// least DD_LINKAGE_OUTPUT_BYTES). `output_buffer` is expected to come from
// hip-ep's EP allocator -- that is the specific cross-toolchain path this
// spike exists to exercise, not a DD requirement.
//
// Never throws across this boundary: every exception is caught here and
// turned into a return code plus a message in `err_msg` (best-effort
// truncated to `err_msg_cap`, always NUL-terminated when err_msg_cap > 0).
// See the "Exceptions must not cross the shim ABI" rule.
DD_LINKAGE_EXPORT int dd_linkage_run_trivial_op(void *output_buffer,
                                                size_t output_buffer_bytes,
                                                char *err_msg,
                                                size_t err_msg_cap);

#ifdef __cplusplus
}
#endif
