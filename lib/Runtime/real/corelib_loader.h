/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- corelib_loader.h - The one ryzenai_corelib loader ------------------===//
//
// Both CoreLib paths, the per-op wrappers (ryzenaicorelib.cpp) and whole
// models (ryzenaicorelib_model.cpp), reach CoreLib through this table. The
// library is loaded once per process at runtime rather than linked, so the EP
// still loads on machines without it, and one count of live users decides
// when ryzenai_corelib_cleanup runs.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_CORELIB_LOADER_H
#define HIPDNN_EP_CORELIB_LOADER_H

// The C headers only: <ryzenai/corelib.h> also pulls in the C++20 wrappers.
#include <ryzenai/corelib/base.h>
#include <ryzenai/corelib/model.h>
#include <ryzenai/corelib/ops/matmul_bf16.h>
#include <ryzenai/corelib/ops/rmsnorm_bf16.h>
#include <ryzenai/corelib/stream.h>
#include <ryzenai/corelib/tensor.h>
#include <ryzenai/corelib/weights.h>

#include <string>

#define HIPDNN_EP_CORELIB_FUNCTIONS(X)                                         \
  X(ryzenai_corelib_get_version)                                               \
  X(ryzenai_corelib_cleanup)                                                   \
  X(ryzenai_corelib_npu_usable)                                                \
  X(ryzenai_corelib_get_last_error_message)                                    \
  X(ryzenai_corelib_object_release)                                            \
  X(ryzenai_corelib_create_stream)                                             \
  X(ryzenai_corelib_stream_use_pdis)                                           \
  X(ryzenai_corelib_stream_set_enabled_backends)                               \
  X(ryzenai_corelib_stream_npu_usable)                                         \
  X(ryzenai_corelib_stream_synchronize)                                        \
  X(ryzenai_corelib_stream_get_kernels_root)                                   \
  X(ryzenai_corelib_stream_enable_kernel_bundle)                               \
  X(ryzenai_corelib_create_device_tensor)                                      \
  X(ryzenai_corelib_tensor_write)                                              \
  X(ryzenai_corelib_tensor_read)                                               \
  X(ryzenai_corelib_matmul_bf16_weights_create)                                \
  X(ryzenai_corelib_matmul_bf16_enum_kernels)                                  \
  X(ryzenai_corelib_matmul_bf16)                                               \
  X(ryzenai_corelib_rmsnorm_bf16_weights_create)                               \
  X(ryzenai_corelib_rmsnorm_bf16_enum_kernels)                                 \
  X(ryzenai_corelib_rmsnorm_bf16)                                              \
  X(ryzenai_corelib_add_rmsnorm_bf16_enum_kernels)                             \
  X(ryzenai_corelib_add_rmsnorm_bf16)                                          \
  X(ryzenai_corelib_create_model_from_path)                                    \
  X(ryzenai_corelib_model_create_session)                                      \
  X(ryzenai_corelib_model_get_metadata)                                        \
  X(ryzenai_corelib_model_run)

namespace hipdnn_ep::corelib {

struct CorelibApi {
#define HIPDNN_EP_CORELIB_MEMBER(name) decltype(&::name) name = nullptr;
  HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_MEMBER)
#undef HIPDNN_EP_CORELIB_MEMBER
};

// ryzenai_corelib from RYZENAICORELIB_ROOT, else the default search path, with
// its version checked against the headers; nullptr when unusable, after one
// `[corelib] ...` line on stderr saying why. Never unloaded.
const CorelibApi *load_api();

// A CoreLib stream or model session starts or ends. ryzenai_corelib_cleanup
// must run before process exit rather than at DLL unload under the loader
// lock, so the last user to go calls it.
void add_user();
void remove_user(const CorelibApi &api);

// std::getenv cannot see the process environment from the static-CRT EP DLL.
std::string read_env(const char *name);

} // namespace hipdnn_ep::corelib

#endif // HIPDNN_EP_CORELIB_LOADER_H
