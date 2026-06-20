/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Per-thread "current session stream" slot, deliberately compiled NATIVELY and
// never into runtime.bc. The JIT cannot lower a `thread_local` defined in the
// bitcode runtime on Windows: codegen emits an emulated-TLS call to
// __emutls_get_address, which has no definition in the JIT process. Keeping the
// storage in host-native code sidesteps that entirely -- the JIT'd runtime
// calls these accessors as ordinary external functions and resolves them from
// the host process's exports (EP DLL / hip-test / hip-inspect), whose native
// thread_local works normally. Native model DLLs link their own copy.
#include "hipdnn_ep_runtime.h"

namespace {
thread_local void *g_current_stream = nullptr;
} // namespace

extern "C" HIPDNN_EP_RT_EXPORT void *hipdnn_ep_get_current_stream(void) {
  return g_current_stream;
}

extern "C" HIPDNN_EP_RT_EXPORT void hipdnn_ep_set_current_stream(void *stream) {
  g_current_stream = stream;
}
