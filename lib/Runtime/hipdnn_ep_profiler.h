/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipdnn_ep_profiler.h - GPU Profiling Instrumentation ---------------===//
//
// Provides hipEvent-based GPU profiling for runtime wrapper functions.
//
// When HIPDNN_EP_ENABLE_PROFILING is defined at compile time AND the
// HIPDNN_EP_PROFILE environment variable is set at runtime, each
// PROFILE_BEGIN / PROFILE_END pair records GPU timestamps via hipEvents.
// On cleanup, results are dumped to JSON (Chrome Tracing), CSV, and a
// summary table on stderr.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_PROFILER_H
#define HIPDNN_EP_PROFILER_H

#include <stddef.h>

struct RuntimeState;

#ifdef HIPDNN_EP_ENABLE_PROFILING

void hipdnn_ep_profiler_init(RuntimeState *state);
void hipdnn_ep_profiler_begin(RuntimeState *state, const char *name,
                              const char *category);
void hipdnn_ep_profiler_end(RuntimeState *state, const char *name,
                            const char *category);
void hipdnn_ep_profiler_dump(RuntimeState *state);
void hipdnn_ep_profiler_cleanup(RuntimeState *state);

#define HIPDNN_EP_PROFILE_BEGIN(state, name, cat)                              \
  hipdnn_ep_profiler_begin((state), (name), (cat))
#define HIPDNN_EP_PROFILE_END(state, name, cat)                                \
  hipdnn_ep_profiler_end((state), (name), (cat))

#else /* HIPDNN_EP_ENABLE_PROFILING not defined */

static inline void hipdnn_ep_profiler_init(RuntimeState *) {}
static inline void hipdnn_ep_profiler_begin(RuntimeState *, const char *,
                                            const char *) {}
static inline void hipdnn_ep_profiler_end(RuntimeState *, const char *,
                                          const char *) {}
static inline void hipdnn_ep_profiler_dump(RuntimeState *) {}
static inline void hipdnn_ep_profiler_cleanup(RuntimeState *) {}

#define HIPDNN_EP_PROFILE_BEGIN(state, name, cat) ((void)0)
#define HIPDNN_EP_PROFILE_END(state, name, cat) ((void)0)

#endif /* HIPDNN_EP_ENABLE_PROFILING */

#endif /* HIPDNN_EP_PROFILER_H */
