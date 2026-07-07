/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>
#include <morphizen-utils/morphizen-utils.hpp>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_PROFILER_COLLECT, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_PROFILER_COLLECT) >= n)
#include <morphizen/morphizen.hpp>

namespace morphizen {

/**
 * Collect trace event of vitis flow.
 *
 * Return trace data to onnxruntime.
 */
typedef void (*profiler_collect_t)(std::vector<EventInfo> &api_events,
                                   std::vector<EventInfo> &kernel_events);

MORPHIZEN_DLL_SPEC void
profiler_collect(std::vector<EventInfo> &api_events,
                 std::vector<EventInfo> &kernel_events) {

  auto profiler_collect_ptrs =
      morphizen::Plugin::get_all_symbols("profiler_collect_real");

  for (const auto &profiler_collect_ptr : profiler_collect_ptrs) {
    MY_LOG(1) << " running " << profiler_collect_ptr.first
              << "::profiler_collect_real";
    auto profiler_collect_func =
        reinterpret_cast<profiler_collect_t>(profiler_collect_ptr.second);
    profiler_collect_func(api_events, kernel_events);
  }
}
} // namespace morphizen
