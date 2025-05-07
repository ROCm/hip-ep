/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <glog/logging.h>
#if USE_VART_TRACE
#  include <vart/trace/event.hpp>
#  include <vart/trace/trace.hpp>
#  include <vitis/ai/profiling.hpp>
#endif
#include "morphizen/vaip_ort.hpp"

namespace vaip_core {

/**
 * Collect trace event of vitis flow.
 *
 * Return trace data to onnxruntime.
 */
VAIP_DLL_SPEC void
profiler_collect([[maybe_unused]] std::vector<EventInfo>& api_events,
                 [[maybe_unused]] std::vector<EventInfo>& kernel_events) {

#if USE_VART_TRACE
  uint64_t start_time = 0;
  uint64_t stop_time = 0;
  std::unordered_map<std::string, std::string> event_args;

  auto ret1 = vitis::ai::trace::trace_collect();
  auto ret2 = vitis::ai::profiling::profiling_collect();

  /* Return trace data from add_trace() into onnxruntime events */
  for (auto& entry : ret1) {

    if (entry[std::string("classname")] == "user-task") {

      if (entry["event_state"] == "1") {
        start_time = stoll(entry["ts"]);
      } else if (entry["event_state"] == "0") {
        stop_time = stoll(entry["ts"]);

        api_events.emplace_back(std::make_tuple(
            entry["event_name"], stoi(entry["pid"]), stoi(entry["tid"]),
            start_time, stop_time - start_time));
      }
    }
  }

  /* Return trace data from __TIC__/__TOC__ into onnxruntime events */
  for (auto& entry : ret2) {
    api_events.emplace_back(std::make_tuple(entry.tag, entry.pid, entry.tid,
                                            entry.timestamp, entry.duration));
  }
#endif
}

} // namespace vaip_core
