/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */

#include <glog/logging.h>
#if USE_VART_TRACE
#  include <vart/trace/event.hpp>
#  include <vart/trace/trace.hpp>
#  include <vitis/ai/profiling.hpp>
#endif
#include "vaip/vaip_ort.hpp"

namespace vaip_core {

/**
 * Collect trace event of vitis flow.
 *
 * Return trace data to onnxruntime.
 */
VAIP_DLL_SPEC void profiler_collect(std::vector<EventInfo>& api_events,
                                    std::vector<EventInfo>& kernel_events) {

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
