/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "roctx.h"
#include "roctracer_roctx.h"

#include <atomic>
#include <cstdint>

namespace {
enum ActivityDomain : std::uint32_t {
  kActivityDomainRoctx = 6,
};

using TracerCallback = int (*)(std::uint32_t domain, std::uint32_t operation_id,
                               void *data);

std::atomic<TracerCallback> g_report_activity{nullptr};
std::atomic<roctx_range_id_t> g_next_range_id{1};
thread_local int g_nested_range_level = 0;

void ReportActivity(roctx_api_id_t operation_id, const char *message = nullptr,
                    roctx_range_id_t id = 0) {
  const TracerCallback callback =
      g_report_activity.load(std::memory_order_relaxed);
  if (!callback) {
    return;
  }

  roctx_api_data_t data{};
  switch (operation_id) {
  case ROCTX_API_ID_roctxMarkA:
    data.args.roctxMarkA.message = message;
    break;
  case ROCTX_API_ID_roctxRangePushA:
    data.args.roctxRangePushA.message = message;
    break;
  case ROCTX_API_ID_roctxRangePop:
    break;
  case ROCTX_API_ID_roctxRangeStartA:
    data.args.roctxRangeStartA.message = message;
    data.args.roctxRangeStartA.id = id;
    break;
  case ROCTX_API_ID_roctxRangeStop:
    data.args.roctxRangeStop.id = id;
    break;
  default:
    return;
  }
  callback(kActivityDomainRoctx, static_cast<std::uint32_t>(operation_id),
           &data);
}
} // namespace

ROCTX_API std::uint32_t roctx_version_major() { return ROCTX_VERSION_MAJOR; }

ROCTX_API std::uint32_t roctx_version_minor() { return ROCTX_VERSION_MINOR; }

ROCTX_API void roctxMarkA(const char *message) {
  ReportActivity(ROCTX_API_ID_roctxMarkA, message);
}

ROCTX_API int roctxRangePushA(const char *message) {
  ReportActivity(ROCTX_API_ID_roctxRangePushA, message);
  return g_nested_range_level++;
}

ROCTX_API int roctxRangePop() {
  ReportActivity(ROCTX_API_ID_roctxRangePop);
  if (g_nested_range_level == 0) {
    return -1;
  }
  return --g_nested_range_level;
}

ROCTX_API roctx_range_id_t roctxRangeStartA(const char *message) {
  const roctx_range_id_t id =
      g_next_range_id.fetch_add(1, std::memory_order_relaxed);
  ReportActivity(ROCTX_API_ID_roctxRangeStartA, message, id);
  return id;
}

ROCTX_API void roctxRangeStop(roctx_range_id_t id) {
  ReportActivity(ROCTX_API_ID_roctxRangeStop, nullptr, id);
}

extern "C" ROCTX_EXPORT void
roctxRegisterTracerCallback(TracerCallback callback) {
  g_report_activity.store(callback, std::memory_order_relaxed);
}
