/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <string>
#include "ut_timing_context.h"

// Optional build-time SDK headers, optional run-time driver DLL. The timer
// builds and runs without either. Never link or distribute the ADLX runtime.
#if defined(_WIN32) && __has_include(<ADLX.h>) && __has_include(<IPerformanceMonitoring.h>)
#define HIPDNN_UT_HAS_ADLX 1
#include <ADLX.h>
#include <ISystem.h>
#include <IPerformanceMonitoring.h>
#endif

namespace hipdnn_ep_test {
namespace timing {

class GpuFrequency {
 public:
  GpuFrequency() {
#ifdef HIPDNN_UT_HAS_ADLX
    // A multi-GPU process needs an explicit HIP/ADLX device mapping before
    // it can safely attach telemetry. Do not silently sample a different GPU.
    auto& session = adlxSession();
    available_ = session.available;
    if (std::getenv("HIPDNN_UT_ADLX") && std::string(std::getenv("HIPDNN_UT_ADLX")) == "0") {
      available_ = false; status_ = "disabled by HIPDNN_UT_ADLX=0"; return;
    }
    status_ = session.status;
    reference_mhz_ = session.reference_mhz;
#else
    status_ = "ADLX SDK headers unavailable or non-Windows build";
#endif
  }

  // Called AFTER event synchronization/timestamp consumption, at most four
  // times a second. Neither ADLX queries nor their host overhead are timed.
  void sample() {
#ifdef HIPDNN_UT_HAS_ADLX
    const auto start = std::chrono::steady_clock::now();
    if (!available_ || (samples_ && start - last_ < std::chrono::milliseconds(250))) return;
    last_ = start;
    adlx::IADLXGPUMetrics* metrics = nullptr;
    auto& session = adlxSession();
    const ADLX_RESULT rc = session.perf->GetCurrentGPUMetrics(session.gpu, &metrics);
    if (ADLX_SUCCEEDED(rc) && metrics) {
      adlx_int clock = 0;
      if (ADLX_SUCCEEDED(metrics->GPUClockSpeed(&clock)) && clock > 0) {
        sum_mhz_ += clock;
        minimum_mhz_ = samples_ ? std::min(minimum_mhz_, int(clock)) : int(clock);
        peak_mhz_ = std::max(peak_mhz_, int(clock));
        ++samples_;
      } else { ++failed_samples_; }
      adlx_double temp = 0.0, power = 0.0;
      if (session.temperature && ADLX_SUCCEEDED(metrics->GPUTemperature(&temp))) {
        sum_temperature_ += temp; ++temperature_samples_;
      }
      if (session.power && ADLX_SUCCEEDED(metrics->GPUPower(&power))) {
        sum_power_ += power; ++power_samples_;
      }
      metrics->Release();
    } else {
      ++failed_samples_;
    }
    sampling_wall_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
#endif
  }

  std::string json() const {
    if (!samples_) return "null";
    const int reference = reference_mhz_ > 0 ? reference_mhz_ : peak_mhz_;
    const char* basis = reference_mhz_ > 0 ? "sdk_range" : "observed_peak";
    char out[768], temperature[64] = "null", power[64] = "null";
    if (temperature_samples_) std::snprintf(temperature, sizeof(temperature), "%.17g", sum_temperature_ / temperature_samples_);
    if (power_samples_) std::snprintf(power, sizeof(power), "%.17g", sum_power_ / power_samples_);
    std::snprintf(out, sizeof(out),
        "{\"source\":\"ADLX\",\"samples\":%u,\"average_mhz\":%.17g,"
        "\"minimum_mhz\":%d,\"peak_mhz\":%d,\"reference_mhz\":%d,\"reference_basis\":\"%s\","
        "\"below_75_percent\":%s,\"failed_samples\":%u,\"sampling_wall_ms\":%.17g,\"average_temperature_c\":%s,\"average_power_w\":%s}",
        samples_, sum_mhz_ / samples_, minimum_mhz_, peak_mhz_, reference, basis,
        reference > 0 && minimum_mhz_ < reference * 0.75 ? "true" : "false",
        failed_samples_, sampling_wall_ms_, temperature, power);
    return out;
  }
  const std::string& status() const { return status_; }

 private:
#ifdef HIPDNN_UT_HAS_ADLX
  struct AdlxSession {
    HMODULE dll = nullptr;
    ADLXTerminate_Fn terminate = nullptr;
    adlx::IADLXPerformanceMonitoringServices* perf = nullptr;
    adlx::IADLXGPU* gpu = nullptr;
    bool available = false, initialized = false, temperature = false, power = false;
    int reference_mhz = 0;
    std::string status = "ADLX unavailable";
    AdlxSession() {
      dll = LoadLibraryExW(L"amdadlx64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
      if (!dll) { status = "ADLX load error " + std::to_string(GetLastError()); return; }
      auto initialize = reinterpret_cast<ADLXInitialize_Fn>(GetProcAddress(dll, ADLX_INIT_FUNCTION_NAME));
      terminate = reinterpret_cast<ADLXTerminate_Fn>(GetProcAddress(dll, ADLX_TERMINATE_FUNCTION_NAME));
      if (!initialize || !terminate) { status = "ADLX exports unavailable"; return; }
      adlx::IADLXSystem* system = nullptr;
      const ADLX_RESULT rc = initialize(ADLX_FULL_VERSION, &system);
      if (!ADLX_SUCCEEDED(rc) || !system) {
        status = "ADLX initialize result " + std::to_string(int(rc)); return;
      }
      initialized = true;
      adlx::IADLXGPUList* list = nullptr;
      if (!ADLX_SUCCEEDED(system->GetGPUs(&list)) || !list) return;
      if (list->Size() == 1 && ADLX_SUCCEEDED(list->At(list->Begin(), &gpu)) && gpu &&
          ADLX_SUCCEEDED(system->GetPerformanceMonitoringServices(&perf)) && perf) {
        adlx::IADLXGPUMetricsSupport* support = nullptr;
        if (ADLX_SUCCEEDED(perf->GetSupportedGPUMetrics(gpu, &support)) && support) {
          adlx_bool clock = false;
          available = ADLX_SUCCEEDED(support->IsSupportedGPUClockSpeed(&clock)) && clock;
          (void)support->IsSupportedGPUTemperature(&temperature);
          (void)support->IsSupportedGPUPower(&power);
          adlx_int minimum = 0, maximum = 0;
          if (ADLX_SUCCEEDED(support->GetGPUClockSpeedRange(&minimum, &maximum))) reference_mhz = maximum;
          support->Release();
          status = available ? "available" : "ADLX GPU clock unsupported";
        }
      } else {
        status = "ADLX requires exactly one GPU and supported performance services";
      }
      list->Release();
    }
    ~AdlxSession() {
      if (perf) perf->Release();
      if (gpu) gpu->Release();
      if (initialized && terminate) (void)terminate();
      if (dll) FreeLibrary(dll);
    }
  };
  static AdlxSession& adlxSession() { static AdlxSession session; return session; }
#endif
  bool available_ = false;
  std::string status_;
  uint32_t samples_ = 0, failed_samples_ = 0;
  uint32_t temperature_samples_ = 0, power_samples_ = 0;
  double sum_temperature_ = 0.0, sum_power_ = 0.0;
  double sum_mhz_ = 0.0, sampling_wall_ms_ = 0.0;
  int minimum_mhz_ = 0, peak_mhz_ = 0, reference_mhz_ = 0;
  std::chrono::steady_clock::time_point last_{};
};

}  // namespace timing
}  // namespace hipdnn_ep_test
