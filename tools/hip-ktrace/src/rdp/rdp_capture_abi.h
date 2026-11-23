/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

// Windows x64 ABI declarations shared by the API 1.0.0 and 2.0.0 profiling
// clients. This file does not include the original SDK header.

#include <array>
#include <cstddef>
#include <cstdint>

namespace rdp_abi {
using Result = std::int32_t;
using Feature = std::int32_t;
using GpuApi = std::int32_t;
using LogLevel = std::int32_t;
using Context = void *;
using ConnectionId = void *;

constexpr Result kSuccess = 0;
constexpr Feature kProfilingFeature = 1;
constexpr std::uint32_t kCaptureFrame = 1;
constexpr std::uint32_t kCaptureDispatch = 2;
constexpr std::uint32_t kShaderInstrumentation = 4;
constexpr std::uint32_t kInstructionTracing = 1;
constexpr std::uint32_t kCounterCollection = 2;

struct ProcessInfo {
  char process_path[1024];
  std::uint32_t process_id;
};

struct AppFilter {
  std::uint8_t(__cdecl *filter)(void *, const ProcessInfo *, GpuApi);
  void *user_data;
};

struct LogCallback {
  void(__cdecl *log)(void *, LogLevel, const char *, std::uint32_t,
                     const char *);
  void *user_data;
};

struct RemoteConnectionInfo {
  const char *hostname;
  std::uint16_t port;
};

struct ContextInitParams {
  AppFilter app_filter;
  LogCallback log_callback;
  RemoteConnectionInfo remote_connection;
};

struct TraceFinishedCallback {
  void(__cdecl *trace_finished)(void *, Feature, ConnectionId, Result,
                                std::uint64_t, const std::uint8_t *);
  void *user_data;
};

struct StatusCallback {
  void(__cdecl *status_changed)(void *, Feature, std::int32_t, std::int32_t);
  void *user_data;
};

struct ProgressInfo;

struct ProgressCallback {
  void(__cdecl *progress_updated)(void *, const ProgressInfo *);
  void *user_data;
};

struct ProfilingEnableParams {
  std::uint32_t flags;
  std::uint32_t frame_capture_index;
  std::uint32_t dispatch_start_index;
  std::uint32_t dispatch_count;
  std::uint32_t dispatch_capture_delay_ms;
};

union FeatureEnableBody {
  ProfilingEnableParams profiling;
  std::array<std::uint8_t, 24> reserved;
};

struct FeatureEnableParams {
  Feature feature;
  TraceFinishedCallback trace_finished_callback;
  StatusCallback status_callback;
  ProgressCallback progress_callback;
  FeatureEnableBody body;
};

struct ProfilingParams {
  std::uint32_t flags;
  std::int32_t sqtt_buffer_size;
  std::uint64_t begin_frame_terminator_tag;
  std::uint64_t end_frame_terminator_tag;
  const char *begin_frame_terminator_string;
  const char *end_frame_terminator_string;
  std::int32_t capture_mode;
  std::uint32_t render_op_count;
};

constexpr std::size_t kFunctionSlotCapacity = 64;
constexpr std::size_t kInitializeSlot = 0;
constexpr std::size_t kEnableFeatureSlot = 1;
constexpr std::size_t kDestroySlot = 3;
constexpr std::size_t kProfilingDefaultsSlot = 14;
constexpr std::size_t kProfilingSetParamsSlot = 15;
constexpr std::size_t kProfilingBeginSlot = 16;
constexpr std::size_t kProfilingAbortSlot = 17;

struct FunctionTable {
  std::array<std::uintptr_t, kFunctionSlotCapacity> slots{};
};

using GetFnTableFn = Result(__cdecl *)(std::uint32_t, std::uint32_t,
                                       std::uint32_t, FunctionTable *);
using InitializeFn = Result(__cdecl *)(const ContextInitParams *, Context *);
using EnableFeatureFn = Result(__cdecl *)(Context, const FeatureEnableParams *);
using DestroyFn = void(__cdecl *)(Context);
using GetDefaultProfilingParamsFn = void(__cdecl *)(void *);
using SetProfilingParamsFn = void(__cdecl *)(Context, const void *);
using BeginTraceFn = Result(__cdecl *)(Context, ConnectionId);
using AbortTraceFn = Result(__cdecl *)(Context);

template <typename Function>
Function FunctionAt(const FunctionTable &table, std::size_t slot) {
  return reinterpret_cast<Function>(table.slots[slot]);
}

static_assert(sizeof(void *) == 8, "rdp_cli supports only 64-bit Windows.");
static_assert(sizeof(ProcessInfo) == 1028, "Unexpected ProcessInfo ABI.");
static_assert(sizeof(ContextInitParams) == 48,
              "Unexpected ContextInitParams ABI.");
static_assert(sizeof(ProfilingEnableParams) == 20,
              "Unexpected profiling enable ABI.");
static_assert(offsetof(FeatureEnableParams, body) == 56,
              "Unexpected feature body offset.");
static_assert(sizeof(FeatureEnableParams) == 80,
              "Unexpected feature enable ABI.");
static_assert(sizeof(ProfilingParams) == 48, "Unexpected profiling ABI.");
} // namespace rdp_abi
