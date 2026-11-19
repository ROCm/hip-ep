/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace ktrace::ipc {
constexpr std::uint32_t kMagic = 0x4352544B; // KTRC
constexpr std::uint16_t kAbiVersion = 2;
constexpr std::uint32_t kRingCapacity = 262144;

enum class RecordType : std::uint16_t {
  op = 1,
  api = 2,
  roctx = 3,
  vram = 4,
  kernel = 5,
  device = 6,
};

struct OpPayload {
  std::uint64_t begin_ns;
  std::uint64_t end_ns;
  std::uint64_t bytes;
  std::uint64_t queue_id;
  std::uint64_t correlation_id;
  std::uint32_t op;
  std::uint32_t kind;
  std::int32_t device_id;
  char kernel_name[256];
};

struct ApiPayload {
  std::uint64_t correlation_id;
  std::uint64_t enter_ns;
  std::uint64_t exit_ns;
  std::uint32_t api_id;
  char api_name[96];
  char arguments[160];
};

struct RoctxPayload {
  std::uint64_t timestamp_ns;
  std::uint64_t range_id;
  std::uint32_t operation_id;
  char message[256];
};

struct VramPayload {
  std::uint64_t timestamp_ns;
  std::uint64_t free_bytes;
  std::uint64_t total_bytes;
};

struct KernelPayload {
  std::uint64_t func_addr;
  std::uint64_t scratch;
  std::uint64_t static_lds;
  std::uint64_t dynamic_lds;
  std::int32_t vgpr;
  std::int32_t max_threads;
  std::int32_t binary_version;
  std::int32_t block_size;
  std::int32_t blocks_per_cu;
  double occupancy_pct;
  char name[256];
};

struct DevicePayload {
  std::int32_t cu_count;
  std::int32_t warp_size;
  std::int32_t threads_per_cu;
  std::int32_t registers_per_block;
  std::int32_t lds_per_cu;
  std::int32_t clock_khz;
  std::int32_t memory_clock_khz;
  std::int32_t memory_bus_width;
  std::int32_t driver_version;
  std::int32_t runtime_version;
  char name[256];
};

struct Record {
  RecordType type;
  std::uint16_t flags;
  std::uint32_t process_id;
  std::uint32_t thread_id;
  union {
    OpPayload op;
    ApiPayload api;
    RoctxPayload roctx;
    VramPayload vram;
    KernelPayload kernel;
    DevicePayload device;
  };
};

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "ktrace IPC requires lock-free 64-bit atomics");

struct alignas(64) Slot {
  std::atomic<std::uint64_t> sequence;
  Record record;
};

struct alignas(64) SharedSession {
  std::uint32_t magic = 0;
  std::uint16_t abi_version = 0;
  std::uint16_t reserved = 0;
  std::uint32_t capacity = 0;
  std::uint32_t launcher_pid = 0;
  std::atomic<std::uint32_t> root_pid{0};
  std::atomic<std::uint32_t> collector_ready{0};
  std::atomic<std::uint32_t> shutdown_requested{0};
  std::atomic<std::uint64_t> resume_timestamp_ns{0};
  std::atomic<std::uint64_t> enqueue_pos{0};
  std::atomic<std::uint64_t> dequeue_pos{0};
  std::atomic<std::uint64_t> pushed{0};
  std::atomic<std::uint64_t> dropped{0};
  std::atomic<std::uint64_t> high_watermark{0};
  Slot slots[kRingCapacity];
};

struct Mapping {
  HANDLE handle = nullptr;
  SharedSession *session = nullptr;
  std::wstring name;
};

std::size_t MappingSize();
std::wstring MappingName(const std::wstring &session_name);
bool Create(const std::wstring &session_name, Mapping *mapping,
            std::wstring *error);
bool Open(const std::wstring &session_name, Mapping *mapping,
          std::wstring *error);
void Close(Mapping *mapping);
bool Push(Mapping *mapping, const Record &record);
bool Pop(Mapping *mapping, Record *record);
bool Empty(const Mapping &mapping);
} // namespace ktrace::ipc
