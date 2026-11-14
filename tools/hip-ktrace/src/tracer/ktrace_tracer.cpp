/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <psapi.h>

#include "ktrace_ipc.h"

// hip_api_data_t is a union over every HIP call's arguments, and
// hip_api_trace_data_t's function-pointer fields sit after it, so the runtime
// and this DLL must agree on its size.
#include <hip/hip_runtime_api.h>

#include <hip/amd_detail/hip_prof_str.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
// ---- mirrored from rocclr/platform/prof_protocol.h -------------------------
enum activity_domain_t {
  ACTIVITY_DOMAIN_HSA_API = 0,
  ACTIVITY_DOMAIN_HSA_OPS = 1,
  ACTIVITY_DOMAIN_HIP_OPS = 2,
  ACTIVITY_DOMAIN_HIP_API = 3,
  ACTIVITY_DOMAIN_KFD_API = 4,
  ACTIVITY_DOMAIN_EXT_API = 5,
  ACTIVITY_DOMAIN_ROCTX = 6,
};

struct activity_record_t {
  uint32_t domain;
  uint32_t kind;
  uint32_t op;
  uint64_t correlation_id;
  uint64_t begin_ns;
  uint64_t end_ns;
  union {
    struct {
      int device_id;
      uint64_t queue_id;
    };
    struct {
      uint32_t process_id;
      uint32_t thread_id;
    };
    struct {
      uint64_t external_id;
    };
  };
  union {
    size_t bytes;
    const char *kernel_name;
  };
};

enum OpId { OP_ID_DISPATCH = 0, OP_ID_COPY = 1, OP_ID_BARRIER = 2 };

// `record.kind` is the underlying cl_command_type, which is what distinguishes
// transfer direction. Values are the OpenCL constants from CL/cl.h; the labels
// follow getOclCommandKindString() in rocclr/platform/activity.cpp so they read
// the same as anything else in the AMD stack.
const char *KindName(uint32_t kind) {
  switch (kind) {
  case 0x0000:
    return "InternalMarker";
  case 0x11F0:
    return "KernelExecution"; // CL_COMMAND_NDRANGE_KERNEL
  case 0x11F1:
    return "Task";
  case 0x11F2:
    return "NativeKernel";
  case 0x11F3:
    return "CopyDeviceToHost"; // CL_COMMAND_READ_BUFFER
  case 0x11F4:
    return "CopyHostToDevice"; // CL_COMMAND_WRITE_BUFFER
  case 0x11F5:
    return "CopyDeviceToDevice"; // CL_COMMAND_COPY_BUFFER
  case 0x11F6:
    return "ReadImage";
  case 0x11F7:
    return "WriteImage";
  case 0x11F8:
    return "CopyImage";
  case 0x11F9:
    return "CopyImageToBuffer";
  case 0x11FA:
    return "CopyBufferToImage";
  case 0x11FB:
    return "MapBuffer";
  case 0x11FC:
    return "MapImage";
  case 0x11FD:
    return "UnmapMemObject";
  case 0x11FE:
    return "Marker";
  case 0x11FF:
    return "AcquireGLObjects";
  case 0x1200:
    return "ReleaseGLObjects";
  case 0x1201:
    return "CopyDeviceToHost2D"; // CL_COMMAND_READ_BUFFER_RECT
  case 0x1202:
    return "CopyHostToDevice2D"; // CL_COMMAND_WRITE_BUFFER_RECT
  case 0x1203:
    return "CopyDeviceToDevice2D"; // CL_COMMAND_COPY_BUFFER_RECT
  case 0x1204:
    return "User";
  case 0x1205:
    return "Barrier";
  case 0x1206:
    return "MigrateMemObjects";
  case 0x1207:
    return "FillBuffer";
  case 0x1208:
    return "FillImage";
  case 0x1209:
    return "SvmFree";
  case 0x120A:
    return "SvmMemcpy";
  case 0x120B:
    return "SvmMemFill";
  case 0x120C:
    return "SvmMap";
  case 0x120D:
    return "SvmUnmap";
  default:
    return "Other";
  }
}

using PfnRegisterTracer = void(__cdecl *)(int (*)(activity_domain_t, uint32_t,
                                                  void *));

// Mirrors hip_prof_api.h. The runtime hands us an *uninitialized* one of these
// and calls whatever phase_enter/phase_exit we leave behind, so a size mismatch
// on hip_api_data_t is fatal rather than merely wrong.
struct hip_api_trace_data_t {
  hip_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;
  void (*phase_enter)(hip_api_id_t, hip_api_trace_data_t *);
  void (*phase_exit)(hip_api_id_t, hip_api_trace_data_t *);
};

// ---- record store ----------------------------------------------------------
struct Rec {
  uint64_t begin_ns;
  uint64_t end_ns;
  // The record's trailing union holds either a byte count or a kernel name,
  // never both, so only one of these is ever meaningful per record.
  uint64_t bytes;
  uint32_t name_id;
  uint64_t queue_id;
  uint64_t correlation_id; // nonzero only while the API domain is armed
  uint32_t op;
  uint32_t kind; // cl_command_type -- carries the transfer direction
  int device_id;
};

// One entry per intercepted HIP API call, so a GPU command's correlation_id can
// be resolved back to the call that issued it.
struct ApiRec {
  uint64_t correlation_id;
  uint64_t enter_ns;
  uint64_t exit_ns;
  uint32_t api_id;
  uint32_t thread_id;
  // Decoded call arguments, meaning depends on api_id; all zero unless
  // KTRACE_API_ARGS is set. See DecodeArgs() and FormatArgs().
  uint64_t a0, a1, a2, a3, a4, a5;
};

enum RoctxApiId : uint32_t {
  ROCTX_API_ID_MARK = 0,
  ROCTX_API_ID_RANGE_PUSH = 1,
  ROCTX_API_ID_RANGE_POP = 2,
  ROCTX_API_ID_RANGE_START = 3,
  ROCTX_API_ID_RANGE_STOP = 4,
};

struct RoctxApiData {
  union {
    struct {
      const char *message;
      uint64_t id;
    };
    struct {
      const char *message;
    } mark;
    struct {
      const char *message;
    } push;
    struct {
      const char *message;
    } pop;
    struct {
      const char *message;
      uint64_t id;
    } start;
    struct {
      const char *message;
      uint64_t id;
    } stop;
  } args;
};

struct RoctxRec {
  uint64_t timestamp_ns;
  uint64_t range_id;
  uint32_t operation_id;
  uint32_t process_id;
  uint32_t thread_id;
  char message[256];
};

// ---- streaming sink --------------------------------------------------------
// Records used to pile up in one vector until the process exited, so memory
// grew with the length of the run and a killed process left nothing behind.
// They are now handed to a writer thread in batches: one buffer is filled while
// the other is formatted, which keeps the footprint flat however long the run
// lasts and puts everything but the last batch on disk as it happens.
//
// Push() runs under the caller's record lock, so `filling_` needs no lock of
// its own; `mtx_` covers only the handoff. The writer releases `mtx_` before
// formatting, so file I/O never blocks the runtime's callback.
template <typename T> class RecordSink {
public:
  using Flush = std::function<void(std::vector<T> &)>;

  void Start(Flush flush, size_t batch) {
    flush_ = std::move(flush);
    batch_ = batch;
    filling_.reserve(batch);
    pending_.reserve(batch);
    writer_ = std::thread(&RecordSink::Consume, this);
  }

  void Push(const T &rec) {
    filling_.push_back(rec);
    if (filling_.size() >= batch_)
      Handoff();
  }

  // `join` is false when called from DLL_PROCESS_DETACH, where waiting on a
  // thread under the loader lock can hang and the writer may already have
  // been terminated by the OS. Nothing is lost either way: whatever the
  // writer did not take is written by DrainInline() on the caller's thread.
  void Stop(bool join) {
    if (!writer_.joinable())
      return;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (join)
      writer_.join();
    else
      writer_.detach();
  }

  // Write out what the writer never got to. Safe to call after Stop(): the
  // flush itself is serialised against the writer by the output lock.
  void DrainInline() {
    std::vector<T> left;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (has_pending_) {
        left.swap(pending_);
        has_pending_ = false;
      }
    }
    if (!left.empty())
      flush_(left);
    if (!filling_.empty()) {
      flush_(filling_);
      filling_.clear();
    }
  }

private:
  void Handoff() {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this] { return !has_pending_; });
    filling_.swap(pending_);
    has_pending_ = true;
    cv_.notify_all();
  }

  void Consume() {
    std::vector<T> batch;
    batch.reserve(batch_);
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] { return has_pending_ || stop_; });
        if (!has_pending_) {
          if (stop_)
            return;
          continue;
        }
        batch.swap(pending_); // pending_ takes the emptied batch
        has_pending_ = false;
        cv_.notify_all();
      }
      flush_(batch);
      batch.clear();
    }
  }

  Flush flush_;
  size_t batch_ = 65536;
  std::vector<T> filling_; // guarded by the caller's lock
  std::vector<T> pending_; // guarded by mtx_
  bool has_pending_ = false;
  bool stop_ = false;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::thread writer_;
};

// Everything the summary reports, accumulated as records stream past instead of
// by walking a full-run vector that no longer exists. Touched only by the
// writer thread.
struct Stats {
  uint64_t dispatch = 0, copy = 0, barrier = 0;
  uint64_t total = 0, correlated = 0;
  std::map<std::pair<int, uint64_t>, uint64_t> queues;
  struct Xfer {
    uint64_t count = 0, bytes = 0, ns = 0;
  };
  std::map<std::string, Xfer> xfer;
  std::map<std::string, std::pair<uint64_t, uint64_t>>
      api; // name -> {count, ns}
  uint64_t api_total = 0;
};

std::mutex g_mtx;
RecordSink<Rec> g_ops_sink;
uint64_t g_ops_pushed = 0; // under g_mtx

// The writer resolves name ids while the callback may be adding new ones, and
// a vector append can move its strings, so the table needs a lock of its own.
// It is taken once per batch by the writer and only on a kernel's first
// sighting by the callback, so it is never contended in practice.
std::mutex g_names_mtx;
std::vector<std::string> g_names;
std::unordered_map<const char *, uint32_t>
    g_name_ids; // keyed by runtime string address

std::atomic<uint64_t> g_dropped{0};
size_t g_max_recs = SIZE_MAX;
bool g_registered = false;

// Measurement aid: register the hook and let the runtime do all of its
// timestamp collection, but throw every record away. The difference against a
// normal run separates the cost the runtime pays for having a tracer at all
// from the cost of this DLL storing what it reports.
bool g_null_mode = false;
std::atomic<uint64_t> g_null_seen{0};

// ---- API domain ------------------------------------------------------------
// Enabled by default, but only after the runtime's API enumeration is verified
// to match the one this DLL was compiled against. See VerifyApiTable().
bool g_api_enabled = false;
bool g_api_checked = false;
bool g_api_args = false; // KTRACE_API_ARGS

std::mutex g_hook_mtx;
std::vector<HMODULE> g_hooked;
std::atomic<int> g_hook_count{0};
char g_hooked_paths[512] = "";
std::mutex g_api_mtx;
RecordSink<ApiRec> g_api_sink;
uint64_t g_api_pushed = 0; // under g_api_mtx
std::atomic<uint64_t> g_next_corr{1};
std::atomic<uint64_t> g_api_enter{0};
std::atomic<uint64_t> g_api_exit{0};
std::atomic<uint64_t> g_api_dropped{0};
size_t g_max_api = SIZE_MAX;
ktrace::ipc::Mapping g_ipc_mapping;
bool g_ipc_mode = false;

std::mutex g_roctx_mtx;
RecordSink<RoctxRec> g_roctx_sink;
uint64_t g_roctx_pushed = 0;
std::atomic<int> g_roctx_hook_count{0};
std::set<HMODULE> g_roctx_hooked;

// Outcome of the compatibility check, kept for the summary so a refusal is
// always explainable.
struct ApiCheck {
  int runtime_last = 0;
  int ours_last = static_cast<int>(HIP_API_ID_LAST);
  int compared = 0;
  int mismatches = 0;
  char detail[192] = "not run";
};
ApiCheck g_api_check;

using PfnApiName = const char *(__cdecl *)(uint32_t);

// ---- static kernel properties ----------------------------------------------
// Registers, scratch and LDS are properties of the compiled kernel, not of a
// dispatch, so they are queried once per kernel and written to their own file.
// The query happens on the install thread rather than in the callback: calling
// into HIP from inside a HIP call's exit hook would re-enter the runtime, and
// g_in_tracer keeps our own calls from being traced when it does happen.
struct KernelInfo {
  std::string name;
  int num_regs = 0;
  uint64_t scratch = 0;    // localSizeBytes: private/spill/arg per work-item
  uint64_t static_lds = 0; // sharedSizeBytes: static group segment
  int max_threads = 0;
  int binary_version = 0;
  // Theoretical only: what the hardware could hold given this kernel's
  // register and LDS footprint. Says nothing about whether the waves had work
  // to do. Computed for the launch geometry actually observed, since the same
  // kernel at a different block size has a different answer.
  int block_size = 0;
  uint64_t dyn_lds = 0;
  int blocks_per_cu = 0;
  double occupancy_pct = 0.0;
};

// The geometry a kernel was first seen launched with, carried alongside the
// address so the occupancy query can use it.
struct PendingKernel {
  uint64_t addr;
  int block_size;
  uint64_t dyn_lds;
};

struct HipFuncAttributes // mirrors hipFuncAttributes
{
  int binaryVersion;
  int cacheModeCA;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;
  int ptxVersion;
  size_t sharedSizeBytes;
};

using PfnFuncGetAttributes = int(__cdecl *)(HipFuncAttributes *, const void *);
using PfnGetFuncBySymbol = int(__cdecl *)(void **, const void *);
using PfnKernelNameRef = const char *(__cdecl *)(void *);
using PfnOccupancy = int(__cdecl *)(int *, const void *, int, size_t);
using PfnDeviceGetAttr = int(__cdecl *)(int *, int, int);
using PfnMemGetInfo = int(__cdecl *)(size_t *, size_t *);
using PfnDeviceGetName = int(__cdecl *)(char *, int, int);
using PfnGetVersion = int(__cdecl *)(int *);

PfnFuncGetAttributes g_func_attrs = nullptr;
PfnGetFuncBySymbol g_func_by_sym = nullptr;
PfnKernelNameRef g_kernel_name_ref = nullptr;
PfnOccupancy g_occupancy = nullptr;
PfnDeviceGetAttr g_dev_attr = nullptr;
PfnMemGetInfo g_mem_info = nullptr;
PfnDeviceGetName g_dev_name = nullptr;
PfnGetVersion g_driver_ver = nullptr;
PfnGetVersion g_runtime_ver = nullptr;
int g_threads_per_mp = 0;

// Device memory is sampled rather than derived: what the application asked for
// can be reconstructed offline from the recorded hipMalloc/hipFree pairs, but
// the driver's view -- runtime-internal buffers, pool reservations,
// fragmentation, other processes -- exists nowhere in the trace.
struct VramSample {
  uint64_t t_ns;
  uint64_t free_bytes;
  uint64_t total_bytes;
};
std::vector<VramSample> g_vram;
unsigned g_vram_ms = 100; // KTRACE_VRAM_MS, 0 disables

std::mutex g_kinfo_mtx;
std::map<uint64_t, KernelInfo> g_kinfo;  // resolved, by host function
std::vector<PendingKernel> g_kinfo_todo; // seen, not yet queried
HANDLE g_kinfo_evt = nullptr;
thread_local bool g_in_tracer = false;
bool PushIpc(ktrace::ipc::Record &record);

// Called on the launch path, so the common case must not take a lock. After the
// first few launches every address is in this thread's set and nothing global
// is touched.
void NoteKernel(const void *func, int block_size, uint64_t dyn_lds) {
  if (!func)
    return;
  thread_local std::unordered_map<const void *, char> seen;
  if (seen.emplace(func, 1).second == false)
    return;
  {
    std::lock_guard<std::mutex> lock(g_kinfo_mtx);
    g_kinfo_todo.push_back(
        {reinterpret_cast<uint64_t>(func), block_size, dyn_lds});
  }
  // Wake the install thread rather than let it find this on its next sweep:
  // a short run can finish before a one-second poll comes round.
  if (g_kinfo_evt)
    SetEvent(g_kinfo_evt);
}

void ResolveKernels() {
  if (!g_func_attrs)
    return;
  std::vector<PendingKernel> todo;
  {
    std::lock_guard<std::mutex> lock(g_kinfo_mtx);
    todo.swap(g_kinfo_todo);
  }
  if (todo.empty())
    return;

  g_in_tracer = true;
  if (g_threads_per_mp == 0 && g_dev_attr) {
    int v = 0;
    // hipDeviceAttributeMaxThreadsPerMultiProcessor
    if (g_dev_attr(
            &v, static_cast<int>(hipDeviceAttributeMaxThreadsPerMultiProcessor),
            0) == 0)
      g_threads_per_mp = v;
  }

  for (const PendingKernel &p : todo) {
    const void *func = reinterpret_cast<const void *>(p.addr);
    KernelInfo info;
    info.block_size = p.block_size;
    info.dyn_lds = p.dyn_lds;

    HipFuncAttributes a{};
    if (g_func_attrs(&a, func) == 0) {
      info.num_regs = a.numRegs;
      info.scratch = a.localSizeBytes;
      info.static_lds = a.sharedSizeBytes;
      info.max_threads = a.maxThreadsPerBlock;
      info.binary_version = a.binaryVersion;
    }
    if (g_func_by_sym && g_kernel_name_ref) {
      void *f = nullptr;
      if (g_func_by_sym(&f, func) == 0 && f)
        if (const char *n = g_kernel_name_ref(f))
          info.name = n;
    }
    // Let the runtime do this rather than deriving it from register and LDS
    // counts here: it knows the allocation granularity and the wave size the
    // kernel was actually compiled for.
    if (g_occupancy && p.block_size > 0) {
      int blocks = 0;
      if (g_occupancy(&blocks, func, p.block_size,
                      static_cast<size_t>(p.dyn_lds)) == 0) {
        info.blocks_per_cu = blocks;
        if (g_threads_per_mp > 0)
          info.occupancy_pct = 100.0 * blocks * p.block_size / g_threads_per_mp;
      }
    }

    if (g_ipc_mode) {
      ktrace::ipc::Record record{};
      record.type = ktrace::ipc::RecordType::kernel;
      record.thread_id = GetCurrentThreadId();
      record.kernel.func_addr = p.addr;
      record.kernel.scratch = info.scratch;
      record.kernel.static_lds = info.static_lds;
      record.kernel.dynamic_lds = info.dyn_lds;
      record.kernel.vgpr = info.num_regs;
      record.kernel.max_threads = info.max_threads;
      record.kernel.binary_version = info.binary_version;
      record.kernel.block_size = info.block_size;
      record.kernel.blocks_per_cu = info.blocks_per_cu;
      record.kernel.occupancy_pct = info.occupancy_pct;
      strncpy_s(record.kernel.name, sizeof(record.kernel.name),
                info.name.c_str(), _TRUNCATE);
      PushIpc(record);
    }
    std::lock_guard<std::mutex> lock(g_kinfo_mtx);
    g_kinfo.emplace(p.addr, std::move(info));
  }
  g_in_tracer = false;
}

uint64_t NowNs();

// Only ever called once the process has produced GPU commands. Querying the
// runtime before the application has a context would create one, which is a
// side effect a tracer has no business having.
void SampleVram() {
  if (!g_mem_info || g_vram_ms == 0)
    return;
  size_t free_b = 0, total_b = 0;
  g_in_tracer = true;
  const int rc = g_mem_info(&free_b, &total_b);
  g_in_tracer = false;
  if (rc == 0 && total_b) {
    if (g_ipc_mode) {
      ktrace::ipc::Record record{};
      record.type = ktrace::ipc::RecordType::vram;
      record.thread_id = GetCurrentThreadId();
      record.vram.timestamp_ns = NowNs();
      record.vram.free_bytes = free_b;
      record.vram.total_bytes = total_b;
      PushIpc(record);
    } else {
      g_vram.push_back({NowNs(), free_b, total_b});
    }
  }
}

// The API domain is only safe if the runtime and this DLL agree on the HIP API
// enumeration, because that enumeration is what sizes hip_api_data_t's argument
// union -- and therefore where phase_enter / phase_exit land. The runtime
// exports its own copy of the generated name table as hipApiName(), so the two
// can be compared directly instead of trusting version numbers: this machine
// runs a HIP 7.2 driver runtime while applications are built with 7.14.
//
// Every overlapping id is compared, not a sample. A single disagreement means
// the enumerations diverged and the union layout cannot be trusted.
bool VerifyApiTable(HMODULE hip) {
  auto api_name =
      reinterpret_cast<PfnApiName>(GetProcAddress(hip, "hipApiName"));
  if (!api_name) {
    snprintf(g_api_check.detail, sizeof(g_api_check.detail),
             "hipApiName not exported; cannot verify");
    return false;
  }

  for (uint32_t i = 1; i < 4096; ++i) {
    const char *n = api_name(i);
    if (n && n[0] && strcmp(n, "unknown") != 0)
      g_api_check.runtime_last = static_cast<int>(i);
  }

  const int overlap = (g_api_check.runtime_last < g_api_check.ours_last)
                          ? g_api_check.runtime_last
                          : g_api_check.ours_last;
  for (int i = 1; i <= overlap; ++i) {
    const char *rt = api_name(static_cast<uint32_t>(i));
    const char *us = hip_api_name(static_cast<uint32_t>(i));
    if (!rt || !us)
      continue;
    if (strcmp(rt, "unknown") == 0 || strcmp(us, "unknown") == 0)
      continue;
    g_api_check.compared++;
    if (strcmp(rt, us) != 0) {
      if (g_api_check.mismatches == 0)
        snprintf(g_api_check.detail, sizeof(g_api_check.detail),
                 "id %d: runtime='%s' ours='%s'", i, rt, us);
      g_api_check.mismatches++;
    }
  }

  const bool ok = (g_api_check.compared > 0 && g_api_check.mismatches == 0);
  if (ok)
    snprintf(g_api_check.detail, sizeof(g_api_check.detail),
             "%d ids compared, all identical", g_api_check.compared);
  return ok;
}

uint64_t NowNs() {
  static LARGE_INTEGER freq = [] {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return f;
  }();
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return static_cast<uint64_t>(c.QuadPart * (1'000'000'000.0 / freq.QuadPart));
}

template <size_t N> void CopyText(char (&destination)[N], const char *source) {
  if (!source)
    return;
  strncpy_s(destination, N, source, _TRUNCATE);
}

bool PushIpc(ktrace::ipc::Record &record) {
  record.process_id = GetCurrentProcessId();
  return ktrace::ipc::Push(&g_ipc_mapping, record);
}

void EmitDeviceMetadata() {
  static std::atomic<bool> emitted{false};
  if (!g_ipc_mode || !g_dev_attr || emitted.exchange(true))
    return;
  auto attr = [](int id) {
    int value = -1;
    return g_dev_attr(&value, id, 0) == 0 ? value : -1;
  };
  ktrace::ipc::Record record{};
  record.type = ktrace::ipc::RecordType::device;
  record.thread_id = GetCurrentThreadId();
  g_in_tracer = true;
  if (g_dev_name)
    g_dev_name(record.device.name, sizeof(record.device.name), 0);
  record.device.cu_count = attr(hipDeviceAttributeMultiprocessorCount);
  record.device.warp_size = attr(hipDeviceAttributeWarpSize);
  record.device.threads_per_cu =
      attr(hipDeviceAttributeMaxThreadsPerMultiProcessor);
  record.device.registers_per_block =
      attr(hipDeviceAttributeMaxRegistersPerBlock);
  record.device.lds_per_cu =
      attr(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
  record.device.clock_khz = attr(hipDeviceAttributeClockRate);
  record.device.memory_clock_khz = attr(hipDeviceAttributeMemoryClockRate);
  record.device.memory_bus_width = attr(hipDeviceAttributeMemoryBusWidth);
  if (g_driver_ver)
    g_driver_ver(&record.device.driver_version);
  if (g_runtime_ver)
    g_runtime_ver(&record.device.runtime_version);
  g_in_tracer = false;
  PushIpc(record);
}

// The runtime hands back a pointer into its own kernel-name string, stable for
// the life of the kernel object. Interning by address keeps the hot path to a
// hash lookup instead of a string copy per dispatch.
uint32_t Intern(const char *s) {
  if (!s)
    return 0;
  auto it = g_name_ids.find(s);
  if (it != g_name_ids.end())
    return it->second;
  std::lock_guard<std::mutex> lock(g_names_mtx);
  const uint32_t id = static_cast<uint32_t>(g_names.size());
  g_names.emplace_back(s);
  g_name_ids.emplace(s, id);
  return id;
}

// Pack a launch geometry into one word. Grid x can be large; y and z are
// bounded by 65535 and block dimensions by 1024, so 32/16/16 loses nothing.
uint64_t PackDim(uint64_t x, uint64_t y, uint64_t z) {
  return (x & 0xFFFFFFFFull) | ((y & 0xFFFFull) << 32) |
         ((z & 0xFFFFull) << 48);
}

// Read the handful of arguments worth keeping. Output parameters are
// dereferenced here rather than at entry, which is the whole reason this runs
// on exit: hipMalloc's pointer is only written by the time the call returns.
void DecodeArgs(hip_api_id_t id, const hip_api_data_t &d, ApiRec &r) {
  switch (id) {
  case HIP_API_ID_hipMalloc:
    r.a0 = d.args.hipMalloc.ptr
               ? reinterpret_cast<uint64_t>(*d.args.hipMalloc.ptr)
               : 0;
    r.a1 = d.args.hipMalloc.size;
    break;
  case HIP_API_ID_hipHostMalloc:
    r.a0 = d.args.hipHostMalloc.ptr
               ? reinterpret_cast<uint64_t>(*d.args.hipHostMalloc.ptr)
               : 0;
    r.a1 = d.args.hipHostMalloc.size;
    break;
  case HIP_API_ID_hipFree:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipFree.ptr);
    break;
  case HIP_API_ID_hipHostFree:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipHostFree.ptr);
    break;

  case HIP_API_ID_hipLaunchKernel:
    NoteKernel(d.args.hipLaunchKernel.function_address,
               static_cast<int>(d.args.hipLaunchKernel.dimBlocks.x *
                                d.args.hipLaunchKernel.dimBlocks.y *
                                d.args.hipLaunchKernel.dimBlocks.z),
               d.args.hipLaunchKernel.sharedMemBytes);
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipLaunchKernel.stream);
    r.a1 = PackDim(d.args.hipLaunchKernel.numBlocks.x,
                   d.args.hipLaunchKernel.numBlocks.y,
                   d.args.hipLaunchKernel.numBlocks.z);
    r.a2 = PackDim(d.args.hipLaunchKernel.dimBlocks.x,
                   d.args.hipLaunchKernel.dimBlocks.y,
                   d.args.hipLaunchKernel.dimBlocks.z);
    r.a3 = d.args.hipLaunchKernel.sharedMemBytes;
    break;
  case HIP_API_ID_hipModuleLaunchKernel:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipModuleLaunchKernel.stream);
    r.a1 = PackDim(d.args.hipModuleLaunchKernel.gridDimX,
                   d.args.hipModuleLaunchKernel.gridDimY,
                   d.args.hipModuleLaunchKernel.gridDimZ);
    r.a2 = PackDim(d.args.hipModuleLaunchKernel.blockDimX,
                   d.args.hipModuleLaunchKernel.blockDimY,
                   d.args.hipModuleLaunchKernel.blockDimZ);
    r.a3 = d.args.hipModuleLaunchKernel.sharedMemBytes;
    break;
  case HIP_API_ID_hipExtModuleLaunchKernel:
    // Ext takes a global work size, not a grid count: divide it down so the
    // column means the same thing across all launch APIs.
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipExtModuleLaunchKernel.hStream);
    {
      const auto &e = d.args.hipExtModuleLaunchKernel;
      r.a1 =
          PackDim(e.localWorkSizeX ? e.globalWorkSizeX / e.localWorkSizeX : 0,
                  e.localWorkSizeY ? e.globalWorkSizeY / e.localWorkSizeY : 0,
                  e.localWorkSizeZ ? e.globalWorkSizeZ / e.localWorkSizeZ : 0);
      r.a2 = PackDim(e.localWorkSizeX, e.localWorkSizeY, e.localWorkSizeZ);
      r.a3 = e.sharedMemBytes;
    }
    break;

  case HIP_API_ID_hipMemcpy:
    r.a1 = d.args.hipMemcpy.sizeBytes;
    r.a2 = static_cast<uint64_t>(d.args.hipMemcpy.kind);
    r.a4 = reinterpret_cast<uint64_t>(d.args.hipMemcpy.dst);
    r.a5 = reinterpret_cast<uint64_t>(d.args.hipMemcpy.src);
    break;
  case HIP_API_ID_hipMemcpyAsync:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipMemcpyAsync.stream);
    r.a1 = d.args.hipMemcpyAsync.sizeBytes;
    r.a2 = static_cast<uint64_t>(d.args.hipMemcpyAsync.kind);
    r.a4 = reinterpret_cast<uint64_t>(d.args.hipMemcpyAsync.dst);
    r.a5 = reinterpret_cast<uint64_t>(d.args.hipMemcpyAsync.src);
    break;
  case HIP_API_ID_hipMemcpyWithStream:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipMemcpyWithStream.stream);
    r.a1 = d.args.hipMemcpyWithStream.sizeBytes;
    r.a2 = static_cast<uint64_t>(d.args.hipMemcpyWithStream.kind);
    r.a4 = reinterpret_cast<uint64_t>(d.args.hipMemcpyWithStream.dst);
    r.a5 = reinterpret_cast<uint64_t>(d.args.hipMemcpyWithStream.src);
    break;
  case HIP_API_ID_hipMemsetAsync:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipMemsetAsync.stream);
    r.a1 = d.args.hipMemsetAsync.sizeBytes;
    break;

  // Synchronisation. These are where the host spends its time waiting, and
  // the stream is what turns "waited 25 s" into "waited on this queue".
  case HIP_API_ID_hipStreamSynchronize:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipStreamSynchronize.stream);
    break;
  case HIP_API_ID_hipStreamQuery:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipStreamQuery.stream);
    break;
  case HIP_API_ID_hipStreamDestroy:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipStreamDestroy.stream);
    break;
  case HIP_API_ID_hipStreamCreate:
    r.a0 = d.args.hipStreamCreate.stream
               ? reinterpret_cast<uint64_t>(*d.args.hipStreamCreate.stream)
               : 0;
    break;
  case HIP_API_ID_hipStreamCreateWithFlags:
    r.a0 = d.args.hipStreamCreateWithFlags.stream
               ? reinterpret_cast<uint64_t>(
                     *d.args.hipStreamCreateWithFlags.stream)
               : 0;
    r.a1 = d.args.hipStreamCreateWithFlags.flags;
    break;
  case HIP_API_ID_hipStreamWaitEvent:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipStreamWaitEvent.stream);
    r.a1 = reinterpret_cast<uint64_t>(d.args.hipStreamWaitEvent.event);
    break;
  case HIP_API_ID_hipEventRecord:
    r.a0 = reinterpret_cast<uint64_t>(d.args.hipEventRecord.stream);
    r.a1 = reinterpret_cast<uint64_t>(d.args.hipEventRecord.event);
    break;
  case HIP_API_ID_hipEventSynchronize:
    r.a1 = reinterpret_cast<uint64_t>(d.args.hipEventSynchronize.event);
    break;
  case HIP_API_ID_hipEventQuery:
    r.a1 = reinterpret_cast<uint64_t>(d.args.hipEventQuery.event);
    break;
  default:
    break;
  }
}

// Installed only when arguments are wanted. Its presence, not its body, is what
// makes the runtime marshal them.
void PhaseEnter(hip_api_id_t, hip_api_trace_data_t *) {}

void FormatArgs(const ApiRec &r, char *out, size_t cap);

void PhaseExit(hip_api_id_t id, hip_api_trace_data_t *d) {
  g_api_exit.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_api_mtx);
  if (g_api_pushed >= g_max_api) {
    g_api_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  ++g_api_pushed;
  ApiRec r{d->api_data.correlation_id,
           d->phase_enter_timestamp,
           NowNs(),
           static_cast<uint32_t>(id),
           GetCurrentThreadId(),
           0,
           0,
           0,
           0,
           0,
           0};
  if (g_api_args)
    DecodeArgs(id, d->api_data, r);
  if (g_ipc_mode) {
    ktrace::ipc::Record record{};
    record.type = ktrace::ipc::RecordType::api;
    record.thread_id = r.thread_id;
    record.api.correlation_id = r.correlation_id;
    record.api.enter_ns = r.enter_ns;
    record.api.exit_ns = r.exit_ns;
    record.api.api_id = r.api_id;
    CopyText(record.api.api_name, hip_api_name(r.api_id));
    FormatArgs(r, record.api.arguments, sizeof(record.api.arguments));
    if (!PushIpc(record))
      g_api_dropped.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_api_sink.Push(r);
  }
}

int __cdecl TracerCallback(activity_domain_t domain, uint32_t op, void *data) {
  // Calls this DLL makes into HIP itself (querying kernel attributes) must
  // not be traced, or they would recurse.
  if (g_in_tracer)
    return -1;

  if (domain == ACTIVITY_DOMAIN_ROCTX) {
    if (data == nullptr || op > ROCTX_API_ID_RANGE_STOP)
      return 0;
    const auto *roctx = static_cast<const RoctxApiData *>(data);
    RoctxRec record{};
    record.timestamp_ns = NowNs();
    record.operation_id = op;
    record.process_id = GetCurrentProcessId();
    record.thread_id = GetCurrentThreadId();
    if (op == ROCTX_API_ID_RANGE_START || op == ROCTX_API_ID_RANGE_STOP)
      record.range_id = roctx->args.id;
    const char *message =
        (op == ROCTX_API_ID_MARK || op == ROCTX_API_ID_RANGE_PUSH ||
         op == ROCTX_API_ID_RANGE_START)
            ? roctx->args.message
            : nullptr;
    if (message)
      strncpy_s(record.message, sizeof(record.message), message, _TRUNCATE);

    std::lock_guard<std::mutex> lock(g_roctx_mtx);
    ++g_roctx_pushed;
    if (g_ipc_mode) {
      ktrace::ipc::Record ipc_record{};
      ipc_record.type = ktrace::ipc::RecordType::roctx;
      ipc_record.thread_id = record.thread_id;
      ipc_record.roctx.timestamp_ns = record.timestamp_ns;
      ipc_record.roctx.range_id = record.range_id;
      ipc_record.roctx.operation_id = record.operation_id;
      CopyText(ipc_record.roctx.message, record.message);
      if (!PushIpc(ipc_record))
        g_dropped.fetch_add(1, std::memory_order_relaxed);
    } else {
      g_roctx_sink.Push(record);
    }
    return 0;
  }

  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    if (!g_api_enabled)
      return -1;
    if (data == nullptr)
      return 0;
    auto *d = static_cast<hip_api_trace_data_t *>(data);
    // Writing correlation_id is what makes the GPU commands issued by this
    // call carry the same id; the runtime copies it into their profiling
    // info. phase_enter/phase_exit must be set: the struct arrives
    // uninitialized and the runtime calls them unconditionally if non-null.
    d->api_data.correlation_id =
        g_next_corr.fetch_add(1, std::memory_order_relaxed);
    d->phase_data = 0;
    d->phase_exit = PhaseExit;
    // This runs at the top of the API call, so it is the entry timestamp.
    d->phase_enter_timestamp = NowNs();
    g_api_enter.fetch_add(1, std::memory_order_relaxed);
    // The runtime only copies the call's arguments into api_data when a
    // phase_enter is installed (see api_callbacks_spawner_t in
    // hip_prof_api.h). Leaving it null is therefore what avoids paying to
    // marshal arguments nobody reads.
    d->phase_enter = g_api_args ? PhaseEnter : nullptr;
    return 0;
  }
  if (domain != ACTIVITY_DOMAIN_HIP_OPS)
    return -1; // decline other domains

  // A null payload is the runtime asking, for one specific command it is
  // about to construct, whether to trace it. Only dispatches are sampled:
  // copies and barriers are a fraction of a percent of the commands here, and
  // dropping them would cost the transfer totals for no measurable gain.
  if (data == nullptr)
    return 0; // yes, trace this op
  if (data == reinterpret_cast<void *>(uintptr_t{1}))
    return 0; // commit sentinel

  if (g_null_mode) {
    g_null_seen.fetch_add(1, std::memory_order_relaxed);
    return 0;
  }

  const auto *r = static_cast<const activity_record_t *>(data);
  std::lock_guard<std::mutex> lock(g_mtx);
  if (g_ops_pushed >= g_max_recs) {
    g_dropped.fetch_add(1, std::memory_order_relaxed);
    return 0;
  }
  ++g_ops_pushed;
  const bool is_dispatch = (r->op == OP_ID_DISPATCH);
  if (g_ipc_mode) {
    ktrace::ipc::Record record{};
    record.type = ktrace::ipc::RecordType::op;
    record.thread_id = GetCurrentThreadId();
    record.op.begin_ns = r->begin_ns;
    record.op.end_ns = r->end_ns;
    record.op.bytes = is_dispatch ? 0 : static_cast<uint64_t>(r->bytes);
    record.op.queue_id = r->queue_id;
    record.op.correlation_id = r->correlation_id;
    record.op.op = r->op;
    record.op.kind = r->kind;
    record.op.device_id = r->device_id;
    if (is_dispatch)
      CopyText(record.op.kernel_name, r->kernel_name);
    if (!PushIpc(record))
      g_dropped.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_ops_sink.Push({r->begin_ns, r->end_ns,
                     is_dispatch ? 0 : static_cast<uint64_t>(r->bytes),
                     is_dispatch ? Intern(r->kernel_name) : 0, r->queue_id,
                     r->correlation_id, r->op, r->kind, r->device_id});
  }
  return 0;
}

const char *OpName(uint32_t op) {
  switch (op) {
  case OP_ID_DISPATCH:
    return "dispatch";
  case OP_ID_COPY:
    return "copy";
  case OP_ID_BARRIER:
    return "barrier";
  default:
    return "other";
  }
}

const char *RoctxOpName(uint32_t operation_id) {
  switch (operation_id) {
  case ROCTX_API_ID_MARK:
    return "MARK";
  case ROCTX_API_ID_RANGE_PUSH:
    return "RANGE_PUSH";
  case ROCTX_API_ID_RANGE_POP:
    return "RANGE_POP";
  case ROCTX_API_ID_RANGE_START:
    return "RANGE_START";
  case ROCTX_API_ID_RANGE_STOP:
    return "RANGE_STOP";
  default:
    return "UNKNOWN";
  }
}

std::atomic<bool> g_written{false};
std::once_flag g_path_once;
std::string g_out_path;
FILE *g_ops_file = nullptr;
FILE *g_api_file = nullptr;
FILE *g_roctx_file = nullptr;
bool g_outputs_closed = false;
Stats g_stats;

// Guards the output files and the running totals. Two writer threads and, at
// shutdown, the exiting thread all format into them. Timed because shutdown
// runs from DllMain: if a writer was killed mid-flush, giving up beats hanging
// the process, and everything flushed before that is already on disk.
std::recursive_timed_mutex g_file_mtx;

std::string CapText(size_t cap) {
  if (cap == SIZE_MAX)
    return "none";
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", cap);
  return buf;
}

// Deferred to the first record on purpose. Descendants are traced too, and most
// of them never touch the GPU (shells, venv stubs, helper processes); resolving
// the path only once there is something to write keeps them from creating files
// at all, let alone overwriting the trace of the process that did the work.
void ResolveOutputPath() {
  char buf[MAX_PATH];
  const char *path = getenv("KTRACE_CSV");
  if (!path) {
    // Public output follows rocprofv3's -d/--output-directory and
    // -o/--output-file model.  During capture this DLL writes a private
    // spool; python/rocprofv3_export.py turns it into the public domain files
    // after the process exits.  PID is always in the private name, so two GPU
    // processes never contend for it.
    const char *dir = getenv("KTRACE_OUTPUT_DIR");
    const char *prefix = getenv("KTRACE_OUTPUT_FILE");
    char default_prefix[32];
    if (!dir || !*dir)
      dir = ".";
    if (!prefix || !*prefix) {
      snprintf(default_prefix, sizeof(default_prefix), "%lu",
               GetCurrentProcessId());
      prefix = default_prefix;
    }
    snprintf(buf, sizeof(buf), "%s\\.%s_%lu_ops.csv", dir, prefix,
             GetCurrentProcessId());
    path = buf;
  }

  // If two processes both dispatch, the second must not clobber the first.
  // Claiming the name with an exclusive create settles that without a race.
  std::string owned(path);
  HANDLE claim = CreateFileA(owned.c_str(), GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (claim == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_EXISTS) {
    const size_t dot = owned.find_last_of('.');
    char suffix[32];
    snprintf(suffix, sizeof(suffix), "_%lu", GetCurrentProcessId());
    owned.insert(dot == std::string::npos ? owned.size() : dot, suffix);
  } else if (claim != INVALID_HANDLE_VALUE) {
    CloseHandle(claim);
  }
  g_out_path = owned;
}

// Timestamps go out absolute rather than relative to the first record: with
// batches leaving as they are produced there is no point at which the earliest
// one is known. csv_to_timeline.py subtracts the minimum itself, so nothing
// downstream changes.
void FlushOpsLocked(std::vector<Rec> &batch) {
  // A writer detached at shutdown can wake after the files are finalised, and
  // the lazy open below would reopen with "w" and truncate the finished CSV.
  if (batch.empty() || g_outputs_closed)
    return;
  std::call_once(g_path_once, ResolveOutputPath);
  if (!g_ops_file) {
    g_ops_file = fopen(g_out_path.c_str(), "w");
    if (!g_ops_file)
      return;
    fprintf(g_ops_file, "Id,Op,Kind,Kernel_Name,Bytes,Device,Queue,Corr,Begin_"
                        "us,End_us,Duration_us\n");
  }

  std::lock_guard<std::mutex> names(g_names_mtx);
  for (const Rec &r : batch) {
    const char *name = (r.op == OP_ID_DISPATCH && r.name_id < g_names.size())
                           ? g_names[r.name_id].c_str()
                           : "";
    fprintf(g_ops_file, "%llu,%s,%s,%s,%llu,%d,%llu,%llu,%.4f,%.4f,%.4f\n",
            (unsigned long long)g_stats.total, OpName(r.op), KindName(r.kind),
            name, (unsigned long long)r.bytes, r.device_id,
            (unsigned long long)r.queue_id,
            (unsigned long long)r.correlation_id, r.begin_ns / 1000.0,
            r.end_ns / 1000.0, (r.end_ns - r.begin_ns) / 1000.0);

    g_stats.total++;
    (r.op == OP_ID_DISPATCH ? g_stats.dispatch
     : r.op == OP_ID_COPY   ? g_stats.copy
                            : g_stats.barrier)++;
    if (r.correlation_id)
      g_stats.correlated++;
    g_stats.queues[{r.device_id, r.queue_id}]++;
    if (r.op == OP_ID_COPY) {
      auto &e = g_stats.xfer[KindName(r.kind)];
      e.count++;
      e.bytes += r.bytes;
      e.ns += (r.end_ns - r.begin_ns);
    }
  }
}

// One CSV field holding `key=value` pairs rather than a column per argument:
// the interesting arguments differ per API, and spreading them out would give a
// mostly-empty table. Separated by spaces so nothing needs quoting.
void FormatArgs(const ApiRec &r, char *out, size_t cap) {
  auto dim = [](uint64_t p, char *b, size_t n) {
    snprintf(b, n, "%ux%ux%u", (unsigned)(p & 0xFFFFFFFF),
             (unsigned)((p >> 32) & 0xFFFF), (unsigned)((p >> 48) & 0xFFFF));
  };
  char g[48], b[48];
  out[0] = '\0';

  switch (r.api_id) {
  case HIP_API_ID_hipMalloc:
  case HIP_API_ID_hipHostMalloc:
    snprintf(out, cap, "ptr=0x%llx size=%llu", (unsigned long long)r.a0,
             (unsigned long long)r.a1);
    break;
  case HIP_API_ID_hipFree:
  case HIP_API_ID_hipHostFree:
    snprintf(out, cap, "ptr=0x%llx", (unsigned long long)r.a0);
    break;
  case HIP_API_ID_hipLaunchKernel:
  case HIP_API_ID_hipModuleLaunchKernel:
  case HIP_API_ID_hipExtModuleLaunchKernel:
    dim(r.a1, g, sizeof(g));
    dim(r.a2, b, sizeof(b));
    snprintf(out, cap, "grid=%s block=%s shmem=%llu stream=0x%llx", g, b,
             (unsigned long long)r.a3, (unsigned long long)r.a0);
    break;
  case HIP_API_ID_hipMemcpy:
  case HIP_API_ID_hipMemcpyAsync:
  case HIP_API_ID_hipMemcpyWithStream:
    snprintf(out, cap,
             "size=%llu kind=%llu stream=0x%llx dst=0x%llx src=0x%llx",
             (unsigned long long)r.a1, (unsigned long long)r.a2,
             (unsigned long long)r.a0, (unsigned long long)r.a4,
             (unsigned long long)r.a5);
    break;
  case HIP_API_ID_hipMemsetAsync:
    snprintf(out, cap, "size=%llu stream=0x%llx", (unsigned long long)r.a1,
             (unsigned long long)r.a0);
    break;

  case HIP_API_ID_hipStreamSynchronize:
  case HIP_API_ID_hipStreamQuery:
  case HIP_API_ID_hipStreamDestroy:
  case HIP_API_ID_hipStreamCreate:
    snprintf(out, cap, "stream=0x%llx", (unsigned long long)r.a0);
    break;
  case HIP_API_ID_hipStreamCreateWithFlags:
    snprintf(out, cap, "stream=0x%llx flags=%llu", (unsigned long long)r.a0,
             (unsigned long long)r.a1);
    break;
  case HIP_API_ID_hipStreamWaitEvent:
  case HIP_API_ID_hipEventRecord:
    snprintf(out, cap, "stream=0x%llx event=0x%llx", (unsigned long long)r.a0,
             (unsigned long long)r.a1);
    break;
  case HIP_API_ID_hipEventSynchronize:
  case HIP_API_ID_hipEventQuery:
    snprintf(out, cap, "event=0x%llx", (unsigned long long)r.a1);
    break;
  default:
    break;
  }
}

// API calls go to their own file because they are a different kind of event,
// not because they are on a different timebase: these timestamps come from
// QueryPerformanceCounter, and rocclr's own Os::timeNanos() -- which the GPU
// timestamps are calibrated against -- is QPC too, so the two files share a
// timeline. Verified by correlation: every GPU command nests inside the host
// call that issued it, to within 0.4 ms.
void FlushApiLocked(std::vector<ApiRec> &batch) {
  if (batch.empty() || g_outputs_closed)
    return;
  std::call_once(g_path_once, ResolveOutputPath);
  if (!g_api_file) {
    g_api_file = fopen((g_out_path + ".api.csv").c_str(), "w");
    if (!g_api_file)
      return;
    fprintf(g_api_file,
            "Corr,Api,Api_Id,Thread,Enter_us,Exit_us,Host_us,Args\n");
  }

  char args[128];
  for (const ApiRec &r : batch) {
    const char *n = hip_api_name(r.api_id);
    FormatArgs(r, args, sizeof(args));
    fprintf(g_api_file, "%llu,%s,%u,%u,%.4f,%.4f,%.4f,%s\n",
            (unsigned long long)r.correlation_id, n, r.api_id, r.thread_id,
            r.enter_ns / 1000.0, r.exit_ns / 1000.0,
            (r.exit_ns - r.enter_ns) / 1000.0, args);

    auto &e = g_stats.api[n];
    e.first++;
    e.second += (r.exit_ns - r.enter_ns);
    g_stats.api_total++;
  }
}

void WriteCsvField(FILE *file, const char *value) {
  fputc('"', file);
  for (const char *cursor = value; *cursor; ++cursor) {
    if (*cursor == '"')
      fputc('"', file);
    fputc(*cursor, file);
  }
  fputc('"', file);
}

void FlushRoctxLocked(std::vector<RoctxRec> &batch) {
  if (batch.empty() || g_outputs_closed)
    return;
  std::call_once(g_path_once, ResolveOutputPath);
  if (!g_roctx_file) {
    g_roctx_file = fopen((g_out_path + ".roctx.csv").c_str(), "w");
    if (!g_roctx_file)
      return;
    fprintf(g_roctx_file,
            "Op,Op_Id,Process,Thread,Range_Id,Timestamp_us,Message\n");
  }

  for (const RoctxRec &record : batch) {
    fprintf(g_roctx_file, "%s,%u,%u,%u,%llu,%.4f,",
            RoctxOpName(record.operation_id), record.operation_id,
            record.process_id, record.thread_id,
            static_cast<unsigned long long>(record.range_id),
            record.timestamp_ns / 1000.0);
    WriteCsvField(g_roctx_file, record.message);
    fputc('\n', g_roctx_file);
  }
}

void FlushOps(std::vector<Rec> &batch) {
  std::lock_guard<std::recursive_timed_mutex> out(g_file_mtx);
  FlushOpsLocked(batch);
}

void FlushApi(std::vector<ApiRec> &batch) {
  std::lock_guard<std::recursive_timed_mutex> out(g_file_mtx);
  FlushApiLocked(batch);
}

void FlushRoctx(std::vector<RoctxRec> &batch) {
  std::lock_guard<std::recursive_timed_mutex> out(g_file_mtx);
  FlushRoctxLocked(batch);
}

// `join` is false at DLL_PROCESS_DETACH, where the writer threads may already
// be gone and waiting on them under the loader lock would hang. Note that on
// a normal exit DllMain's detach runs *before* the CRT's atexit list, so this
// is the path that does the work -- it cannot simply defer to atexit.
void Shutdown(bool join) {
  if (g_written.exchange(true))
    return;
  if (g_ipc_mode) {
    ktrace::ipc::Close(&g_ipc_mapping);
    return;
  }

  // Lock order is producer locks, then the output lock. Holding the producer
  // locks across the whole shutdown stops new records arriving while the
  // leftovers are written; taking them after the output lock instead would
  // deadlock, because a producer blocked in Handoff holds its lock while
  // waiting for a writer that needs the output lock.
  std::lock_guard<std::mutex> ops_lock(g_mtx);
  std::lock_guard<std::mutex> api_lock(g_api_mtx);
  std::lock_guard<std::mutex> roctx_lock(g_roctx_mtx);

  // Must happen before the output lock is taken: a writer finishing its last
  // batch still needs it.
  g_ops_sink.Stop(join);
  g_api_sink.Stop(join);
  g_roctx_sink.Stop(join);

  std::unique_lock<std::recursive_timed_mutex> out(g_file_mtx, std::defer_lock);
  if (!out.try_lock_for(std::chrono::seconds(2)))
    return;

  g_ops_sink.DrainInline();
  g_api_sink.DrainInline();
  g_roctx_sink.DrainInline();

  if (g_ops_file) {
    fclose(g_ops_file);
    g_ops_file = nullptr;
  }
  if (g_api_file) {
    fclose(g_api_file);
    g_api_file = nullptr;
  }
  if (g_roctx_file) {
    fclose(g_roctx_file);
    g_roctx_file = nullptr;
  }
  g_outputs_closed = true;

  if (!g_vram.empty()) {
    std::call_once(g_path_once, ResolveOutputPath);
    if (FILE *v = fopen((g_out_path + ".vram.csv").c_str(), "w")) {
      fprintf(v, "Time_us,Free_Bytes,Total_Bytes,Used_Bytes\n");
      for (const VramSample &s : g_vram)
        fprintf(v, "%.4f,%llu,%llu,%llu\n", s.t_ns / 1000.0,
                (unsigned long long)s.free_bytes,
                (unsigned long long)s.total_bytes,
                (unsigned long long)(s.total_bytes - s.free_bytes));
      fclose(v);
    }
  }

  // Static per-kernel properties, one row per kernel rather than extra
  // columns on millions of dispatch rows. Join on Kernel_Name.
  if (!g_kinfo.empty()) {
    std::call_once(g_path_once, ResolveOutputPath);
    if (FILE *k = fopen((g_out_path + ".kernels.csv").c_str(), "w")) {
      // "Theoretical" is in the column names on purpose: these are what
      // the hardware limits permit, not what was observed. A kernel can
      // show 100% here and still spend its time stalled on memory.
      fprintf(k,
              "Kernel_Name,Func_Addr,VGPR,Scratch_Bytes,Static_LDS,Max_Threads,"
              "Binary_Version,Occ_Block_Size,Occ_Dyn_LDS,"
              "Theoretical_Blocks_Per_CU,Theoretical_Occupancy_Pct\n");
      std::lock_guard<std::mutex> lock(g_kinfo_mtx);
      for (const auto &kv : g_kinfo)
        fprintf(k, "%s,0x%llx,%d,%llu,%llu,%d,%d,%d,%llu,%d,%.1f\n",
                kv.second.name.c_str(), (unsigned long long)kv.first,
                kv.second.num_regs, (unsigned long long)kv.second.scratch,
                (unsigned long long)kv.second.static_lds, kv.second.max_threads,
                kv.second.binary_version, kv.second.block_size,
                (unsigned long long)kv.second.dyn_lds, kv.second.blocks_per_cu,
                kv.second.occupancy_pct);
      fclose(k);
    }
  }

  // A process that loaded HIP but produced nothing still reports -- that
  // combination is worth seeing. One that never saw HIP stays silent.
  if (g_stats.total == 0 && g_hook_count.load() == 0 &&
      g_roctx_hook_count.load() == 0 && !g_null_mode)
    return;
  std::call_once(g_path_once, ResolveOutputPath);

  // A short sidecar so a run can be sanity-checked without parsing the CSV.
  std::string sum(g_out_path);
  sum += ".summary.txt";
  if (FILE *s = fopen(sum.c_str(), "w")) {
    const uint64_t nd = g_stats.dispatch, nc = g_stats.copy,
                   nb = g_stats.barrier;
    fprintf(s, "pid              : %lu\n", GetCurrentProcessId());

    // Without this a trace cannot be read on another machine: the occupancy
    // numbers in the kernel file only mean something against these limits.
    if (g_dev_attr && g_hook_count.load()) {
      g_in_tracer = true;
      char name[256] = "";
      if (g_dev_name)
        g_dev_name(name, sizeof(name), 0);
      auto attr = [](int a) {
        int v = 0;
        return g_dev_attr(&v, a, 0) == 0 ? v : -1;
      };
      int drv = 0, rt = 0;
      if (g_driver_ver)
        g_driver_ver(&drv);
      if (g_runtime_ver)
        g_runtime_ver(&rt);
      fprintf(s, "\ndevice           : %s\n", name);
      fprintf(s, "  CUs            : %d\n",
              attr(hipDeviceAttributeMultiprocessorCount));
      fprintf(s, "  warp size      : %d\n", attr(hipDeviceAttributeWarpSize));
      fprintf(s, "  threads/CU     : %d\n",
              attr(hipDeviceAttributeMaxThreadsPerMultiProcessor));
      fprintf(s, "  regs/block     : %d\n",
              attr(hipDeviceAttributeMaxRegistersPerBlock));
      fprintf(s, "  LDS/CU         : %d\n",
              attr(hipDeviceAttributeMaxSharedMemoryPerMultiprocessor));
      fprintf(s, "  clock / mem    : %d kHz / %d kHz, bus %d bit\n",
              attr(hipDeviceAttributeClockRate),
              attr(hipDeviceAttributeMemoryClockRate),
              attr(hipDeviceAttributeMemoryBusWidth));
      fprintf(s, "  driver / hip   : %d / %d\n", drv, rt);
      fprintf(s, "\n");
      g_in_tracer = false;
    }

    fprintf(s, "hook registered  : %s (%d runtime module%s)\n",
            g_registered ? "yes" : "NO", g_hook_count.load(),
            g_hook_count.load() == 1 ? "" : "s");
    fprintf(s, "ROCTx hooks      : %d\n", g_roctx_hook_count.load());
    fprintf(s, "ROCTx events     : %llu\n",
            static_cast<unsigned long long>(g_roctx_pushed));
    if (g_hooked_paths[0])
      fprintf(s, "  hooked         : %s\n", g_hooked_paths);
    if (g_null_mode)
      fprintf(
          s,
          "MEASUREMENT MODE : KTRACE_NULL -- %llu records seen and discarded\n",
          (unsigned long long)g_null_seen.load());
    fprintf(s, "records          : %llu\n", (unsigned long long)g_stats.total);
    fprintf(s, "  dispatch       : %llu\n", (unsigned long long)nd);
    fprintf(s, "  copy           : %llu\n", (unsigned long long)nc);
    fprintf(s, "  barrier/other  : %llu\n", (unsigned long long)nb);
    {
      std::lock_guard<std::mutex> names(g_names_mtx);
      fprintf(s, "distinct kernels : %zu\n", g_names.size());
    }
    fprintf(s, "dropped (cap)    : %llu  (cap %s)\n",
            (unsigned long long)g_dropped.load(), CapText(g_max_recs).c_str());

    fprintf(s, "\nAPI domain       : %s\n", g_api_enabled ? "on" : "off");
    fprintf(s, "  compat check   : %s\n", g_api_check.detail);
    fprintf(s, "  api ids        : runtime %d / built-for %d\n",
            g_api_check.runtime_last, g_api_check.ours_last);
    if (g_api_enabled) {
      // sizeof is version-dependent; recording it makes a layout mismatch
      // on another driver diagnosable instead of mysterious.
      fprintf(s, "  sizeof api_data: %zu (built against HIP_API_ID_LAST=%d)\n",
              sizeof(hip_api_data_t), (int)HIP_API_ID_LAST);
      fprintf(s, "  api calls      : %llu  (enter %llu / exit %llu)\n",
              (unsigned long long)g_stats.api_total,
              (unsigned long long)g_api_enter.load(),
              (unsigned long long)g_api_exit.load());
      fprintf(s, "  api dropped    : %llu  (cap %s)\n",
              (unsigned long long)g_api_dropped.load(),
              CapText(g_max_api).c_str());
      fprintf(s, "  gpu cmds with correlation : %llu / %llu\n",
              (unsigned long long)g_stats.correlated,
              (unsigned long long)g_stats.total);

      std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> top(
          g_stats.api.begin(), g_stats.api.end());
      std::sort(top.begin(), top.end(), [](auto &a, auto &b) {
        return a.second.second > b.second.second;
      });
      fprintf(s, "\n  top HIP APIs by host time:\n");
      fprintf(s, "  %-34s%12s%14s%12s\n", "api", "count", "total_ms", "avg_us");
      for (size_t i = 0; i < top.size() && i < 12; ++i)
        fprintf(s, "  %-34s%12llu%14.2f%12.3f\n", top[i].first.c_str(),
                (unsigned long long)top[i].second.first,
                top[i].second.second / 1e6,
                top[i].second.second / 1000.0 / top[i].second.first);
    }

    // Hardware queues actually used. HIP streams are a software concept;
    // this is how many command queues the work really landed on.
    fprintf(s, "\ndevice/queue usage:\n");
    for (const auto &kv : g_stats.queues)
      fprintf(s, "  device %d queue %-6llu : %llu records\n", kv.first.first,
              (unsigned long long)kv.first.second,
              (unsigned long long)kv.second);

    // Per-direction transfer totals. The effective rate uses only the time
    // the transfer itself was on the GPU, so it is a per-operation rate,
    // not a share of wall time.
    if (!g_stats.xfer.empty()) {
      fprintf(s, "\ntransfers by direction:\n");
      fprintf(s, "  %-24s%12s%14s%14s\n", "kind", "count", "total_MB", "GB/s");
      for (const auto &kv : g_stats.xfer)
        fprintf(s, "  %-24s%12llu%14.2f%14.2f\n", kv.first.c_str(),
                (unsigned long long)kv.second.count, kv.second.bytes / 1e6,
                kv.second.ns ? kv.second.bytes / (double)kv.second.ns : 0.0);
    }
    fclose(s);
  }
}

// `report_activity` is a global inside whichever amdhip64 copy exports it, so
// the hook must be installed per loaded module. That matters more than it
// sounds: an application can ship its own HIP runtime next to itself and load
// it by full path, which gives the process two amdhip64_7.dll modules. Forcing
// one in by bare name resolves to System32 (it outranks cwd and PATH) and would
// arm a runtime the application never calls -- the hook reports success and
// then records nothing.
//
// So: never load HIP here, just adopt whatever the process has. Modules can
// appear late, so the install thread keeps re-running this.
void HookLoadedRuntimes() {
  HMODULE mods[512];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed))
    return;
  const size_t n = needed / sizeof(HMODULE);

  for (size_t i = 0; i < n && i < 512; ++i) {
    char base[MAX_PATH];
    if (!K32GetModuleBaseNameA(GetCurrentProcess(), mods[i], base, MAX_PATH))
      continue;
    if (_strnicmp(base, "amdhip64", 8) != 0)
      continue;

    std::lock_guard<std::mutex> lock(g_hook_mtx);
    if (std::find(g_hooked.begin(), g_hooked.end(), mods[i]) != g_hooked.end())
      continue;

    auto reg = reinterpret_cast<PfnRegisterTracer>(
        GetProcAddress(mods[i], "hipRegisterTracerCallback"));
    if (!reg)
      continue;

    if (!g_api_checked) {
      g_api_checked = true;
      if (getenv("KTRACE_NO_API") != nullptr)
        snprintf(g_api_check.detail, sizeof(g_api_check.detail),
                 "disabled by KTRACE_NO_API");
      else
        g_api_enabled = VerifyApiTable(mods[i]);
    }

    if (!g_func_attrs) {
      g_func_attrs = reinterpret_cast<PfnFuncGetAttributes>(
          GetProcAddress(mods[i], "hipFuncGetAttributes"));
      g_func_by_sym = reinterpret_cast<PfnGetFuncBySymbol>(
          GetProcAddress(mods[i], "hipGetFuncBySymbol"));
      g_kernel_name_ref = reinterpret_cast<PfnKernelNameRef>(
          GetProcAddress(mods[i], "hipKernelNameRef"));
      g_occupancy = reinterpret_cast<PfnOccupancy>(GetProcAddress(
          mods[i], "hipOccupancyMaxActiveBlocksPerMultiprocessor"));
      g_dev_attr = reinterpret_cast<PfnDeviceGetAttr>(
          GetProcAddress(mods[i], "hipDeviceGetAttribute"));
      g_mem_info = reinterpret_cast<PfnMemGetInfo>(
          GetProcAddress(mods[i], "hipMemGetInfo"));
      g_dev_name = reinterpret_cast<PfnDeviceGetName>(
          GetProcAddress(mods[i], "hipDeviceGetName"));
      g_driver_ver = reinterpret_cast<PfnGetVersion>(
          GetProcAddress(mods[i], "hipDriverGetVersion"));
      g_runtime_ver = reinterpret_cast<PfnGetVersion>(
          GetProcAddress(mods[i], "hipRuntimeGetVersion"));
    }

    reg(TracerCallback);
    g_hooked.push_back(mods[i]);
    g_registered = true;
    g_hook_count.fetch_add(1, std::memory_order_relaxed);

    char full[MAX_PATH];
    if (GetModuleFileNameA(mods[i], full, MAX_PATH)) {
      const size_t used = strlen(g_hooked_paths);
      snprintf(g_hooked_paths + used, sizeof(g_hooked_paths) - used, "%s%s",
               used ? "\n                   " : "", full);
    }
  }
}

void HookLoadedRoctx() {
  HMODULE modules[512];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules),
                             &needed))
    return;
  const size_t count = needed / sizeof(HMODULE);

  for (size_t i = 0; i < count && i < 512; ++i) {
    char base[MAX_PATH]{};
    if (!K32GetModuleBaseNameA(GetCurrentProcess(), modules[i], base, MAX_PATH))
      continue;
    _strlwr_s(base, sizeof(base));
    if (strstr(base, "roctx") == nullptr)
      continue;

    std::lock_guard<std::mutex> lock(g_hook_mtx);
    if (g_roctx_hooked.count(modules[i]))
      continue;

    using RegisterRoctx =
        void(__cdecl *)(int(__cdecl *)(activity_domain_t, uint32_t, void *));
    auto register_callback = reinterpret_cast<RegisterRoctx>(
        GetProcAddress(modules[i], "roctxRegisterTracerCallback"));
    if (!register_callback)
      continue;

    register_callback(TracerCallback);
    g_roctx_hooked.insert(modules[i]);
    g_roctx_hook_count.fetch_add(1, std::memory_order_relaxed);
  }
}

DWORD WINAPI Install(LPVOID) {
  g_null_mode = getenv("KTRACE_NULL") != nullptr;
  g_api_args = getenv("KTRACE_API_ARGS") != nullptr;
  const char *ipc_session = getenv("KTRACE_IPC_SESSION");
  const char *ipc_mode = getenv("KTRACE_IPC_MODE");
  if (ipc_session && *ipc_session &&
      (!ipc_mode || _stricmp(ipc_mode, "inline") != 0)) {
    const std::wstring session(ipc_session,
                               ipc_session + std::strlen(ipc_session));
    std::wstring ipc_error;
    g_ipc_mode = ktrace::ipc::Open(session, &g_ipc_mapping, &ipc_error);
  }
  if (const char *v = getenv("KTRACE_VRAM_MS")) {
    const long long ms = atoll(v);
    if (ms >= 0)
      g_vram_ms = static_cast<unsigned>(ms);
  }

  // Records stream to disk, so these bound how much output a run produces
  // rather than how much memory it holds. 0 lifts them.
  if (const char *m = getenv("KTRACE_MAX")) {
    const long long v = atoll(m);
    if (v >= 0)
      g_max_recs = v ? static_cast<size_t>(v) : SIZE_MAX;
  }
  if (const char *m = getenv("KTRACE_API_MAX")) {
    const long long v = atoll(m);
    if (v >= 0)
      g_max_api = v ? static_cast<size_t>(v) : SIZE_MAX;
  }
  // Batch size trades flush frequency against the footprint. 64K records is
  // 4 MB of Rec, and at the ~24k records/s this workload produces that is a
  // flush every few seconds -- rare enough that the writer is never behind.
  if (!g_ipc_mode) {
    g_ops_sink.Start(FlushOps, 65536);
    g_api_sink.Start(FlushApi, 65536);
    g_roctx_sink.Start(FlushRoctx, 4096);
  }

  // The process is still suspended at injection time, so HIP is usually not
  // mapped yet. Watch for it rather than forcing a load; a Python host that
  // dlopens its execution provider can take seconds to get there. Keep
  // scanning after the first hit as well, since a second runtime copy may
  // arrive later and would otherwise go untraced.
  g_kinfo_evt = CreateEventA(nullptr, FALSE, FALSE, nullptr);
  for (;;) {
    HookLoadedRuntimes();
    HookLoadedRoctx();
    if (g_ops_pushed > 0)
      EmitDeviceMetadata();
    // Off the callback path on purpose: this calls back into HIP, which
    // must not happen from inside another HIP call's exit hook.
    if (g_api_args)
      ResolveKernels();

    // Wait for the sampling interval, but stay responsive to new kernels.
    // Only sample once the application is actually using the GPU.
    DWORD wait = (g_hook_count.load() == 0) ? 50 : 1000;
    if (g_vram_ms && g_hook_count.load() && g_ops_pushed > 0) {
      SampleVram();
      wait = g_vram_ms;
    }
    if (g_kinfo_evt)
      WaitForSingleObject(g_kinfo_evt, wait);
    else
      Sleep(wait);
  }
}
} // namespace

BOOL APIENTRY DllMain(HMODULE self, DWORD reason, LPVOID) {
  switch (reason) {
  case DLL_PROCESS_ATTACH:
    DisableThreadLibraryCalls(self);
    // LoadLibrary under the loader lock would deadlock, so arm the hook on
    // a separate thread.
    if (HANDLE t = CreateThread(nullptr, 0, Install, nullptr, 0, nullptr))
      CloseHandle(t);
    atexit([] { Shutdown(true); });
    break;
  case DLL_PROCESS_DETACH:
    // atexit has normally run already; this only covers an unload that
    // skipped it, and must not wait on threads under the loader lock.
    Shutdown(false);
    break;
  default:
    break;
  }
  return TRUE;
}
