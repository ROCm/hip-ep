/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Profiling-only trace shim (HIPDNN_EP_ENABLE_TRACE_SHIM). Compiled into
// hipdnn-ep-tls, and therefore into the EP host DLL, only when that CMake
// option is ON; the default build contains none of this file.
//
// Three roles, all inert unless HIPEP_TRACE or HIPEP_TRACE_FENCE is set:
//
//  * In-process ROCTx library. roctxRangePushA / roctxRangePop / roctxMarkA are
//    exported from the DLL that links this file, so pointing
//    HIPDNN_EP_ROCTX_LIB at that DLL routes both ROCTx bridges (the runtime's
//    roctx_shim.cpp and ort-bridge's roctx-range.hpp) here without touching
//    runtime.bc.
//  * HIP activity tracer. hipRegisterTracerCallback delivers every completed
//    GPU command with the runtime's own begin/end timestamps. The HIP API
//    domain, which supplies the correlation ids that tie a GPU command to the
//    call (and thread) that issued it, is armed only after the runtime's API
//    name table matches the one compiled here: hip_api_data_t is sized by that
//    enumeration and the runtime writes our phase_exit pointer after it.
//  * Op fence. On the N-th push of hip_ep.op.<HIPEP_TRACE_FENCE> after Run
//    HIPEP_TRACE_FENCE_AFTER_RUNS it drains the GPU, prints [RGP_FENCE_ARMED]
//    for an external RGP trigger, optionally launches a uniquely named marker
//    kernel on the session stream (an in-process SQTT window keyed on a kernel
//    name opens there), and idles HIPEP_TRACE_FENCE_MS.
//
// Records go to a named shared-memory ring created by the collector
// (tools/perf-harness, hipep_perf/collector.py), so committed records survive
// any kind of target exit. The layout below is mirrored there; bump
// kRingVersion on any change.

#include "trace_shim.h"
#include "hipdnn_ep_runtime.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <hip/amd_detail/hip_prof_str.h>
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {

// ---- ring layout (mirrored in hipep_perf/ring.py) ---------------------------
constexpr uint64_t kRingMagic = 0x3152544550454950ull; // "PIEPETR1"
constexpr uint32_t kRingVersion = 1;
constexpr uint32_t kRecordSize = 64;
constexpr size_t kHeaderSize = 4096;

struct RingHeader {
  uint64_t magic;                         // 0
  uint32_t version;                       // 8
  uint32_t recordSize;                    // 12
  uint64_t capacity;                      // 16, records, power of two
  uint64_t qpcFrequency;                  // 24, written by the shim
  uint32_t producerPid;                   // 32
  uint32_t flags;                         // 36, kFlag*
  uint32_t apiCompared;                   // 40
  uint32_t apiMismatches;                 // 44
  char apiDetail[128];                    // 48
  uint8_t pad0[256 - 176];                // to 256
  alignas(64) std::atomic<uint64_t> head; // 256, next reservation
  uint8_t pad1[56];
  alignas(64) std::atomic<uint64_t> tail; // 320, owned by the collector
  uint8_t pad2[56];
  alignas(64) std::atomic<uint64_t> dropped; // 384
};
static_assert(offsetof(RingHeader, head) == 256, "ring layout");
static_assert(offsetof(RingHeader, tail) == 320, "ring layout");
static_assert(offsetof(RingHeader, dropped) == 384, "ring layout");
static_assert(sizeof(RingHeader) <= kHeaderSize, "ring layout");

constexpr uint32_t kFlagAttached = 1u << 0;
constexpr uint32_t kFlagTracer = 1u << 1;
constexpr uint32_t kFlagApi = 1u << 2;
constexpr uint32_t kFlagDetached = 1u << 3;

struct Record {
  std::atomic<uint64_t> seq; // position + 1 once committed
  uint16_t type;
  uint16_t flags;
  uint32_t tid;
  uint64_t f[6];
};
static_assert(sizeof(Record) == kRecordSize, "record layout");

enum RecordType : uint16_t {
  kRecOp = 1,    // f: corr, begin_ns, end_ns, queue_id, kind|op<<32|dev<<40,
                 // name_id|bytes
  kRecApi = 2,   // f: corr, enter_ns, exit_ns, api_id
  kRecPush = 3,  // f: ts_ns, name_id, depth
  kRecPop = 4,   // f: ts_ns, depth
  kRecMark = 5,  // f: ts_ns, name_id
  kRecName = 6,  // f0: id | chunk<<32 | nchunks<<48; f1..f5: 40 bytes of text
  kRecFence = 7, // f: ts_ns, name_id (op), run, instance, marker_launched
  kRecDevice =
      8, // f: name_id, arch_id, cu|warp<<32, threads_per_cu|lds_per_cu<<32,
         //    clock_khz|mem_clock_khz<<32, bus_bits|device<<32
};

constexpr uint32_t kApiNameBase = 0x80000000u;

// ---- mirrored from rocclr/platform/prof_protocol.h
// ---------------------------
enum ActivityDomain : uint32_t {
  kDomainHipOps = 2,
  kDomainHipApi = 3,
};

struct ActivityRecord {
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
    uint64_t external_id;
  };
  union {
    size_t bytes;
    const char *kernel_name;
  };
};
constexpr uint32_t kOpDispatch = 0;

// Mirrors hip_prof_api.h. The runtime hands over an uninitialized instance
// and calls whatever phase_enter / phase_exit is left in it.
struct ApiTraceData {
  hip_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;
  void (*phase_enter)(hip_api_id_t, ApiTraceData *);
  void (*phase_exit)(hip_api_id_t, ApiTraceData *);
};

using TracerFn = int (*)(uint32_t, uint32_t, void *);
using RegisterTracerFn = void(__cdecl *)(TracerFn);
using ApiNameFn = const char *(__cdecl *)(uint32_t);

// ---- clock
// ------------------------------------------------------------------- QPC in
// nanoseconds: the clock the Windows HIP runtime reports command timestamps on,
// so host and GPU records share one axis.
uint64_t qpcFrequency() {
  static const uint64_t f = [] {
    LARGE_INTEGER v;
    QueryPerformanceFrequency(&v);
    return static_cast<uint64_t>(v.QuadPart);
  }();
  return f;
}

uint64_t nowNs() {
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  const uint64_t f = qpcFrequency();
  const uint64_t q = static_cast<uint64_t>(c.QuadPart);
  return (q / f) * 1000000000ull + (q % f) * 1000000000ull / f;
}

std::string envString(const char *name) {
  char buf[512];
  DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
}

long envLong(const char *name, long fallback) {
  const std::string v = envString(name);
  return v.empty() ? fallback : std::strtol(v.c_str(), nullptr, 0);
}

// ---- ring
// ----------------------------------------------------------------------
class Ring {
public:
  bool attach(const std::string &session) {
    const std::string name = "Local\\hipep_trace_" + session;
    HANDLE h = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
    if (!h)
      return false;
    void *view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!view) {
      CloseHandle(h);
      return false;
    }
    auto *hdr = static_cast<RingHeader *>(view);
    const uint64_t cap = hdr->capacity;
    if (hdr->magic != kRingMagic || hdr->version != kRingVersion ||
        hdr->recordSize != kRecordSize || cap == 0 || (cap & (cap - 1))) {
      std::fprintf(stderr,
                   "[hipep-trace] ring %s has an unexpected layout; tracing "
                   "disabled\n",
                   name.c_str());
      UnmapViewOfFile(view);
      CloseHandle(h);
      return false;
    }
    mapping_ = h;
    hdr_ = hdr;
    records_ =
        reinterpret_cast<Record *>(static_cast<char *>(view) + kHeaderSize);
    mask_ = cap - 1;
    hdr_->qpcFrequency = qpcFrequency();
    hdr_->producerPid = GetCurrentProcessId();
    setFlag(kFlagAttached);
    return true;
  }

  bool attached() const { return hdr_ != nullptr; }

  void setFlag(uint32_t f) {
    reinterpret_cast<std::atomic<uint32_t> *>(&hdr_->flags)
        ->fetch_or(f, std::memory_order_acq_rel);
  }

  RingHeader *header() { return hdr_; }

  // Lock-free multi-producer reservation. A full ring drops the record and
  // counts it instead of blocking the HIP runtime thread that reported it.
  bool push(uint16_t type, const uint64_t (&f)[6], uint16_t flags = 0) {
    if (!hdr_)
      return false;
    uint64_t pos = hdr_->head.load(std::memory_order_relaxed);
    for (;;) {
      const uint64_t tail = hdr_->tail.load(std::memory_order_acquire);
      // pos may be stale and already behind tail (the collector drained past
      // it); signed, that is not full, and the CAS below refreshes pos.
      if (static_cast<int64_t>(pos - tail) > static_cast<int64_t>(mask_)) {
        hdr_->dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (hdr_->head.compare_exchange_weak(pos, pos + 1,
                                           std::memory_order_acq_rel,
                                           std::memory_order_relaxed))
        break;
    }
    Record &r = records_[pos & mask_];
    r.type = type;
    r.flags = flags;
    r.tid = GetCurrentThreadId();
    std::memcpy(r.f, f, sizeof(r.f));
    r.seq.store(pos + 1, std::memory_order_release);
    return true;
  }

private:
  HANDLE mapping_ = nullptr;
  RingHeader *hdr_ = nullptr;
  Record *records_ = nullptr;
  uint64_t mask_ = 0;
};

Ring g_ring;

// ---- name interning
// ------------------------------------------------------------ Strings travel
// once as kRecName chunks and are referenced by id afterwards. A name whose
// chunks were dropped is forgotten so the next use resends it.
class Names {
public:
  uint32_t byString(const char *s) {
    if (!s)
      return 0;
    AcquireSRWLockShared(&lock_);
    auto it = byString_.find(s);
    const uint32_t found = it != byString_.end() ? it->second : 0;
    ReleaseSRWLockShared(&lock_);
    if (found)
      return found;
    AcquireSRWLockExclusive(&lock_);
    it = byString_.find(s);
    uint32_t id = it != byString_.end() ? it->second : 0;
    if (!id) {
      id = ++next_;
      if (emit(id, s))
        byString_.emplace(s, id);
    }
    ReleaseSRWLockExclusive(&lock_);
    return id;
  }

  // Kernel-name pointers from the runtime are stable per kernel; caching by
  // address keeps the per-dispatch path to one shared-lock hash lookup.
  uint32_t byPointer(const char *s) {
    if (!s)
      return 0;
    AcquireSRWLockShared(&lock_);
    auto it = byPtr_.find(s);
    const uint32_t found = it != byPtr_.end() ? it->second : 0;
    ReleaseSRWLockShared(&lock_);
    if (found)
      return found;
    const uint32_t id = byString(s);
    AcquireSRWLockExclusive(&lock_);
    if (byString_.count(s))
      byPtr_.emplace(s, id);
    ReleaseSRWLockExclusive(&lock_);
    return id;
  }

  // Runs on every HIP call's exit, so the already-sent check is one atomic
  // load.
  void apiName(uint32_t apiId) {
    if (apiId >= kApiIds)
      return;
    std::atomic<uint64_t> &word = apiSent_[apiId / 64];
    const uint64_t bit = 1ull << (apiId % 64);
    if (word.load(std::memory_order_relaxed) & bit)
      return;
    if (word.fetch_or(bit, std::memory_order_acq_rel) & bit)
      return;
    if (!emit(kApiNameBase | apiId, hip_api_name(apiId)))
      word.fetch_and(~bit, std::memory_order_acq_rel);
  }

private:
  static bool emit(uint32_t id, const char *s) {
    const size_t n = std::strlen(s);
    const size_t chunks = n == 0 ? 1 : (n + 39) / 40;
    bool ok = true;
    for (size_t c = 0; c < chunks; ++c) {
      uint64_t f[6] = {};
      f[0] = uint64_t(id) | (uint64_t(c) << 32) | (uint64_t(chunks) << 48);
      const size_t off = c * 40;
      const size_t len = n - off < 40 ? n - off : 40;
      std::memcpy(&f[1], s + off, len);
      ok &= g_ring.push(kRecName, f);
    }
    return ok;
  }

  static constexpr uint32_t kApiIds = 1024;
  SRWLOCK lock_ = SRWLOCK_INIT;
  uint32_t next_ = 0;
  std::unordered_map<std::string, uint32_t> byString_;
  std::unordered_map<const void *, uint32_t> byPtr_;
  std::atomic<uint64_t> apiSent_[kApiIds / 64] = {};
};

Names g_names;

// ---- HIP entry points (resolved, never linked)
// ---------------------------------
struct HipApi {
  HMODULE module = nullptr;
  hipError_t (*deviceSynchronize)() = nullptr;
  hipError_t (*moduleLoadData)(hipModule_t *, const void *) = nullptr;
  hipError_t (*moduleGetFunction)(hipFunction_t *, hipModule_t,
                                  const char *) = nullptr;
  hipError_t (*moduleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void **, void **) = nullptr;
  hipError_t (*getDeviceProperties)(hipDeviceProp_t *, int) = nullptr;
  hipError_t (*getDevice)(int *) = nullptr;
};

HipApi g_hip;
void hookHip();
std::atomic<uint64_t> g_nextCorrelation{1};
std::atomic<bool> g_apiEnabled{false};

void apiPhaseExit(hip_api_id_t id, ApiTraceData *d) {
  const uint32_t apiId = static_cast<uint32_t>(id);
  g_names.apiName(apiId);
  const uint64_t f[6] = {d->api_data.correlation_id,
                         d->phase_enter_timestamp,
                         nowNs(),
                         apiId,
                         0,
                         0};
  g_ring.push(kRecApi, f);
}

int __cdecl tracerCallback(uint32_t domain, uint32_t op, void *data) {
  (void)op;
  if (domain == kDomainHipApi) {
    if (!g_apiEnabled.load(std::memory_order_relaxed))
      return -1;
    if (!data)
      return 0;
    auto *d = static_cast<ApiTraceData *>(data);
    // Setting correlation_id is what makes the GPU commands this call issues
    // carry the same id. phase_enter stays null so no arguments are marshalled.
    d->api_data.correlation_id =
        g_nextCorrelation.fetch_add(1, std::memory_order_relaxed);
    d->phase_data = 0;
    d->phase_enter = nullptr;
    d->phase_exit = apiPhaseExit;
    d->phase_enter_timestamp = nowNs();
    return 0;
  }
  if (domain != kDomainHipOps)
    return -1;
  // A null payload asks whether to trace the command about to be built; 1 is
  // the runtime's commit sentinel.
  if (!data || data == reinterpret_cast<void *>(uintptr_t{1}))
    return 0;
  const auto *r = static_cast<const ActivityRecord *>(data);
  const bool dispatch = r->op == kOpDispatch;
  const uint64_t tail = dispatch ? g_names.byPointer(r->kernel_name)
                                 : static_cast<uint64_t>(r->bytes);
  const uint64_t f[6] = {
      r->correlation_id,
      r->begin_ns,
      r->end_ns,
      r->queue_id,
      uint64_t(r->kind) | (uint64_t(r->op & 0xff) << 32) |
          (uint64_t(uint32_t(r->device_id) & 0xffffff) << 40),
      tail};
  g_ring.push(kRecOp, f);
  return 0;
}

// The API domain is safe only if the runtime enumerates HIP calls exactly as
// the header compiled here does; every overlapping id is compared.
bool verifyApiTable(HMODULE hip, RingHeader *hdr) {
  auto runtimeName =
      reinterpret_cast<ApiNameFn>(GetProcAddress(hip, "hipApiName"));
  if (!runtimeName) {
    std::snprintf(hdr->apiDetail, sizeof(hdr->apiDetail),
                  "hipApiName not exported");
    return false;
  }
  int runtimeLast = 0;
  for (uint32_t i = 1; i < 4096; ++i) {
    const char *n = runtimeName(i);
    if (n && n[0] && std::strcmp(n, "unknown") != 0)
      runtimeLast = static_cast<int>(i);
  }
  const int oursLast = static_cast<int>(HIP_API_ID_LAST);
  const int overlap = runtimeLast < oursLast ? runtimeLast : oursLast;
  uint32_t compared = 0, mismatches = 0;
  for (int i = 1; i <= overlap; ++i) {
    const char *rt = runtimeName(static_cast<uint32_t>(i));
    const char *us = hip_api_name(static_cast<uint32_t>(i));
    if (!rt || !us || !std::strcmp(rt, "unknown") ||
        !std::strcmp(us, "unknown"))
      continue;
    ++compared;
    if (std::strcmp(rt, us) != 0) {
      if (mismatches == 0)
        std::snprintf(hdr->apiDetail, sizeof(hdr->apiDetail),
                      "id %d: runtime=%s ours=%s", i, rt, us);
      ++mismatches;
    }
  }
  hdr->apiCompared = compared;
  hdr->apiMismatches = mismatches;
  // The tables may differ in length: ids are only ever appended, so a runtime
  // with fewer (or more) of them is compatible as long as every shared id
  // names the same call. The union is sized by its largest member, which the
  // appended calls have not changed so far; trace.validate's causality check
  // is what would expose a layout mismatch.
  const bool ok = compared > 0 && mismatches == 0;
  if (ok)
    std::snprintf(hdr->apiDetail, sizeof(hdr->apiDetail),
                  "%u ids identical (runtime last %d, ours %d)", compared,
                  runtimeLast, oursLast);
  return ok;
}

// ---- op fence
// ------------------------------------------------------------------
struct FenceConfig {
  std::string op;
  std::string graphFilter;
  long skip = 0;
  long afterRuns = 0;
  long sleepMs = 200;
  bool marker = false;
};

FenceConfig g_fence;
std::atomic<long> g_runs{-1};
std::atomic<long> g_fenceMatches{0};
std::atomic<bool> g_fenceFired{false};
thread_local int t_depth = 0;

constexpr const char kMarkerKernel[] = "hipep_trace_sqtt_marker";

// Compiles and launches the marker kernel once, through whichever hiprtc
// library sits beside the loaded HIP runtime. Returns whether it was enqueued.
bool launchMarker() {
  if (!g_hip.moduleLoadData || !g_hip.moduleGetFunction ||
      !g_hip.moduleLaunchKernel)
    return false;
  static hipFunction_t fn = nullptr;
  if (!fn) {
    char path[MAX_PATH];
    if (!GetModuleFileNameA(g_hip.module, path, MAX_PATH))
      return false;
    std::string dir(path);
    dir = dir.substr(0, dir.find_last_of("\\/") + 1);
    WIN32_FIND_DATAA fd;
    HANDLE it = FindFirstFileA((dir + "hiprtc*.dll").c_str(), &fd);
    HMODULE rtc = nullptr;
    if (it != INVALID_HANDLE_VALUE) {
      do {
        if (!std::strstr(fd.cFileName, "builtins"))
          rtc = LoadLibraryA((dir + fd.cFileName).c_str());
      } while (!rtc && FindNextFileA(it, &fd));
      FindClose(it);
    }
    if (!rtc) {
      std::fprintf(stderr, "[hipep-trace] marker: no hiprtc beside %s\n", path);
      return false;
    }
    using Program = void *;
    auto create =
        reinterpret_cast<int (*)(Program *, const char *, const char *, int,
                                 const char **, const char **)>(
            GetProcAddress(rtc, "hiprtcCreateProgram"));
    auto compile = reinterpret_cast<int (*)(Program, int, const char **)>(
        GetProcAddress(rtc, "hiprtcCompileProgram"));
    auto codeSize = reinterpret_cast<int (*)(Program, size_t *)>(
        GetProcAddress(rtc, "hiprtcGetCodeSize"));
    auto code = reinterpret_cast<int (*)(Program, char *)>(
        GetProcAddress(rtc, "hiprtcGetCode"));
    auto destroy = reinterpret_cast<int (*)(Program *)>(
        GetProcAddress(rtc, "hiprtcDestroyProgram"));
    if (!create || !compile || !codeSize || !code || !destroy)
      return false;
    std::string arch;
    int dev = 0;
    hipDeviceProp_t props{};
    if (g_hip.getDevice && g_hip.getDeviceProperties &&
        g_hip.getDevice(&dev) == hipSuccess &&
        g_hip.getDeviceProperties(&props, dev) == hipSuccess) {
      arch = props.gcnArchName;
      arch = arch.substr(0, arch.find(':'));
    }
    const std::string src = std::string("extern \"C\" __global__ void ") +
                            kMarkerKernel + "() {}\n";
    Program prog = nullptr;
    if (create(&prog, src.c_str(), "hipep_trace_marker.hip", 0, nullptr,
               nullptr) != 0)
      return false;
    const std::string archOpt = "--offload-arch=" + arch;
    const char *opts[] = {archOpt.c_str()};
    const bool built = compile(prog, arch.empty() ? 0 : 1, opts) == 0;
    std::string blob;
    size_t n = 0;
    if (built && codeSize(prog, &n) == 0 && n) {
      blob.resize(n);
      code(prog, &blob[0]);
    }
    destroy(&prog);
    hipModule_t mod = nullptr;
    if (blob.empty() || g_hip.moduleLoadData(&mod, blob.data()) != hipSuccess ||
        g_hip.moduleGetFunction(&fn, mod, kMarkerKernel) != hipSuccess) {
      std::fprintf(stderr, "[hipep-trace] marker: build or load failed (%s)\n",
                   arch.c_str());
      fn = nullptr;
      return false;
    }
  }
  auto *stream = static_cast<hipStream_t>(hipdnn_ep_get_current_stream());
  return g_hip.moduleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, stream, nullptr,
                                  nullptr) == hipSuccess;
}

void maybeFence(const char *label) {
  // label is "hip_ep.op.<op>[ <shape>]".
  const char *op = label + 10;
  const size_t n = std::strcspn(op, " ");
  if (n != g_fence.op.size() || std::strncmp(op, g_fence.op.c_str(), n) != 0)
    return;
  if (g_fenceFired.load(std::memory_order_acquire))
    return;
  const long run = g_runs.load(std::memory_order_acquire);
  if (run < g_fence.afterRuns)
    return;
  const long instance = g_fenceMatches.fetch_add(1, std::memory_order_acq_rel);
  if (instance < g_fence.skip)
    return;
  bool expected = false;
  if (!g_fenceFired.compare_exchange_strong(expected, true))
    return;
  hookHip();
  if (g_hip.deviceSynchronize)
    (void)g_hip.deviceSynchronize();
  const bool marker = g_fence.marker && launchMarker();
  const uint64_t f[6] = {nowNs(),
                         g_names.byString(g_fence.op.c_str()),
                         static_cast<uint64_t>(run),
                         static_cast<uint64_t>(instance),
                         marker ? 1u : 0u,
                         0};
  g_ring.push(kRecFence, f);
  std::fprintf(stderr,
               "[RGP_FENCE_ARMED] op=%s run=%ld instance=%ld marker=%d "
               "idling sleep_ms=%ld\n",
               g_fence.op.c_str(), run, instance, marker ? 1 : 0,
               g_fence.sleepMs);
  std::fflush(stderr);
  Sleep(static_cast<DWORD>(g_fence.sleepMs));
}

void countRun(const char *label) {
  // label is "hip_ep.graph.<model> sub=<n>".
  if (g_fence.graphFilter.empty()
          ? std::strstr(label, " sub=0") != nullptr
          : std::strstr(label, g_fence.graphFilter.c_str()) != nullptr)
    g_runs.fetch_add(1, std::memory_order_acq_rel);
}

// ---- lifetime
// ------------------------------------------------------------------
HMODULE findHip() {
  HMODULE m = GetModuleHandleA("amdhip64_7.dll");
  return m ? m : GetModuleHandleA("amdhip64.dll");
}

// Resolves the HIP entry points and, when collecting, registers the activity
// callback. Idempotent; never called with the loader lock held, because it
// calls into the runtime.
void hookHip() {
  static std::atomic<bool> done{false};
  static SRWLOCK lock = SRWLOCK_INIT;
  if (done.load(std::memory_order_acquire))
    return;
  AcquireSRWLockExclusive(&lock);
  HMODULE m = done.load() ? nullptr : findHip();
  if (m) {
    auto get = [m](const char *name) { return GetProcAddress(m, name); };
    g_hip.deviceSynchronize =
        reinterpret_cast<decltype(g_hip.deviceSynchronize)>(
            get("hipDeviceSynchronize"));
    g_hip.moduleLoadData = reinterpret_cast<decltype(g_hip.moduleLoadData)>(
        get("hipModuleLoadData"));
    g_hip.moduleGetFunction =
        reinterpret_cast<decltype(g_hip.moduleGetFunction)>(
            get("hipModuleGetFunction"));
    g_hip.moduleLaunchKernel =
        reinterpret_cast<decltype(g_hip.moduleLaunchKernel)>(
            get("hipModuleLaunchKernel"));
    g_hip.getDeviceProperties =
        reinterpret_cast<decltype(g_hip.getDeviceProperties)>(
            get("hipGetDevicePropertiesR0600"));
    g_hip.getDevice =
        reinterpret_cast<decltype(g_hip.getDevice)>(get("hipGetDevice"));
    g_hip.module = m;
    if (g_ring.attached()) {
      auto reg =
          reinterpret_cast<RegisterTracerFn>(get("hipRegisterTracerCallback"));
      if (reg) {
        if (envLong("HIPEP_TRACE_API", 1) != 0 &&
            verifyApiTable(m, g_ring.header())) {
          g_apiEnabled.store(true);
          g_ring.setFlag(kFlagApi);
        }
        reg(&tracerCallback);
        g_ring.setFlag(kFlagTracer);
      } else {
        std::fprintf(stderr, "[hipep-trace] hipRegisterTracerCallback not "
                             "exported; GPU commands will not be traced\n");
      }
    }
    done.store(true, std::memory_order_release);
  }
  ReleaseSRWLockExclusive(&lock);
}

// The EP DLL delay-loads the HIP runtime, so it is usually not mapped yet when
// this DLL initializes. The loader notification only signals (it runs under
// the loader lock); a watcher thread then hooks the runtime, whose
// GetProcAddress waits for that lock, i.e. for the runtime's own init.
HANDLE g_hipLoaded = nullptr;

struct LdrUnicodeString {
  USHORT length;
  USHORT maximum;
  PWSTR buffer;
};
struct LdrLoadedData {
  ULONG flags;
  const LdrUnicodeString *fullName;
  const LdrUnicodeString *baseName;
  PVOID base;
  ULONG size;
};

VOID CALLBACK onDllLoaded(ULONG reason, const LdrLoadedData *data, PVOID) {
  if (reason != 1 || !data || !data->baseName || !data->baseName->buffer)
    return;
  const size_t n = data->baseName->length / sizeof(wchar_t);
  const wchar_t *s = data->baseName->buffer;
  for (size_t i = 0; i + 8 <= n; ++i)
    if (!_wcsnicmp(s + i, L"amdhip64", 8)) {
      SetEvent(g_hipLoaded);
      return;
    }
}

DWORD WINAPI hipWatcher(LPVOID) {
  for (;;) {
    WaitForSingleObject(g_hipLoaded, 2);
    if (findHip()) {
      hookHip();
      return 0;
    }
  }
}

void watchForHip() {
  g_hipLoaded = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  using Callback = VOID(CALLBACK *)(ULONG, const LdrLoadedData *, PVOID);
  using Register = LONG(NTAPI *)(ULONG, Callback, PVOID, PVOID *);
  auto reg = reinterpret_cast<Register>(GetProcAddress(
      GetModuleHandleW(L"ntdll.dll"), "LdrRegisterDllNotification"));
  static PVOID cookie = nullptr;
  if (reg)
    reg(0, onDllLoaded, nullptr, &cookie);
  if (HANDLE t = CreateThread(nullptr, 0, hipWatcher, nullptr, 0, nullptr))
    CloseHandle(t);
}

struct Shim {
  bool active = false;

  Shim() {
    const std::string session = envString("HIPEP_TRACE");
    g_fence.op = envString("HIPEP_TRACE_FENCE");
    g_fence.graphFilter = envString("HIPEP_TRACE_FENCE_GRAPH");
    g_fence.skip = envLong("HIPEP_TRACE_FENCE_SKIP", 0);
    g_fence.afterRuns = envLong("HIPEP_TRACE_FENCE_AFTER_RUNS", 0);
    g_fence.sleepMs = envLong("HIPEP_TRACE_FENCE_MS", 200);
    g_fence.marker = envLong("HIPEP_TRACE_SQTT_MARKER", 0) != 0;
    if (session.empty() && g_fence.op.empty())
      return;
    active = true;
    if (!session.empty() && !g_ring.attach(session)) {
      std::fprintf(stderr,
                   "[hipep-trace] no collector ring for session '%s'; "
                   "start the collector first. Tracing disabled.\n",
                   session.c_str());
    }
    if (findHip())
      hookHip();
    else
      watchForHip();
  }

  ~Shim() {
    if (g_ring.attached())
      g_ring.setFlag(kFlagDetached);
  }
};

Shim &shim() {
  static Shim s;
  return s;
}

// Constructed at DLL load so commands issued while the model loads are traced
// too; the bridge functions below also reach it lazily.
const bool g_shimInit = (shim(), true);

bool tracing() { return g_ring.attached() || !g_fence.op.empty(); }

// One device summary per process, for the agent tables of the exported trace.
// Sent from the first range push: by then the runtime is initialized on the
// thread that uses it, whereas querying from the loader watcher would
// initialize it early on a thread of the shim's own.
void sendDeviceOnce() {
  static std::atomic<bool> sent{false};
  if (!g_ring.attached() || sent.exchange(true))
    return;
  hookHip();
  int dev = 0;
  hipDeviceProp_t p{};
  if (!g_hip.getDevice || !g_hip.getDeviceProperties ||
      g_hip.getDevice(&dev) != hipSuccess ||
      g_hip.getDeviceProperties(&p, dev) != hipSuccess)
    return;
  auto lo_hi = [](uint64_t lo, uint64_t hi) {
    return (lo & 0xffffffffull) | (hi << 32);
  };
  const uint64_t f[6] = {
      g_names.byString(p.name),
      g_names.byString(p.gcnArchName),
      lo_hi(uint64_t(p.multiProcessorCount), uint64_t(p.warpSize)),
      lo_hi(uint64_t(p.maxThreadsPerMultiProcessor),
            uint64_t(p.maxSharedMemoryPerMultiProcessor)),
      lo_hi(uint64_t(p.clockRate), uint64_t(p.memoryClockRate)),
      lo_hi(uint64_t(p.memoryBusWidth), uint64_t(dev))};
  g_ring.push(kRecDevice, f);
}

} // namespace

extern "C" int hipdnn_ep_trace_shim_active(void) {
  return shim().active && tracing() ? 1 : 0;
}

extern "C" __declspec(dllexport) int roctxRangePushA(const char *message) {
  if (!shim().active || !tracing() || !message)
    return -1;
  sendDeviceOnce();
  if (!std::strncmp(message, "hip_ep.graph.", 13))
    countRun(message);
  else if (!g_fence.op.empty() && !std::strncmp(message, "hip_ep.op.", 10))
    maybeFence(message);
  const int depth = t_depth++;
  const uint64_t f[6] = {
      nowNs(), g_names.byString(message), static_cast<uint64_t>(depth), 0, 0,
      0};
  g_ring.push(kRecPush, f);
  return depth;
}

extern "C" __declspec(dllexport) int roctxRangePop(void) {
  if (!shim().active || !tracing())
    return -1;
  const int depth = t_depth > 0 ? --t_depth : 0;
  const uint64_t f[6] = {nowNs(), static_cast<uint64_t>(depth), 0, 0, 0, 0};
  g_ring.push(kRecPop, f);
  return depth;
}

extern "C" __declspec(dllexport) void roctxMarkA(const char *message) {
  if (!shim().active || !tracing() || !message)
    return;
  const uint64_t f[6] = {nowNs(), g_names.byString(message), 0, 0, 0, 0};
  g_ring.push(kRecMark, f);
}
