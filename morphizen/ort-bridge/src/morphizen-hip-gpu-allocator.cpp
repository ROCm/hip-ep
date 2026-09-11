/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HIP-runtime backed implementation of OrtAllocator and OrtDataTransferImpl
// for the MorphiZen EP. Mirrors onnxruntime's NvTensorRtRtx EP factory
// (nv_provider_factory.cc, NvTrtRtxOrtAllocator / NvTrtRtxDataTransferImpl)
// translated to the HIP runtime API.

#include "./morphizen-hip-gpu-allocator.hpp"

#include "./ort-api-version.hpp"
#include <glog/logging.h>
#include <hip/hip_runtime.h>

// ORT_API_MANUAL_INIT must be set before <onnxruntime_cxx_api.h> to match
// the rest of morphizen, which initializes the Ort::api global manually
// via Ort::InitApi() in ort-bridge.cpp at DLL load. Without this define
// the C++ wrapper's GetApi() would compile against an internal default
// initializer and link-mismatch with main.cpp.obj (LNK2038 on
// ORT_API_MANUAL_INIT enabled vs disabled).
#define ORT_API_MANUAL_INIT 1
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstring>
#include <string>

// For sizing the large-pool retention cap from physical RAM.
#if defined(_WIN32)
// NOMINMAX / WIN32_LEAN_AND_MEAN keep windows.h from defining min/max macros
// and dragging in winsock, either of which breaks the C++ headers above.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace morphizen {

namespace {

// Convert a non-success hipError_t into an OrtStatus*. The caller owns the
// returned status (must release via ort_api.ReleaseStatus).
inline OrtStatus *MakeHipStatus(const OrtApi &api, hipError_t err,
                                const char *what) {
  std::string msg = std::string("[MorphiZen HIP] ") + what +
                    " failed: " + hipGetErrorString(err) + " (" +
                    std::to_string(static_cast<int>(err)) + ")";
  return api.CreateStatus(ORT_FAIL, msg.c_str());
}

// Lightweight RAII helper to set the active HIP device for the duration of
// a single allocator call. We can't rely on the process-wide hipSetDevice
// state because ORT may interleave calls into allocators bound to different
// GPUs from the same thread (e.g. a session that uses MorphiZen on GPU 0
// and another EP on GPU 1). The device id is read from the OrtMemoryInfo
// that was passed to HipGpuAllocator's constructor; -1 means "leave the
// current HIP device alone" (the OrtMemoryInfo didn't carry a device id).
struct ScopedDevice {
  explicit ScopedDevice(int device_id) {
    if (device_id >= 0) {
      (void)hipSetDevice(device_id);
    }
  }
};

// Read the device id off an OrtMemoryInfo, returning -1 on null / on any
// underlying ORT error. We swallow the (rare) error path because this is
// invoked from the allocator constructor and we'd rather degrade to "use
// whatever HIP device the caller had selected" than throw across the EP
// factory ABI boundary. Real production callers go through
// MorphiZenEpFactory::CreateMemoryInfo_V2, which always sets a valid id.
// Map a request (or an allocated buffer's size) to a fixed size class. Returns
// the index into kSizeClasses of the first class whose capacity is >= size, or
// -1 when size exceeds the largest class (a "large" buffer handled best-fit).
// Because pooled buffers are allocated at exactly their class capacity, calling
// this on an allocated buffer's stored size recovers the class it belongs to.
int SizeClassIndex(size_t size) noexcept {
  for (int i = 0; i < static_cast<int>(kNumSizeClasses); ++i) {
    if (size <= kSizeClasses[i]) {
      return i;
    }
  }
  return -1;
}

// Round a > 16 MB request up to a poolable capacity: eighth-of-octave steps
// applied to the request plus a 1/64 growth headroom.
//
// The headroom is what lets one generation of KV growth share a single
// capacity across an octave edge, where the step doubles. Without it a request
// landing just under a power of two rounds to exactly that power of two and
// the next token pushes it into the next class. That is the common case, not a
// rare alignment: a KV tensor's bytes per token is a power of two and so are
// benchmark prompt lengths, so the two land exactly on an edge.
constexpr size_t LargeCapacity(size_t size) noexcept {
  const size_t with_headroom = size + size / 64;
  // Highest power of two <= with_headroom. Callers only reach here above
  // 16 MB, so `step` cannot round down to zero.
  size_t octave = 1;
  while (octave <= with_headroom / 2) {
    octave *= 2;
  }
  const size_t step = octave / 8;
  return ((with_headroom + step - 1) / step) * step;
}

// Checked at compile time because no unit-test target covers this file and
// returning less than the request would overflow the buffer. Swept across
// every octave the large pool can reach rather than at sizes taken from one
// model, so the guarantee is a property of the function.
constexpr bool LargeCapacityHolds() {
  for (size_t n = 16ull << 20; n <= 16ull << 30; n += n / 8) {
    const size_t c = LargeCapacity(n);
    if (c < n + n / 64 ||           // covers the request, keeps headroom
        c > n + n / 8 + n / 64 ||   // waste bounded
        c > LargeCapacity(n + 1)) { // monotonic
      return false;
    }
  }
  return true;
}
static_assert(LargeCapacityHolds(),
              "LargeCapacity must cover the request with its growth headroom, "
              "bound its waste and stay monotonic at every octave");
// The alignment the headroom was added for, kept because it is easy to
// reintroduce: 4096 bytes per token at 16379 tokens sits just under the 64 MiB
// octave edge and one 128-token generation grows it past that edge.
static_assert(LargeCapacity(4096ull * 16379) == LargeCapacity(4096ull * 16507),
              "one generation of KV growth must map to one capacity");

size_t PhysicalMemoryBytes() noexcept {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return static_cast<size_t>(status.ullTotalPhys);
  }
  return 0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
  }
  return 0;
#endif
}

// Retention has to clear one full KV set or every step evicts what the next
// one wants, which is the churn this pooling removes. One set measured ~3.1
// GiB on the 26B A4B and ~5.2 GiB on the 12B at a 16 K context, against the
// 7.96 GiB an eighth comes to on the 64 GB box they ran on. Scales with RAM.
//
// 16 K is the only context measured. Extrapolating the sets linearly, the 12B
// crosses this cap near 24 K and the 26B near 41 K; above that retention keeps
// evicting what the next step wants and decode degrades toward a page-pin per
// step. FreeImpl's warning does not fire there -- it catches a single buffer
// too large to retain, not a live set that no longer fits.
constexpr size_t kRamDivisor = 8;
constexpr size_t kFallbackCap = 8ull << 30;
constexpr size_t kMinCap = 1ull << 30;

// Hand a batch of buffers back to the driver. Called with pool_mutex_
// released: hipHostFree unpins pages and is as heavyweight as the
// hipHostMalloc that pinned them, so holding the lock across a batch would
// serialize every other allocation behind it.
void ReleaseToDriver(const std::vector<void *> &buffers, int device_id) {
  if (buffers.empty()) {
    return;
  }
  ScopedDevice _(device_id);
  for (void *p : buffers) {
    (void)hipHostFree(p);
  }
}

int TryGetDeviceId(const OrtMemoryInfo *memory_info) noexcept {
  if (memory_info == nullptr) {
    return -1;
  }
  try {
    // Ort::ConstMemoryInfo is an Unowned wrapper around the raw pointer
    // (see onnxruntime_cxx_api.h: ConstMemoryInfo =
    // MemoryInfoImpl<Unowned<...>>). Construction is free; GetDeviceId calls
    // OrtApi::MemoryInfoGetId and ThrowOnError-routes a failure status into
    // Ort::Exception. Ort::api was wired up by Ort::InitApi() in ort-bridge.cpp
    // at DLL load.
    return Ort::ConstMemoryInfo(memory_info).GetDeviceId();
  } catch (const Ort::Exception &) {
    return -1;
  }
}

} // namespace

// =============== LargePoolBudget ===============

LargePoolBudget::LargePoolBudget() noexcept {
  const size_t phys = PhysicalMemoryBytes();
  cap_ = std::max((phys == 0) ? kFallbackCap : phys / kRamDivisor, kMinCap);
}

bool LargePoolBudget::TryReserve(size_t n) noexcept {
  size_t cur = used_.load(std::memory_order_relaxed);
  for (;;) {
    // Subtraction rather than cur + n: used_ never exceeds cap_, so this
    // cannot overflow the way the addition could.
    if (n > cap_ - cur) {
      return false;
    }
    if (used_.compare_exchange_weak(cur, cur + n, std::memory_order_relaxed)) {
      return true;
    }
  }
}

void LargePoolBudget::Release(size_t n) noexcept {
  size_t cur = used_.load(std::memory_order_relaxed);
  while (!used_.compare_exchange_weak(cur, (n > cur) ? 0 : cur - n,
                                      std::memory_order_relaxed)) {
  }
}

// =============== HipGpuAllocator ===============

HipGpuAllocator::HipGpuAllocator(const OrtMemoryInfo *memory_info,
                                 const OrtApi & /*api*/,
                                 LargePoolBudget &budget)
    : budget_{&budget}, memory_info_{memory_info},
      device_id_{TryGetDeviceId(memory_info)} {
  version = NegotiatedOrtApiVersion();
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
}

void HipGpuAllocator::DropRetained(void *ptr, std::vector<void *> &out) {
  const size_t bytes = blocks_[ptr].bytes;
  retained_large_bytes_ -= bytes;
  budget_->Release(bytes);
  blocks_.erase(ptr);
  out.push_back(ptr);
}

void HipGpuAllocator::TrimStaleLarge(std::vector<void *> &out) {
  // A step allocates and frees each live buffer once, so a step is about
  // 2 * outstanding_large_ ops and two steps four times that. Two rather than
  // one leaves room for a class holding a single buffer, which is touched only
  // twice per step. The floor keeps a session whose first step is still
  // allocating from trimming classes it is in the middle of filling.
  const uint64_t stale_after = std::max<uint64_t>(256, 4 * outstanding_large_);
  if (large_ops_ < stale_after) {
    return;
  }
  const uint64_t cutoff = large_ops_ - stale_after;
  for (auto &entry : large_classes_) {
    if (entry.second.free.empty() || entry.second.last_used > cutoff) {
      continue;
    }
    for (void *ptr : entry.second.free) {
      DropRetained(ptr, out);
    }
    entry.second.free.clear();
  }
}

bool HipGpuAllocator::EvictLruLarge(size_t keep, std::vector<void *> &out) {
  // `keep` is skipped: evicting from the class we are about to retain into
  // would free a buffer of exactly the size we are making room for.
  auto victim = large_classes_.end();
  for (auto it = large_classes_.begin(); it != large_classes_.end(); ++it) {
    if (it->first == keep || it->second.free.empty()) {
      continue;
    }
    if (victim == large_classes_.end() ||
        it->second.last_used < victim->second.last_used) {
      victim = it;
    }
  }
  if (victim == large_classes_.end()) {
    return false;
  }
  // One buffer at a time, so a class merely older than the incoming one is not
  // emptied further than the caller needs.
  void *ptr = victim->second.free.back();
  victim->second.free.pop_back();
  DropRetained(ptr, out);
  return true;
}

void *ORT_API_CALL HipGpuAllocator::AllocImpl(OrtAllocator *this_,
                                              size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);

  const int cls = SizeClassIndex(size);
  // Bytes to allocate on a miss, and the large capacity class this request
  // belongs to (0 for the fixed table). A fixed class always rounds to its
  // capacity; a large request is served at exactly its size until that
  // capacity is seen to serve more than one, so a model whose tensors never
  // change size pays no rounding waste.
  size_t alloc_size = (cls >= 0) ? kSizeClasses[cls] : size;
  size_t klass = 0;

  std::vector<void *> stale;
  void *pooled = nullptr;
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    if (cls >= 0) {
      auto &fl = self->free_lists_[cls];
      if (!fl.empty()) {
        pooled = fl.back();
        fl.pop_back();
      }
    } else {
      klass = LargeCapacity(size);
      ++self->large_ops_;
      // Trimmed here rather than on free: a request for a *different* capacity
      // is what reveals one the model has outgrown, and nothing else in the
      // allocator would notice.
      self->TrimStaleLarge(stale);
      auto &state = self->large_classes_[klass];
      // Touched even on a miss: a class whose buffers are all checked out is
      // in active use, and trimming it would be exactly wrong.
      state.last_used = self->large_ops_;

      if (state.witness_size == 0) {
        state.witness_size = size;
      } else if (state.witness_size != size) {
        // Two sizes in one capacity: the tensor behind it is moving, so exact
        // sizes will never pool again and this class starts rounding.
        state.rounded = true;
      }
      alloc_size = state.rounded ? klass : size;

      // Anything retained at a size this class no longer serves is dropped
      // here rather than matched against: it can only be left over from
      // before the flip to rounding, and nothing will ask for it again.
      while (!state.free.empty()) {
        void *cand = state.free.back();
        state.free.pop_back();
        const size_t cand_bytes = self->blocks_[cand].bytes;
        self->retained_large_bytes_ -= cand_bytes;
        self->budget_->Release(cand_bytes);
        if (cand_bytes == alloc_size) {
          pooled = cand;
          break;
        }
        self->blocks_.erase(cand);
        stale.push_back(cand);
      }
      if (pooled != nullptr) {
        ++self->outstanding_large_;
      }
    }
  }
  ReleaseToDriver(stale, self->device_id_);
  if (pooled != nullptr) {
    return pooled;
  }

  ScopedDevice _(self->device_id_);

  // On AMD APU iGPU (the only hardware MorphiZen EP currently targets) the
  // GPU shares physical memory with the CPU, so we use hipHostMalloc(Mapped)
  // for both DEFAULT and HOST_ACCESSIBLE OrtMemoryInfos: the same pointer is
  // dereferenceable from CPU and GPU. (hipMalloc would return a GPU-only
  // virtual address that the host cannot deref, which crashes OGA's KV
  // cache zero-init.)
  //
  // TODO(discrete-gpu): on a discrete AMD GPU (PCIe), DEFAULT memory should
  // be hipMalloc'd into VRAM for locality + to avoid the host coherency
  // traffic. HOST_ACCESSIBLE should keep using hipHostMalloc(Mapped). To
  // do that we need to (a) read OrtMemoryInfoGetMemType from memory_info_
  // and branch here, and (b) update OGA's MorphiZenEP::Memory struct to
  // stop aliasing p_cpu_ = p_device_ and instead route Zero / CopyDeviceToCpu
  // / CopyCpuToDevice through the HIP runtime. Tracked separately because
  // the OGA-side change cuts across the smartptrs / DeviceBuffer abstraction.
  void *ptr = nullptr;
  hipError_t err = hipHostMalloc(&ptr, alloc_size,
                                 hipHostMallocMapped | hipHostMallocCoherent);
  if (err != hipSuccess) {
    LOG(ERROR) << "[MorphiZen HIP] hipHostMalloc(Mapped|Coherent) failed for "
               << alloc_size << " bytes (device " << self->device_id_
               << "): " << hipGetErrorString(err);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    self->blocks_[ptr] = Block{alloc_size, klass};
    if (cls < 0) {
      ++self->outstanding_large_;
    }
  }
  return ptr;
}

void ORT_API_CALL HipGpuAllocator::FreeImpl(OrtAllocator *this_, void *p) {
  if (p == nullptr) {
    return;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);
  // Buffers bound for the driver: p itself when it cannot be retained, plus
  // whatever eviction dropped to make room for it.
  std::vector<void *> to_release;
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    auto it = self->blocks_.find(p);
    if (it == self->blocks_.end()) {
      // Defensive: a pointer we never handed out. Release rather than leak.
      to_release.push_back(p);
    } else if (it->second.klass == 0) {
      // Fixed class: back on its free list and still tracked, so the
      // destructor releases it. Never returned to the driver here. Safe to
      // reuse without a stream sync -- see the pool comment in the header.
      self->free_lists_[static_cast<size_t>(SizeClassIndex(it->second.bytes))]
          .push_back(p);
    } else {
      const Block blk = it->second;
      ++self->large_ops_;
      if (self->outstanding_large_ > 0) {
        --self->outstanding_large_;
      }
      auto &state = self->large_classes_[blk.klass];
      state.last_used = self->large_ops_;
      // A buffer at a size its class no longer serves is left over from before
      // the flip to rounding; retaining it would hold pages nothing can ask
      // for. The rest are retained at whatever size they were allocated,
      // which is the request size itself while the class is still fixed.
      const bool servable =
          blk.bytes == (state.rounded ? blk.klass : state.witness_size);
      bool reserved = false;
      if (servable) {
        // Reaching the cap evicts the least-recently-used capacity rather than
        // refusing to retain this one, so the buffers that survive are the
        // ones still in rotation.
        reserved = self->budget_->TryReserve(blk.bytes);
        while (!reserved && self->EvictLruLarge(blk.klass, to_release)) {
          reserved = self->budget_->TryReserve(blk.bytes);
        }
      }
      if (reserved) {
        state.free.push_back(p);
        self->retained_large_bytes_ += blk.bytes;
      } else {
        if (servable && !self->warned_cap_) {
          // Nothing left to evict and still no room: the live set alone
          // exceeds the cap, so from here every step re-pins what it just
          // released. That is pre-pooling behavior and the one state worth
          // warning about.
          self->warned_cap_ = true;
          LOG(WARNING) << "[MorphiZen HIP] large-buffer pool cannot retain a "
                       << (blk.bytes >> 20) << " MB buffer within its "
                       << (self->budget_->cap() >> 20)
                       << " MB cap even after evicting every other capacity; "
                          "releasing it to the driver. The buffers this model "
                          "cycles through do not fit in what this machine's "
                          "memory allows us to retain, so decode pays a full "
                          "page-pin per step.";
        }
        self->blocks_.erase(p);
        to_release.push_back(p);
      }
    }
  }
  ReleaseToDriver(to_release, self->device_id_);
}

HipGpuAllocator::~HipGpuAllocator() {
  ScopedDevice _(device_id_);
  std::lock_guard<std::mutex> lk(pool_mutex_);
  // blocks_ rather than the free lists, so a buffer still checked out at
  // teardown is released instead of leaked.
  for (const auto &kv : blocks_) {
    hipError_t err = hipHostFree(kv.first);
    if (err != hipSuccess) {
      LOG(WARNING) << "[MorphiZen HIP] hipHostFree failed (device "
                   << device_id_ << "): " << hipGetErrorString(err);
    }
  }
  for (auto &fl : free_lists_) {
    fl.clear();
  }
  large_classes_.clear();
  blocks_.clear();
  // Hand back exactly this instance's share, so a session that comes and goes
  // does not permanently consume another's headroom.
  budget_->Release(retained_large_bytes_);
  retained_large_bytes_ = 0;
}

const OrtMemoryInfo *ORT_API_CALL
HipGpuAllocator::InfoImpl(const OrtAllocator *this_) {
  return static_cast<const HipGpuAllocator *>(this_)->memory_info_;
}

void *ORT_API_CALL HipGpuAllocator::ReserveImpl(OrtAllocator *this_,
                                                size_t size) {
  // No special reservation strategy; behave like Alloc.
  return AllocImpl(this_, size);
}

// =============== HipDataTransferImpl ===============

HipDataTransferImpl::HipDataTransferImpl(const OrtApi &ort_api_in)
    : ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
  ort_version_supported = NegotiatedOrtApiVersion();
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

bool HipDataTransferImpl::CanCopyImpl(
    const OrtDataTransferImpl *this_ptr,
    const OrtMemoryDevice *src_memory_device,
    const OrtMemoryDevice *dst_memory_device) noexcept {
  const auto &impl = *static_cast<const HipDataTransferImpl *>(this_ptr);

  OrtMemoryInfoDeviceType src_type =
      impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_type =
      impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

  // We only support copies that involve our GPU; any GPU vendor != AMD on
  // either side means another EP should handle it. AMD GPUs (discrete and
  // integrated/APU) report PCI vendor id 0x1002 (== OrtDevice::VendorIds::AMD),
  // matching MorphiZenEpFactory::vendor_id_. (0x1022 is the AuthenticAMD
  // CPU/NPU vendor id and is not used here.)
  constexpr uint32_t kAmdGpuVendorId = 0x1002;
  uint32_t src_vendor_id =
      impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
  uint32_t dst_vendor_id =
      impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

  if ((src_type == OrtMemoryInfoDeviceType_GPU &&
       src_vendor_id != kAmdGpuVendorId) ||
      (dst_type == OrtMemoryInfoDeviceType_GPU &&
       dst_vendor_id != kAmdGpuVendorId)) {
    return false;
  }

  return (src_type == OrtMemoryInfoDeviceType_GPU &&
          dst_type == OrtMemoryInfoDeviceType_GPU) ||
         (src_type == OrtMemoryInfoDeviceType_GPU &&
          dst_type == OrtMemoryInfoDeviceType_CPU) ||
         (src_type == OrtMemoryInfoDeviceType_CPU &&
          dst_type == OrtMemoryInfoDeviceType_GPU);
}

OrtStatus *HipDataTransferImpl::CopyTensorsImpl(OrtDataTransferImpl *this_ptr,
                                                const OrtValue **src_tensors,
                                                OrtValue **dst_tensors,
                                                OrtSyncStream **streams,
                                                size_t num_tensors) noexcept {
  auto &impl = *static_cast<HipDataTransferImpl *>(this_ptr);
  bool need_stream_sync = false;

  for (size_t idx = 0; idx < num_tensors; ++idx) {
    const OrtValue *src_tensor = src_tensors[idx];
    OrtValue *dst_tensor = dst_tensors[idx];
    OrtSyncStream *stream = streams ? streams[idx] : nullptr;

    const OrtMemoryDevice *src_device =
        impl.ep_api.Value_GetMemoryDevice(src_tensor);
    const OrtMemoryDevice *dst_device =
        impl.ep_api.Value_GetMemoryDevice(dst_tensor);

    size_t bytes = 0;
    if (auto *st = impl.ort_api.GetTensorSizeInBytes(src_tensor, &bytes)) {
      return st;
    }

    const void *src_data = nullptr;
    void *dst_data = nullptr;
    if (auto *st = impl.ort_api.GetTensorData(src_tensor, &src_data)) {
      return st;
    }
    if (auto *st = impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data)) {
      return st;
    }

    OrtMemoryInfoDeviceType src_type =
        impl.ep_api.MemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_type =
        impl.ep_api.MemoryDevice_GetDeviceType(dst_device);
    OrtDeviceMemoryType src_mem_type =
        impl.ep_api.MemoryDevice_GetMemoryType(src_device);
    OrtDeviceMemoryType dst_mem_type =
        impl.ep_api.MemoryDevice_GetMemoryType(dst_device);

    const bool src_is_gpu_default = src_type == OrtMemoryInfoDeviceType_GPU &&
                                    src_mem_type == OrtDeviceMemoryType_DEFAULT;
    const bool dst_is_gpu_default = dst_type == OrtMemoryInfoDeviceType_GPU &&
                                    dst_mem_type == OrtDeviceMemoryType_DEFAULT;

    hipStream_t hip_stream = nullptr;
    if (stream) {
      hip_stream =
          static_cast<hipStream_t>(impl.ort_api.SyncStream_GetHandle(stream));
    }

    hipError_t err = hipSuccess;

    if (dst_is_gpu_default) {
      if (src_is_gpu_default) {
        if (dst_data != src_data) {
          if (hip_stream) {
            err = hipMemcpyAsync(dst_data, src_data, bytes,
                                 hipMemcpyDeviceToDevice, hip_stream);
          } else {
            err = hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToDevice);
            // hipMemcpy D2D is not host-synchronous; force a sync below.
            need_stream_sync = true;
          }
        }
      } else {
        // CPU (pinned or pageable) -> GPU
        if (hip_stream) {
          err = hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice,
                               hip_stream);
        } else {
          err = hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice);
          if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
            // Pageable host -> device may still have a pending DMA after
            // return.
            need_stream_sync = true;
          }
        }
      }
    } else if (src_is_gpu_default) {
      // GPU -> CPU (always blocking)
      if (hip_stream) {
        err = hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost,
                             hip_stream);
      } else {
        err = hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost);
      }
    } else {
      // CPU/host-accessible <-> CPU/host-accessible: plain memcpy.
      //
      // The hipStreamSynchronize below is intentional and serializes the
      // stream, but is necessary, not a performance bug:
      //
      //   * src_mem_type == HOST_ACCESSIBLE means the source buffer is our
      //     hipHostMalloc(Mapped|Coherent) memory, which the GPU may have
      //     written to via an earlier hipMemcpyAsync H2D / kernel that was
      //     queued on `hip_stream`. Since that write is async, the host
      //     memcpy here would otherwise race the in-flight GPU write and
      //     read stale data — sync forces all prior `hip_stream` work to
      //     drain before we touch the buffer from CPU.
      //
      //   * The other branches (D2D, H2D, D2H above) don't need this
      //     because they all submit further work onto `hip_stream`, so
      //     ORT's own stream-ordering contract handles the dependency.
      //
      //   * We only sync when src is HOST_ACCESSIBLE. A pure CPU->CPU copy
      //     (src_mem_type == DEFAULT cpu, no stream involvement) skips
      //     the sync to avoid a spurious GPU pipeline stall.
      //
      // If a future ORT API gives us a per-OrtValue "ready event" we can
      // replace this with a hipStreamWaitEvent + drop the full-stream
      // sync; until then the correctness/perf trade-off favors sync.
      if (dst_data != src_data) {
        if (hip_stream && src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
          err = hipStreamSynchronize(hip_stream);
          if (err != hipSuccess) {
            return MakeHipStatus(impl.ort_api, err, "hipStreamSynchronize");
          }
        }
        std::memcpy(dst_data, src_data, bytes);
      }
    }

    if (err != hipSuccess) {
      return MakeHipStatus(impl.ort_api, err, "hipMemcpy*");
    }
  }

  if (need_stream_sync) {
    hipError_t err = hipStreamSynchronize(nullptr);
    if (err != hipSuccess) {
      return MakeHipStatus(impl.ort_api, err, "hipStreamSynchronize(nullptr)");
    }
  }

  return nullptr;
}

void HipDataTransferImpl::ReleaseImpl(
    OrtDataTransferImpl * /*this_ptr*/) noexcept {
  // The factory owns the single shared instance; nothing to release here.
}

} // namespace morphizen
