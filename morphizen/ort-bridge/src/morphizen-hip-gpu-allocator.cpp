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

#include <cstring>
#include <string>

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

constexpr size_t kPageSize = 4096;

// Capacity for a large request that is known to be growing. The 1/64 headroom
// is the reuse span: a Gemma-4 KV tensor is 64 MiB at a 16 K context and grows
// 4096 bytes per token, so one capacity covers the next 256 tokens.
//
// Continuing the size-class table's geometry, whose steps go 1/2, 1/4, 1/16,
// 1/32 as buffers grow, because the waste is proportional to the size.
constexpr size_t LargeCapacity(size_t size) noexcept {
  const size_t want = size + size / 64;
  return (want + kPageSize - 1) / kPageSize * kPageSize;
}

// Checked at compile time because no unit-test target covers this file and
// returning less than the request would overflow the buffer.
constexpr bool LargeCapacityHolds() {
  for (size_t n = 16ull << 20; n <= 16ull << 30; n += n / 8) {
    const size_t c = LargeCapacity(n);
    if (c < n + n / 64 || c > n + n / 64 + kPageSize) {
      return false;
    }
  }
  return true;
}
static_assert(LargeCapacityHolds(),
              "LargeCapacity must cover the request with its growth headroom "
              "and waste at most one page beyond it");
static_assert(LargeCapacity(64ull << 20) - (64ull << 20) >= 128 * 4096,
              "one capacity must span more than a 128-token generation at a "
              "16 K context");

int OctaveOf(size_t size) noexcept {
  int octave = 0;
  while (size > 1) {
    size >>= 1;
    ++octave;
  }
  return octave;
}

// Hand a batch of buffers back to the driver with pool_mutex_ released:
// hipHostFree unpins pages and is as heavyweight as the hipHostMalloc that
// pinned them, so holding the lock would serialize every other allocation
// behind it.
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

// =============== HipGpuAllocator ===============

HipGpuAllocator::HipGpuAllocator(const OrtMemoryInfo *memory_info,
                                 const OrtApi & /*api*/)
    : memory_info_{memory_info}, device_id_{TryGetDeviceId(memory_info)} {
  version = NegotiatedOrtApiVersion();
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
}

// A capacity the model has just grown past cannot serve this request or any
// later one, so its idle buffers are dead weight. Bounded to the immediate
// neighbourhood below `size` so an unrelated, much smaller tensor pooled in
// the same octave is left alone.
void HipGpuAllocator::DropOutgrown(size_t size, std::vector<void *> &out) {
  auto it = large_free_.lower_bound(size - size / 16);
  while (it != large_free_.end() && it->first < size) {
    for (void *ptr : it->second.free) {
      ptr_to_size_.erase(ptr);
      large_retained_bytes_ -= it->first;
      out.push_back(ptr);
    }
    it = large_free_.erase(it);
  }
}

// One buffer from the least-recently-used capacity, skipping `keep` because
// evicting there would free a buffer of exactly the size being made room for.
// Buffers nothing can ask for again are never touched, so they are always the
// oldest and go first.
bool HipGpuAllocator::EvictLruLarge(size_t keep, std::vector<void *> &out) {
  auto victim = large_free_.end();
  for (auto it = large_free_.begin(); it != large_free_.end(); ++it) {
    if (it->first == keep) {
      continue;
    }
    if (victim == large_free_.end() ||
        it->second.last_used < victim->second.last_used) {
      victim = it;
    }
  }
  if (victim == large_free_.end()) {
    return false;
  }
  void *ptr = victim->second.free.back();
  victim->second.free.pop_back();
  ptr_to_size_.erase(ptr);
  large_retained_bytes_ -= victim->first;
  out.push_back(ptr);
  if (victim->second.free.empty()) {
    large_free_.erase(victim);
  }
  return true;
}

void HipGpuAllocator::DrainLarge(std::vector<void *> &out) {
  for (auto &entry : large_free_) {
    for (void *ptr : entry.second.free) {
      ptr_to_size_.erase(ptr);
      out.push_back(ptr);
    }
  }
  large_free_.clear();
  large_retained_bytes_ = 0;
}

void *ORT_API_CALL HipGpuAllocator::AllocImpl(OrtAllocator *this_,
                                              size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);

  const int cls = SizeClassIndex(size);
  // Bytes to allocate on a miss: the class capacity for a pooled request, and
  // for a large one either the exact size or a grown capacity depending on
  // whether its octave has been seen to move.
  size_t alloc_size = (cls >= 0) ? kSizeClasses[cls] : size;
  std::vector<void *> stale;

  // Fast path: reuse a buffer with no driver call. A size class holds buffers
  // all at the class capacity, so any of them fits. A large request takes the
  // smallest retained capacity that covers it, close enough that it cannot
  // consume a buffer a much larger request will need.
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    if (cls >= 0) {
      auto &fl = self->free_lists_[cls];
      if (!fl.empty()) {
        void *ptr = fl.back();
        fl.pop_back();
        return ptr;
      }
    } else {
      ++self->large_ops_;
      auto &octave = self->octaves_[OctaveOf(size)];
      if (octave.high_water != 0 && size > octave.high_water) {
        octave.growing = true;
      }
      if (size > octave.high_water) {
        octave.high_water = size;
      }

      auto it = self->large_free_.lower_bound(size);
      if (it != self->large_free_.end() && it->first <= size + size / 32) {
        void *ptr = it->second.free.back();
        it->second.free.pop_back();
        it->second.last_used = self->large_ops_;
        self->large_retained_bytes_ -= it->first;
        self->large_live_bytes_ += it->first;
        if (self->large_live_bytes_ > self->large_peak_live_bytes_) {
          self->large_peak_live_bytes_ = self->large_live_bytes_;
        }
        if (it->second.free.empty()) {
          self->large_free_.erase(it);
        }
        return ptr;
      }
      if (octave.growing) {
        alloc_size = LargeCapacity(size);
        self->DropOutgrown(size, stale);
      }
    }
  }
  ReleaseToDriver(stale, self->device_id_);

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
    // The large pool may be sitting on pinned pages this allocation could use,
    // and failing while holding reclaimable memory would be worse than not
    // pooling at all.
    std::vector<void *> drained;
    {
      std::lock_guard<std::mutex> lk(self->pool_mutex_);
      self->DrainLarge(drained);
    }
    if (!drained.empty()) {
      ReleaseToDriver(drained, self->device_id_);
      err = hipHostMalloc(&ptr, alloc_size,
                          hipHostMallocMapped | hipHostMallocCoherent);
    }
  }
  if (err != hipSuccess) {
    LOG(ERROR) << "[MorphiZen HIP] hipHostMalloc(Mapped|Coherent) failed for "
               << alloc_size << " bytes (device " << self->device_id_
               << "): " << hipGetErrorString(err);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    self->ptr_to_size_[ptr] = alloc_size;
    if (cls < 0) {
      self->large_live_bytes_ += alloc_size;
      if (self->large_live_bytes_ > self->large_peak_live_bytes_) {
        self->large_peak_live_bytes_ = self->large_live_bytes_;
      }
    }
  }
  return ptr;
}

void ORT_API_CALL HipGpuAllocator::FreeImpl(OrtAllocator *this_, void *p) {
  if (p == nullptr) {
    return;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);
  // p itself when it cannot be retained, plus whatever eviction dropped to
  // make room for it.
  std::vector<void *> to_release;
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    auto it = self->ptr_to_size_.find(p);
    if (it == self->ptr_to_size_.end()) {
      // Defensive: a pointer we never handed out. Release rather than leak.
      to_release.push_back(p);
    } else if (const int cls = SizeClassIndex(it->second); cls >= 0) {
      // Pooled small/medium buffer (<= 16 MB): return to its free list for
      // reuse; do NOT release to the driver here. Stays tracked in
      // ptr_to_size_ so the destructor can release it. Safe to reuse without
      // a stream sync: see the pool comment in the header.
      self->free_lists_[cls].push_back(p);
      return;
    } else {
      const size_t capacity = it->second;
      ++self->large_ops_;
      self->large_live_bytes_ -= (self->large_live_bytes_ < capacity)
                                     ? self->large_live_bytes_
                                     : capacity;
      // Retaining this would put the pool over a ceiling that tracks the most
      // this model has had checked out at once, so evict the capacities that
      // have gone longest without being asked for before giving up on it.
      bool room = self->large_retained_bytes_ + capacity <=
                  self->large_peak_live_bytes_;
      while (!room && self->EvictLruLarge(capacity, to_release)) {
        room = self->large_retained_bytes_ + capacity <=
               self->large_peak_live_bytes_;
      }
      if (room) {
        auto &bucket = self->large_free_[capacity];
        bucket.free.push_back(p);
        bucket.last_used = self->large_ops_;
        self->large_retained_bytes_ += capacity;
        return;
      }
      self->ptr_to_size_.erase(it);
      to_release.push_back(p);
    }
  }
  ReleaseToDriver(to_release, self->device_id_);
}

HipGpuAllocator::~HipGpuAllocator() {
  // Release every pinned buffer the pool ever allocated. By teardown ORT has
  // freed all outstanding tensors (they are back on free_lists_), but we walk
  // ptr_to_size_ rather than free_lists_ so any still-checked-out buffer is
  // also released instead of leaked.
  ScopedDevice _(device_id_);
  std::lock_guard<std::mutex> lk(pool_mutex_);
  for (const auto &kv : ptr_to_size_) {
    hipError_t err = hipHostFree(kv.first);
    if (err != hipSuccess) {
      LOG(WARNING) << "[MorphiZen HIP] hipHostFree failed (device "
                   << device_id_ << "): " << hipGetErrorString(err);
    }
  }
  for (auto &fl : free_lists_) {
    fl.clear();
  }
  large_free_.clear();
  large_retained_bytes_ = 0;
  large_live_bytes_ = 0;
  ptr_to_size_.clear();
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
