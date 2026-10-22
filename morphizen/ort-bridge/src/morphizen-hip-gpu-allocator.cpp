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

#include <cstdint>
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
// -1 when size exceeds the largest class (a "large" buffer).
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

// See the large-request comment on HipGpuAllocator.
#ifdef _WIN32
constexpr uint64_t kPoolReleaseThreshold = 0;
#else
constexpr uint64_t kPoolReleaseThreshold = UINT64_MAX;
#endif

// pool == nullptr allocates from the driver.
hipError_t Allocate(void **ptr, size_t size, bool device_only,
                    hipMemPool_t pool, hipStream_t pool_stream) {
  if (pool != nullptr) {
    return hipMallocFromPoolAsync(ptr, size, pool, pool_stream);
  }
  return device_only
             ? hipMalloc(ptr, size)
             : hipHostMalloc(ptr, size,
                             hipHostMallocMapped | hipHostMallocCoherent);
}

// The sync hands a pool block back to the pool's address range, where a
// request of another size can take it. A block left in the pool's free list
// serves only requests up to 1/8 smaller, and once those blocks outweigh the
// ones in use, HIP unmaps every free chunk regardless of the release
// threshold.
hipError_t Release(void *ptr, bool device_only, hipStream_t pool_stream) {
  if (pool_stream != nullptr) {
    (void)hipFreeAsync(ptr, pool_stream);
    return hipStreamSynchronize(pool_stream);
  }
  return device_only ? hipFree(ptr) : hipHostFree(ptr);
}

// Returns false with nothing created on failure, leaving large requests to
// hipMalloc.
bool CreateDefaultPool(int device_id, hipMemPool_t *pool, hipStream_t *stream) {
  int device = device_id;
  if (device < 0 && hipGetDevice(&device) != hipSuccess) {
    return false;
  }
  int supported = 0;
  if (hipDeviceGetAttribute(&supported, hipDeviceAttributeMemoryPoolsSupported,
                            device) != hipSuccess ||
      supported == 0) {
    LOG(WARNING) << "[MorphiZen HIP] device " << device
                 << " has no stream-ordered memory pools; using hipMalloc";
    return false;
  }
  hipMemPoolProps props = {};
  props.allocType = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  if (hipMemPoolCreate(pool, &props) != hipSuccess) {
    return false;
  }
  uint64_t threshold = kPoolReleaseThreshold;
  if (hipMemPoolSetAttribute(*pool, hipMemPoolAttrReleaseThreshold,
                             &threshold) != hipSuccess ||
      hipStreamCreateWithFlags(stream, hipStreamNonBlocking) != hipSuccess) {
    (void)hipMemPoolDestroy(*pool);
    *pool = nullptr;
    return false;
  }
  return true;
}

bool UseDeviceOnly(const OrtMemoryInfo *memory_info) noexcept {
  if (memory_info == nullptr) {
    return false;
  }
  try {
    return Ort::ConstMemoryInfo(memory_info).GetDeviceMemoryType() ==
           OrtDeviceMemoryType_DEFAULT;
  } catch (const Ort::Exception &) {
    return false;
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
    : memory_info_{memory_info}, device_id_{TryGetDeviceId(memory_info)},
      device_only_{UseDeviceOnly(memory_info)} {
  version = NegotiatedOrtApiVersion();
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
  if (device_only_) {
    ScopedDevice _(device_id_);
    hipMemPool_t pool = nullptr;
    hipStream_t stream = nullptr;
    if (CreateDefaultPool(device_id_, &pool, &stream)) {
      pool_ = pool;
      pool_stream_ = stream;
    }
  }
}

void *ORT_API_CALL HipGpuAllocator::AllocImpl(OrtAllocator *this_,
                                              size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);

  // Fast path: reuse a pooled buffer with no driver call. A size class holds
  // buffers all at the class capacity, so any of them fits.
  const int cls = SizeClassIndex(size);
  if (cls >= 0) {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    auto &fl = self->free_lists_[cls];
    if (!fl.empty()) {
      void *ptr = fl.back();
      fl.pop_back();
      return ptr;
    }
  }
  const size_t alloc_size = (cls >= 0) ? kSizeClasses[cls] : size;
  hipMemPool_t pool = (cls < 0) ? self->pool_ : nullptr;

  ScopedDevice _(self->device_id_);
  void *ptr = nullptr;
  hipError_t err =
      Allocate(&ptr, alloc_size, self->device_only_, pool, self->pool_stream_);
  if (err != hipSuccess && self->pool_ != nullptr) {
    // Free chunks the pool keeps mapped may hold the memory this request
    // needs.
    (void)hipMemPoolTrimTo(self->pool_, 0);
    err = Allocate(&ptr, alloc_size, self->device_only_, pool,
                   self->pool_stream_);
  }
  if (err != hipSuccess) {
    LOG(ERROR) << "[MorphiZen HIP] "
               << (pool != nullptr      ? "hipMallocFromPoolAsync"
                   : self->device_only_ ? "hipMalloc"
                                        : "hipHostMalloc")
               << " failed for " << alloc_size << " bytes (device "
               << self->device_id_ << "): " << hipGetErrorString(err);
    return nullptr;
  }
  if (pool != nullptr) {
    // On Linux HIP creates the pool's address ranges on first use and starts
    // each with a release threshold of zero, whatever the pool's was; setting
    // it again reaches any range this request created before a free can unmap
    // it.
    uint64_t threshold = kPoolReleaseThreshold;
    (void)hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold,
                                 &threshold);
  }
  std::lock_guard<std::mutex> lk(self->pool_mutex_);
  self->ptr_to_size_[ptr] = alloc_size;
  return ptr;
}

void ORT_API_CALL HipGpuAllocator::FreeImpl(OrtAllocator *this_, void *p) {
  if (p == nullptr) {
    return;
  }
  auto *self = static_cast<HipGpuAllocator *>(this_);
  bool from_pool = false;
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    // A pointer we never handed out is released rather than leaked.
    if (auto it = self->ptr_to_size_.find(p); it != self->ptr_to_size_.end()) {
      if (const int cls = SizeClassIndex(it->second); cls >= 0) {
        // Pooled small/medium buffer (<= 16 MB): return to its free list for
        // reuse; do NOT release to the driver here. Stays tracked in
        // ptr_to_size_ so the destructor can release it. Safe to reuse
        // without a stream sync: see the pool comment in the header.
        self->free_lists_[cls].push_back(p);
        return;
      }
      from_pool = self->pool_ != nullptr;
      self->ptr_to_size_.erase(it);
    }
  }
  // With pool_mutex_ released: hipHostFree unpins pages and is as heavyweight
  // as the hipHostMalloc that pinned them, so holding the lock would
  // serialize every other allocation behind it.
  ScopedDevice _(self->device_id_);
  (void)Release(p, self->device_only_,
                from_pool ? self->pool_stream_ : nullptr);
}

HipGpuAllocator::~HipGpuAllocator() {
  // Release every buffer this allocator ever allocated. By teardown ORT has
  // freed all outstanding tensors (they are back on free_lists_), but we walk
  // ptr_to_size_ rather than free_lists_ so any still-checked-out buffer is
  // also released instead of leaked.
  ScopedDevice _(device_id_);
  std::lock_guard<std::mutex> lk(pool_mutex_);
  for (const auto &kv : ptr_to_size_) {
    const bool from_pool = pool_ != nullptr && SizeClassIndex(kv.second) < 0;
    hipError_t err =
        Release(kv.first, device_only_, from_pool ? pool_stream_ : nullptr);
    if (err != hipSuccess) {
      LOG(WARNING) << "[MorphiZen HIP] driver free failed (device "
                   << device_id_ << "): " << hipGetErrorString(err);
    }
  }
  if (pool_ != nullptr) {
    // Destroying a pool frees only the blocks returned to it, which the loop
    // above has done for every checked-out one.
    (void)hipMemPoolDestroy(pool_);
    (void)hipStreamDestroy(pool_stream_);
  }
  for (auto &fl : free_lists_) {
    fl.clear();
  }
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
