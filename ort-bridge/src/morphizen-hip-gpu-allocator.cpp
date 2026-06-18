/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HIP-runtime backed implementation of OrtAllocator and OrtDataTransferImpl
// for the MorphiZen EP. Mirrors onnxruntime's NvTensorRtRtx EP factory
// (nv_provider_factory.cc, NvTrtRtxOrtAllocator / NvTrtRtxDataTransferImpl)
// translated to the HIP runtime API.

#include "./morphizen-hip-gpu-allocator.hpp"

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

#include <atomic>
#include <chrono>
#include <cstdio>

namespace morphizen {

namespace {

// TEMPORARY verification instrumentation: measure how much wall time the EP
// allocator spends in hipHostMalloc / hipHostFree across the whole process,
// and how many times each is called. Prints a one-line summary at DLL unload
// (static destructor). Remove once the alloc-cost hypothesis is confirmed.
struct AllocProfile {
  std::atomic<uint64_t> alloc_reqs{0};  // every AllocImpl call
  std::atomic<uint64_t> pool_hits{0};   // served from the free-list (no driver)
  std::atomic<uint64_t> alloc_calls{0}; // actual hipHostMalloc (cold miss)
  std::atomic<uint64_t> alloc_ns{0};
  std::atomic<uint64_t> alloc_bytes{0};
  std::atomic<uint64_t> free_calls{0}; // actual hipHostFree (destructor only)
  std::atomic<uint64_t> free_ns{0};
  ~AllocProfile() {
    uint64_t ar = alloc_reqs.load(), ph = pool_hits.load();
    uint64_t ac = alloc_calls.load(), an = alloc_ns.load();
    uint64_t fc = free_calls.load(), fn = free_ns.load();
    std::fprintf(
        stderr,
        "[ALLOC PROFILE] AllocImpl reqs=%llu pool_hits=%llu | hipHostMalloc "
        "calls=%llu total=%.3fms avg=%.4fms bytes=%llu | hipHostFree "
        "calls=%llu total=%.3fms avg=%.4fms\n",
        (unsigned long long)ar, (unsigned long long)ph,
        (unsigned long long)ac, an / 1e6, ac ? (an / 1e6) / ac : 0.0,
        (unsigned long long)alloc_bytes.load(), (unsigned long long)fc,
        fn / 1e6, fc ? (fn / 1e6) / fc : 0.0);
    std::fflush(stderr);
  }
};
AllocProfile g_alloc_profile;

// Convert a non-success hipError_t into an OrtStatus*. The caller owns the
// returned status (must release via ort_api.ReleaseStatus).
inline OrtStatus* MakeHipStatus(const OrtApi& api, hipError_t err,
                                const char* what) {
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
int TryGetDeviceId(const OrtMemoryInfo* memory_info) noexcept {
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
  } catch (const Ort::Exception&) {
    return -1;
  }
}

} // namespace

// =============== HipGpuAllocator ===============

HipGpuAllocator::HipGpuAllocator(const OrtMemoryInfo* memory_info,
                                 const OrtApi& /*api*/)
    : memory_info_{memory_info}, device_id_{TryGetDeviceId(memory_info)} {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = ReserveImpl;
  GetStats = nullptr;
}

void* ORT_API_CALL HipGpuAllocator::AllocImpl(OrtAllocator* this_,
                                              size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto* self = static_cast<HipGpuAllocator*>(this_);
  g_alloc_profile.alloc_reqs.fetch_add(1, std::memory_order_relaxed);

  // Fast path: reuse a same-size buffer from the free list (no driver call).
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    auto it = self->free_lists_.find(size);
    if (it != self->free_lists_.end() && !it->second.empty()) {
      void* ptr = it->second.back();
      it->second.pop_back();
      g_alloc_profile.pool_hits.fetch_add(1, std::memory_order_relaxed);
      return ptr;
    }
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
  void* ptr = nullptr;
  auto _t0 = std::chrono::high_resolution_clock::now();
  hipError_t err =
      hipHostMalloc(&ptr, size, hipHostMallocMapped | hipHostMallocCoherent);
  auto _t1 = std::chrono::high_resolution_clock::now();
  g_alloc_profile.alloc_calls.fetch_add(1, std::memory_order_relaxed);
  g_alloc_profile.alloc_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count(),
      std::memory_order_relaxed);
  g_alloc_profile.alloc_bytes.fetch_add(size, std::memory_order_relaxed);
  if (err != hipSuccess) {
    LOG(ERROR) << "[MorphiZen HIP] hipHostMalloc(Mapped|Coherent) failed for "
               << size << " bytes (device " << self->device_id_
               << "): " << hipGetErrorString(err);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(self->pool_mutex_);
    self->ptr_to_size_[ptr] = size;
  }
  return ptr;
}

void ORT_API_CALL HipGpuAllocator::FreeImpl(OrtAllocator* this_, void* p) {
  if (p == nullptr) {
    return;
  }
  auto* self = static_cast<HipGpuAllocator*>(this_);
  std::lock_guard<std::mutex> lk(self->pool_mutex_);
  auto it = self->ptr_to_size_.find(p);
  if (it != self->ptr_to_size_.end()) {
    // Return to the free list for reuse; do NOT release to the driver here.
    // Safe to reuse without a stream sync: see the pool comment in the header.
    self->free_lists_[it->second].push_back(p);
    return;
  }
  // Defensive: a pointer we never handed out (should not happen). Release it
  // directly so we don't leak, rather than pooling an unknown-size buffer.
  ScopedDevice _(self->device_id_);
  (void)hipHostFree(p);
}

HipGpuAllocator::~HipGpuAllocator() {
  // Release every pinned buffer the pool ever allocated. By teardown ORT has
  // freed all outstanding tensors (they are back on free_lists_), but we walk
  // ptr_to_size_ rather than free_lists_ so any still-checked-out buffer is
  // also released instead of leaked.
  ScopedDevice _(device_id_);
  std::lock_guard<std::mutex> lk(pool_mutex_);
  for (const auto& kv : ptr_to_size_) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    hipError_t err = hipHostFree(kv.first);
    auto _t1 = std::chrono::high_resolution_clock::now();
    g_alloc_profile.free_calls.fetch_add(1, std::memory_order_relaxed);
    g_alloc_profile.free_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(_t1 - _t0).count(),
        std::memory_order_relaxed);
    if (err != hipSuccess) {
      LOG(WARNING) << "[MorphiZen HIP] hipHostFree failed (device " << device_id_
                   << "): " << hipGetErrorString(err);
    }
  }
  free_lists_.clear();
  ptr_to_size_.clear();
}

const OrtMemoryInfo* ORT_API_CALL
HipGpuAllocator::InfoImpl(const OrtAllocator* this_) {
  return static_cast<const HipGpuAllocator*>(this_)->memory_info_;
}

void* ORT_API_CALL HipGpuAllocator::ReserveImpl(OrtAllocator* this_,
                                                size_t size) {
  // No special reservation strategy; behave like Alloc.
  return AllocImpl(this_, size);
}

// =============== HipDataTransferImpl ===============

HipDataTransferImpl::HipDataTransferImpl(const OrtApi& ort_api_in)
    : ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
  ort_version_supported = ORT_API_VERSION;
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

bool HipDataTransferImpl::CanCopyImpl(
    const OrtDataTransferImpl* this_ptr,
    const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const HipDataTransferImpl*>(this_ptr);

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

OrtStatus* HipDataTransferImpl::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                const OrtValue** src_tensors,
                                                OrtValue** dst_tensors,
                                                OrtSyncStream** streams,
                                                size_t num_tensors) noexcept {
  auto& impl = *static_cast<HipDataTransferImpl*>(this_ptr);
  bool need_stream_sync = false;

  for (size_t idx = 0; idx < num_tensors; ++idx) {
    const OrtValue* src_tensor = src_tensors[idx];
    OrtValue* dst_tensor = dst_tensors[idx];
    OrtSyncStream* stream = streams ? streams[idx] : nullptr;

    const OrtMemoryDevice* src_device =
        impl.ep_api.Value_GetMemoryDevice(src_tensor);
    const OrtMemoryDevice* dst_device =
        impl.ep_api.Value_GetMemoryDevice(dst_tensor);

    size_t bytes = 0;
    if (auto* st = impl.ort_api.GetTensorSizeInBytes(src_tensor, &bytes)) {
      return st;
    }

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    if (auto* st = impl.ort_api.GetTensorData(src_tensor, &src_data)) {
      return st;
    }
    if (auto* st = impl.ort_api.GetTensorMutableData(dst_tensor, &dst_data)) {
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
    OrtDataTransferImpl* /*this_ptr*/) noexcept {
  // The factory owns the single shared instance; nothing to release here.
}

} // namespace morphizen
