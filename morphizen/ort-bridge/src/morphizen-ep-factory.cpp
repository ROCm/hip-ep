/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/morphizen.hpp"

#include "./morphizen-ep-factory.hpp"
#include "./morphizen-ep.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include "morphizen-utils/morphizen_plugin.hpp"
#include "morphizen/onnxruntime_morphizen_ep.hpp"
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
#include "./morphizen-hip-gpu-allocator.hpp"
#endif

DEF_ENV_PARAM(MORPHIZEN_DEBUG_MORPHIZEN_EP_FACTORY, "0")
//  Why MORPHIZEN_EP_ENABLE_CPU_DEVICE is Required
//
//  The Morphizen Execution Provider runs on **AMD GPU hardware** in
//  production via the hipMalloc-based OrtAllocator + hipMemcpy DataTransfer
//  registered by this factory. Only debug / unit-test builds that need the
//  EP to load on machines without a usable AMD GPU should opt into the
//  legacy "CPU device" mode by setting MORPHIZEN_EP_ENABLE_CPU_DEVICE=1.
//
// **Production Mode (default, ENV=0):**
// - Morphizen EP only accepts **AMD GPU** devices (vendor_id 0x1002 ==
// OrtDevice::VendorIds::AMD)
// - Rejects CPU/NPU devices
// - Returns zero EP devices if no AMD GPU is present
//
// **Debug / Test Mode (ENV=1):**
// - Morphizen EP accepts **CPU devices** instead
// - Allows MLIR pass execution and session creation on machines without a
//   usable AMD GPU; no GPU allocator / DataTransfer is registered, so any
//   model I/O stays in host RAM and the EP runtime does the copy itself
//   (the pre-2026-04 behavior)
DEF_ENV_PARAM(MORPHIZEN_EP_ENABLE_CPU_DEVICE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MORPHIZEN_EP_FACTORY) >= n)
namespace morphizen {

MorphiZenEpFactory::MorphiZenEpFactory(const char *ep_name, ApiPtrs apis,
                                       const OrtLogger &default_logger)
    : OrtEpFactory{}, // Ensure optional functions are default initialized to
                      // nullptr
      ApiPtrs(apis), default_logger_{default_logger}, ep_name_{ep_name},
      ep_metadata_{nullptr, apis.ort_api.ReleaseKeyValuePairs},
      ep_options_{nullptr, apis.ort_api.ReleaseKeyValuePairs} {
  ort_version_supported =
      ORT_API_VERSION; // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  ValidateCompiledModelCompatibilityInfo =
      ValidateCompiledModelCompatibilityInfoImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
}
const char *ORT_API_CALL
MorphiZenEpFactory::GetNameImpl(const OrtEpFactory *this_ptr) noexcept {
  const auto *factory = static_cast<const MorphiZenEpFactory *>(this_ptr);
  return factory->ep_name_.c_str();
}

const char *ORT_API_CALL
MorphiZenEpFactory::GetVendorImpl(const OrtEpFactory *this_ptr) noexcept {
  const auto *factory = static_cast<const MorphiZenEpFactory *>(this_ptr);
  return factory->vendor_.c_str();
}
uint32_t ORT_API_CALL
MorphiZenEpFactory::GetVendorIdImpl(const OrtEpFactory *this_ptr) noexcept {
  const auto *factory = static_cast<const MorphiZenEpFactory *>(this_ptr);
  return factory->vendor_id_;
}
const char *ORT_API_CALL
MorphiZenEpFactory::GetVersionImpl(const OrtEpFactory *this_ptr) noexcept {
  const auto *factory = static_cast<const MorphiZenEpFactory *>(this_ptr);
  return factory->ep_version_.c_str();
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory *this_ptr, const OrtHardwareDevice *const *devices,
    size_t num_devices, OrtEpDevice **ep_devices, size_t max_ep_devices,
    size_t *p_num_ep_devices) noexcept {
  size_t &num_ep_devices = *p_num_ep_devices;
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);

  // ORT plugin EP V2 requires every OrtEpDevice registered by a single
  // factory to share the same OrtDeviceMemoryInfo. We therefore pick a single
  // device "mode" up-front: production = AMD GPU, debug = CPU. See the
  // MORPHIZEN_EP_ENABLE_CPU_DEVICE comment near the top of this file.
  const bool cpu_debug_mode = ENV_PARAM(MORPHIZEN_EP_ENABLE_CPU_DEVICE) != 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice *hardware_device = devices[i];
    const std::uint32_t vendor_id =
        factory->ort_api.HardwareDevice_VendorId(hardware_device);
    const OrtHardwareDeviceType device_type =
        factory->ort_api.HardwareDevice_Type(hardware_device);

    if (cpu_debug_mode) {
      // Debug mode: pretend to support CPU EP so unit tests can load the EP
      // on machines without a usable AMD GPU.
      if (device_type != OrtHardwareDeviceType_CPU) {
        continue;
      }
    } else {
      // Production: only accept AMD GPU devices (PCI vendor 0x1002 ==
      // OrtDevice::VendorIds::AMD).
      if (device_type != OrtHardwareDeviceType_GPU ||
          vendor_id != factory->vendor_id_) {
        continue;
      }
    }
    // these can be returned as nullptr if you have nothing to add.
    OrtKeyValuePairs *ep_metadata = nullptr;
    OrtKeyValuePairs *ep_options = nullptr;
    factory->ort_api.CreateKeyValuePairs(&ep_metadata);
    factory->ort_api.CreateKeyValuePairs(&ep_options);
    factory->ep_metadata_ =
        std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs *)>(
            ep_metadata, factory->ort_api.ReleaseKeyValuePairs);
    factory->ep_options_ =
        std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs *)>(
            ep_options, factory->ort_api.ReleaseKeyValuePairs);
    if (num_ep_devices == max_ep_devices) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT, "Not enough space to return EP devices.");
    }
    // OrtEpDevice copies ep_metadata and ep_options.
    OrtEpDevice *registered_ep_device = nullptr;
    auto *status = factory->ort_api.GetEpApi()->CreateEpDevice(
        factory, hardware_device, ep_metadata, ep_options,
        &registered_ep_device);
    if (status != nullptr) {
      return status;
    }
    ep_devices[num_ep_devices++] = registered_ep_device;

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
    // For AMD GPU devices, also register the OrtMemoryInfo entries that
    // describe where our allocator places tensors. ORT will later look these
    // up by OrtMemoryInfo and call CreateAllocator/CreateDataTransfer.
    if (!cpu_debug_mode) {
      const OrtEpApi *ep_api_ptr = factory->ort_api.GetEpApi();

      // Lazily create the two OrtMemoryInfo instances on first use; reuse
      // them across subsequent factory invocations.
      if (!factory->gpu_memory_info_) {
        OrtMemoryInfo *raw = nullptr;
        auto *st = factory->ort_api.CreateMemoryInfo_V2(
            "MorphiZen", OrtMemoryInfoDeviceType_GPU,
            /*vendor*/ factory->vendor_id_,
            /*device_id*/ 0, OrtDeviceMemoryType_DEFAULT,
            /*alignment*/ 0, OrtAllocatorType::OrtDeviceAllocator, &raw);
        if (st != nullptr) {
          return st;
        }
        factory->gpu_memory_info_ = MorphiZenEpFactory::MemoryInfoPtr(
            raw, factory->ort_api.ReleaseMemoryInfo);
      }
      if (!factory->gpu_host_accessible_memory_info_) {
        OrtMemoryInfo *raw = nullptr;
        auto *st = factory->ort_api.CreateMemoryInfo_V2(
            "MorphiZen host accessible", OrtMemoryInfoDeviceType_GPU,
            /*vendor*/ factory->vendor_id_,
            /*device_id*/ 0, OrtDeviceMemoryType_HOST_ACCESSIBLE,
            /*alignment*/ 0, OrtAllocatorType::OrtDeviceAllocator, &raw);
        if (st != nullptr) {
          return st;
        }
        factory->gpu_host_accessible_memory_info_ =
            MorphiZenEpFactory::MemoryInfoPtr(
                raw, factory->ort_api.ReleaseMemoryInfo);
      }

      if (auto *st = ep_api_ptr->EpDevice_AddAllocatorInfo(
              registered_ep_device, factory->gpu_memory_info_.get())) {
        return st;
      }
      if (auto *st = ep_api_ptr->EpDevice_AddAllocatorInfo(
              registered_ep_device,
              factory->gpu_host_accessible_memory_info_.get())) {
        return st;
      }
    }
#endif
  }
  return nullptr;
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::CreateEpImpl(
    OrtEpFactory *this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice *const * /*devices*/,
    _In_reads_(num_devices) const OrtKeyValuePairs *const *ep_metadata,
    _In_ size_t num_devices, _In_ const OrtSessionOptions *session_options,
    _In_ const OrtLogger *logger, _Out_ OrtEp **ep) noexcept {
  MY_LOG(1) << "CreateEpImpl: num_devices=" << num_devices;
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);
  *ep = nullptr;

  // MorphiZen does not use the selected-device array: it compiles for and runs
  // on the default / currently-selected HIP device. Any num_devices is
  // accepted, including a parent EP (e.g. the AMD GPU umbrella) that creates
  // the EP without forwarding the selection.

  // Create the execution provider
  factory->throw_if_error(factory->ort_api.Logger_LogMessage(
      logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Creating MorphiZen EP",
      ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  auto morphizen_ep = std::make_unique<MorphiZenEP>(
      *factory, factory->ep_name_, ep_metadata, *session_options, *logger);

  *ep = morphizen_ep.release();
  return nullptr;
}
void ORT_API_CALL MorphiZenEpFactory::ReleaseEpImpl(OrtEpFactory * /*this_ptr*/,
                                                    OrtEp *ep) noexcept {
  MorphiZenEP *morphizen_ep = static_cast<MorphiZenEP *>(ep);
  delete morphizen_ep;
}

OrtStatus *ORT_API_CALL
MorphiZenEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory *this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice *const *devices,
    _In_ size_t num_devices, _In_ const char *compatibility_info,
    _Out_ OrtCompiledModelCompatibility *model_compatibility) noexcept {
  // Note: this_ptr (MorphiZenEpFactory*) is not currently used for validation
  // Empty eps pointer since validation is based on compatibility_info and
  // devices eps may be used in future for EP-specific validation
  (void)this_ptr; // Suppress unused parameter warning
  int compatibility_result = 0;
  int status = validate_compiled_model_compatibility_info(
      nullptr, // eps - not available in factory context
      compatibility_info, reinterpret_cast<const void *const *>(devices),
      num_devices, &compatibility_result);

  if (status != 0) {
    LOG(WARNING)
        << "MorphiZenEP validate compiled model compatibility info failed: "
        << status << ", compatibility_info: " << compatibility_info;
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  *model_compatibility =
      static_cast<OrtCompiledModelCompatibility>(compatibility_result);
  return nullptr;
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::CreateAllocatorImpl(
    OrtEpFactory *this_ptr,
#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
    const OrtMemoryInfo *memory_info,
#else
    const OrtMemoryInfo * /*memory_info*/,
#endif
    const OrtKeyValuePairs * /*allocator_options*/,
    OrtAllocator **allocator) noexcept {
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
  if (memory_info != nullptr) {
    *allocator = new HipGpuAllocator(memory_info, factory->ort_api);
    return nullptr;
  }
#endif

  *allocator = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateAllocator should not be called as we did "
                            "not add OrtMemoryInfo to our OrtEpDevice.");
}

void ORT_API_CALL MorphiZenEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory * /*this*/, OrtAllocator *allocator) noexcept {
#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
  if (allocator != nullptr) {
    delete static_cast<HipGpuAllocator *>(allocator);
  }
#else
  (void)allocator;
  // Should never be called when CreateAllocator returns an error.
  LOG(FATAL) << "TODO";
#endif
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::CreateDataTransferImpl(
    OrtEpFactory *this_ptr, OrtDataTransferImpl **data_transfer) noexcept {
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);
#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
  if (!factory->data_transfer_impl_) {
    factory->data_transfer_impl_ =
        std::make_unique<HipDataTransferImpl>(factory->ort_api);
  }
  *data_transfer = factory->data_transfer_impl_.get();
  return nullptr;
#else
  (void)factory;
  *data_transfer = nullptr; // not implemented when GPU allocator is disabled
  return nullptr;
#endif
}

bool ORT_API_CALL MorphiZenEpFactory::IsStreamAwareImpl(
    const OrtEpFactory * /*this_ptr*/) noexcept {
  return false;
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory *this_ptr, const OrtMemoryDevice * /*memory_device*/,
    const OrtKeyValuePairs * /*stream_options*/,
    OrtSyncStreamImpl **stream) noexcept {
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);

  *stream = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateSyncStreamForDevice should not be called as "
                            "IsStreamAware returned false.");
}

} // namespace morphizen
