/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/morphizen.hpp"

#include "./device-selection.hpp"
#include "./morphizen-ep-factory.hpp"
#include "./morphizen-ep.hpp"
#include "./ort-api-version.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include "morphizen-utils/morphizen_plugin.hpp"
#include "morphizen/onnxruntime_morphizen_ep.hpp"
#include <cstring>
#include <glog/logging.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/util/json_util.h>
#include <string>

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
#include "./morphizen-hip-gpu-allocator.hpp"
#include <hip/hip_runtime.h>
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
  ort_version_supported = NegotiatedOrtApiVersion();
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
  GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
  GetCustomOpDomains = GetCustomOpDomainsImpl;

  // Populate custom_op_domains_ once here so its lifetime is tied to this
  // factory instance instead of the process.
  CollectCustomOpDomains(custom_op_domains_);

  morphizen::add_cleanup_function("protobuf shutdown", []() {
#ifdef _WIN32
    google::protobuf::ShutdownProtobufLibrary();
#endif
  });
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

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
namespace {

device_selection::PlatformId ReadPlatformId(const OrtApi &api,
                                            const OrtHardwareDevice *device) {
  if (api.HardwareDevice_Metadata == nullptr || api.GetKeyValue == nullptr ||
      device == nullptr) {
    return {};
  }
  const OrtKeyValuePairs *metadata = api.HardwareDevice_Metadata(device);
  if (metadata == nullptr) {
    return {};
  }
  const char *key = device_selection::PlatformIdMetadataKey();
  return device_selection::PlatformIdFromMetadata(
      key, api.GetKeyValue(metadata, key));
}

// Walks HIP ordinals until one reports the same canonical identity. Returns
// -1 and writes *error when enumeration fails or nothing matches. Does not
// change the calling thread's current device.
int FindHipOrdinal(const device_selection::PlatformId &identity,
                   std::string *error) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
    (void)hipGetLastError();
    *error = "HIP device enumeration failed while selecting a GPU.";
    return -1;
  }

  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
#ifdef _WIN32
    char luid_bytes[8] = {};
    unsigned int node_mask = 0;
    if (hipDeviceGetLuid(luid_bytes, &node_mask, ordinal) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    std::uint64_t luid = 0;
    static_assert(sizeof(luid) == sizeof(luid_bytes), "LUID is 8 bytes");
    std::memcpy(&luid, luid_bytes, sizeof(luid));
    if (device_selection::CanonicalLuid(luid) == identity.value) {
      return ordinal;
    }
#else
    char pci_bus_id[32] = {};
    if (hipDeviceGetPCIBusId(pci_bus_id, static_cast<int>(sizeof(pci_bus_id)),
                             ordinal) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    device_selection::PciBusId parsed;
    if (device_selection::ParsePciBusId(pci_bus_id, &parsed) &&
        device_selection::CanonicalPciBusId(parsed) == identity.value) {
      return ordinal;
    }
#endif
  }

  *error = "No HIP device matches the GPU selected in ONNX Runtime.";
  return -1;
}

OrtStatus *MakeMemoryInfo(const OrtApi &api, const char *name,
                          uint32_t vendor_id, uint32_t device_id,
                          OrtDeviceMemoryType memory_type,
                          MorphiZenEpFactory::MemoryInfoPtr *out) {
  OrtMemoryInfo *raw = nullptr;
  OrtStatus *status = api.CreateMemoryInfo_V2(
      name, OrtMemoryInfoDeviceType_GPU, vendor_id, device_id, memory_type,
      /*alignment*/ 0, OrtAllocatorType::OrtDeviceAllocator, &raw);
  if (status != nullptr) {
    return status;
  }
  *out = MorphiZenEpFactory::MemoryInfoPtr(raw, api.ReleaseMemoryInfo);
  return nullptr;
}

} // namespace

OrtStatus *MorphiZenEpFactory::ResolveSlotOrdinal(size_t slot,
                                                  int *hip_ordinal) {
  std::lock_guard<std::mutex> lock(gpu_device_slots_mutex_);
  if (slot >= gpu_device_slots_.size()) {
    return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                "GPU allocator slot is not registered.");
  }
  GpuDeviceSlot &entry = gpu_device_slots_[slot];
  if (!entry.ordinal_resolved) {
    int resolved = 0;
    if (entry.identity.kind == device_selection::PlatformIdKind::Malformed) {
      return ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "Selected GPU platform identity could not be parsed.");
    }
    if (entry.identity.kind != device_selection::PlatformIdKind::None) {
      std::string error;
      resolved = FindHipOrdinal(entry.identity, &error);
      if (resolved < 0) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
      }
    }
    entry.hip_ordinal = resolved;
    entry.ordinal_resolved = true;
  }
  *hip_ordinal = entry.hip_ordinal;
  return nullptr;
}

int MorphiZenEpFactory::SlotIndexForIdentity(
    const device_selection::PlatformId &identity) const {
  std::lock_guard<std::mutex> lock(gpu_device_slots_mutex_);
  for (size_t i = 0; i < gpu_device_slots_.size(); ++i) {
    const GpuDeviceSlot &slot = gpu_device_slots_[i];
    if (slot.identity.kind == identity.kind &&
        slot.identity.value == identity.value) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int MorphiZenEpFactory::SlotIndexForDeviceId(uint32_t device_id) const {
  std::lock_guard<std::mutex> lock(gpu_device_slots_mutex_);
  for (size_t i = 0; i < gpu_device_slots_.size(); ++i) {
    if (gpu_device_slots_[i].ort_device_id == device_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}
#endif

OrtStatus *ORT_API_CALL MorphiZenEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory *this_ptr, const OrtHardwareDevice *const *devices,
    size_t num_devices, OrtEpDevice **ep_devices, size_t max_ep_devices,
    size_t *p_num_ep_devices) noexcept {
  size_t &num_ep_devices = *p_num_ep_devices;
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);

  // A factory serves one device mode: production AMD GPUs, or the CPU debug
  // fallback. Each registered GPU gets its own allocator slot. See the
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
#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
    device_selection::PlatformId platform_id;
    if (!cpu_debug_mode) {
      // Record the identity ORT already discovered. This does not call HIP.
      platform_id = ReadPlatformId(factory->ort_api, hardware_device);
      if (platform_id.kind == device_selection::PlatformIdKind::PciBusId ||
          platform_id.kind == device_selection::PlatformIdKind::Luid) {
        factory->ort_api.AddKeyValuePair(
            ep_metadata, device_selection::PlatformIdMetadataKey(),
            platform_id.value.c_str());
      }
    }
#endif
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
      const uint32_t ort_device_id =
          factory->ort_api.HardwareDevice_DeviceId(hardware_device);
      MorphiZenEpFactory::GpuDeviceSlot slot;
      slot.identity = platform_id;
      slot.ort_device_id = ort_device_id;
      if (auto *st = MakeMemoryInfo(
              factory->ort_api, "MorphiZen", factory->vendor_id_, ort_device_id,
              OrtDeviceMemoryType_DEFAULT, &slot.default_memory)) {
        return st;
      }
      if (auto *st =
              MakeMemoryInfo(factory->ort_api, "MorphiZen host accessible",
                             factory->vendor_id_, ort_device_id,
                             OrtDeviceMemoryType_HOST_ACCESSIBLE,
                             &slot.host_accessible_memory)) {
        return st;
      }

      if (auto *st = ep_api_ptr->EpDevice_AddAllocatorInfo(
              registered_ep_device, slot.default_memory.get())) {
        return st;
      }
      if (auto *st = ep_api_ptr->EpDevice_AddAllocatorInfo(
              registered_ep_device, slot.host_accessible_memory.get())) {
        return st;
      }
      factory->gpu_device_slots_.push_back(std::move(slot));
    }
#endif
  }
  return nullptr;
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::CreateEpImpl(
    OrtEpFactory *this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice *const *devices,
    _In_reads_(num_devices) const OrtKeyValuePairs *const *ep_metadata,
    _In_ size_t num_devices, _In_ const OrtSessionOptions *session_options,
    _In_ const OrtLogger *logger, _Out_ OrtEp **ep) noexcept {
  MY_LOG(1) << "CreateEpImpl: num_devices=" << num_devices;
  auto *factory = static_cast<MorphiZenEpFactory *>(this_ptr);
  *ep = nullptr;

  // A session runs on one GPU. Zero forwarded devices keep ordinal 0 so a
  // parent EP that does not pass a selection still works. CPU debug mode
  // never touches HIP, so session creation still works without a GPU.
  if (num_devices > 1) {
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MorphiZen EP supports one selected GPU per session.");
  }
  const OrtKeyValuePairs *metadata_for_ep =
      (num_devices > 0 && ep_metadata != nullptr) ? ep_metadata[0] : nullptr;

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
  const bool cpu_debug_mode = ENV_PARAM(MORPHIZEN_EP_ENABLE_CPU_DEVICE) != 0;
  OrtKeyValuePairs *merged_metadata = nullptr;
  if (!cpu_debug_mode) {
    const OrtHardwareDevice *selected =
        (num_devices == 1 && devices != nullptr) ? devices[0] : nullptr;
    const device_selection::PlatformId identity =
        ReadPlatformId(factory->ort_api, selected);
    const auto decision = device_selection::Decide(num_devices, identity.kind);
    int hip_ordinal = 0;
    if (decision ==
        device_selection::DeviceSelectionDecision::RejectMalformed) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "Selected GPU platform identity could not be parsed.");
    }
    if (decision == device_selection::DeviceSelectionDecision::MatchIdentity) {
      const int slot = factory->SlotIndexForIdentity(identity);
      if (slot < 0) {
        return factory->ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "Selected GPU was not registered by this EP factory.");
      }
      if (auto *status = factory->ResolveSlotOrdinal(static_cast<size_t>(slot),
                                                     &hip_ordinal)) {
        return status;
      }
    }
    if (hipSetDevice(hip_ordinal) != hipSuccess) {
      (void)hipGetLastError();
      return factory->ort_api.CreateStatus(
          ORT_RUNTIME_EXCEPTION, "hipSetDevice failed for the selected GPU.");
    }

    factory->ort_api.CreateKeyValuePairs(&merged_metadata);
    if (metadata_for_ep != nullptr) {
      const char *const *keys = nullptr;
      const char *const *values = nullptr;
      size_t num_keys = 0;
      factory->ort_api.GetKeyValuePairs(metadata_for_ep, &keys, &values,
                                        &num_keys);
      for (size_t i = 0; i < num_keys; ++i) {
        if (keys[i] != nullptr) {
          factory->ort_api.AddKeyValuePair(
              merged_metadata, keys[i], values[i] != nullptr ? values[i] : "");
        }
      }
    }
    const std::string ordinal_text = std::to_string(hip_ordinal);
    factory->ort_api.AddKeyValuePair(merged_metadata, "hip_device_id",
                                     ordinal_text.c_str());
    metadata_for_ep = merged_metadata;
  }
  struct MetadataRelease {
    const OrtApi &api;
    OrtKeyValuePairs *pairs;
    ~MetadataRelease() {
      if (pairs != nullptr) {
        api.ReleaseKeyValuePairs(pairs);
      }
    }
  } release{factory->ort_api, merged_metadata};
#else
  (void)devices;
#endif

  // Create the execution provider
  factory->throw_if_error(factory->ort_api.Logger_LogMessage(
      logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Creating MorphiZen EP",
      ORT_FILE, __LINE__, __FUNCTION__));

  const OrtKeyValuePairs *metadata_list[] = {metadata_for_ep};
  auto morphizen_ep = std::make_unique<MorphiZenEP>(
      *factory, factory->ep_name_, metadata_list, *session_options, *logger);

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
    int device_id = 0;
    if (auto *status =
            factory->ort_api.MemoryInfoGetId(memory_info, &device_id)) {
      return status;
    }
    // The id is the hardware device id stored at registration. An id this
    // factory did not record is a caller error, including an empty table.
    const int slot =
        device_id < 0
            ? -1
            : factory->SlotIndexForDeviceId(static_cast<uint32_t>(device_id));
    if (slot < 0) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT, "GPU allocator device id is not registered.");
    }
    int hip_ordinal = 0;
    if (auto *status = factory->ResolveSlotOrdinal(static_cast<size_t>(slot),
                                                   &hip_ordinal)) {
      return status;
    }
    *allocator =
        new HipGpuAllocator(memory_info, factory->ort_api, hip_ordinal);
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

// ORT calls GetNumCustomOpDomainsImpl then GetCustomOpDomainsImpl back to
// back and requires both to agree on the same set; custom_op_domains_ is
// populated once in the constructor (see MorphiZenEpFactory::
// MorphiZenEpFactory above), so both accessors simply read the same
// factory-owned vector.
OrtStatus *ORT_API_CALL MorphiZenEpFactory::GetNumCustomOpDomainsImpl(
    OrtEpFactory *this_ptr, size_t *num_domains) noexcept {
  *num_domains =
      static_cast<MorphiZenEpFactory *>(this_ptr)->custom_op_domains_.size();
  return nullptr;
}

OrtStatus *ORT_API_CALL MorphiZenEpFactory::GetCustomOpDomainsImpl(
    OrtEpFactory *this_ptr, OrtCustomOpDomain **domains,
    size_t num_domains) noexcept {
  const auto &domains_vec =
      static_cast<MorphiZenEpFactory *>(this_ptr)->custom_op_domains_;
  for (size_t i = 0; i < num_domains && i < domains_vec.size(); ++i) {
    domains[i] = domains_vec[i];
  }
  return nullptr;
}

} // namespace morphizen
