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
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MORPHIZEN_EP_FACTORY, "0")
DEF_ENV_PARAM(MORPHIZEN_MORPHIZEN_EP_ENABLE_CPU_DEVICE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MORPHIZEN_EP_FACTORY) >= n)
namespace morphizen {

MorphiZenEpFactory::MorphiZenEpFactory(const char* ep_name, ApiPtrs apis,
                                       const OrtLogger& default_logger)
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
const char* ORT_API_CALL
MorphiZenEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MorphiZenEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL
MorphiZenEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MorphiZenEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}
uint32_t ORT_API_CALL
MorphiZenEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MorphiZenEpFactory*>(this_ptr);
  return factory->vendor_id_;
}
const char* ORT_API_CALL
MorphiZenEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const MorphiZenEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

OrtStatus* ORT_API_CALL MorphiZenEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<MorphiZenEpFactory*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice* hardware_device = devices[i];
    const std::uint32_t vendor_id =
        factory->ort_api.HardwareDevice_VendorId(hardware_device);
    const OrtHardwareDeviceType device_type =
        factory->ort_api.HardwareDevice_Type(hardware_device);
    static constexpr std::uint32_t hardware_vendor_id{0x1022};

    if (ENV_PARAM(MORPHIZEN_MORPHIZEN_EP_ENABLE_CPU_DEVICE)) {
      // only for internal test, we pretend to support CPU EP.
      if (device_type != OrtHardwareDeviceType_CPU) {
        continue;
      }
    } else {
      if ((vendor_id != factory->vendor_id_) ||
          (device_type != OrtHardwareDeviceType_NPU)) {
        continue;
      }
    }
    // these can be returned as nullptr if you have nothing to add.
    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    factory->ort_api.CreateKeyValuePairs(&ep_metadata);
    factory->ort_api.CreateKeyValuePairs(&ep_options);
    factory->ep_metadata_ =
        std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs*)>(
            ep_metadata, factory->ort_api.ReleaseKeyValuePairs);
    factory->ep_options_ =
        std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs*)>(
            ep_options, factory->ort_api.ReleaseKeyValuePairs);
    if (num_ep_devices == max_ep_devices) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT, "Not enough space to return EP devices.");
    }
    // OrtEpDevice copies ep_metadata and ep_options.
    auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(
        factory, hardware_device, ep_metadata, ep_options,
        &ep_devices[num_ep_devices++]);
    if (status != nullptr) {
      return status;
    }
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL MorphiZenEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
    _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
    _In_ size_t num_devices, _In_ const OrtSessionOptions* session_options,
    _In_ const OrtLogger* logger, _Out_ OrtEp** ep) noexcept {
  MY_LOG(1) << "CreateEpImpl: ";
  auto* factory = static_cast<MorphiZenEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for NPU and only expected to be selected for one NPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you
    // will get an entry for each device the EP has been selected for.
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "MorphiZen EP only supports selection for one device.");
  }

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
void ORT_API_CALL MorphiZenEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                                    OrtEp* ep) noexcept {
  MorphiZenEP* morphizen_ep = static_cast<MorphiZenEP*>(ep);
  delete morphizen_ep;
}

OrtStatus* ORT_API_CALL
MorphiZenEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
    _In_ size_t num_devices, _In_ const char* compatibility_info,
    _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept {
  auto* factory = static_cast<MorphiZenEpFactory*>(this_ptr);

  // Empty eps pointer since validation is based on compatibility_info and
  // devices eps may be used in future for EP-specific validation
  int compatibility_result = 0;
  int status = validate_compiled_model_compatibility_info(
      nullptr, // eps - not available in factory context
      compatibility_info, reinterpret_cast<const void* const*>(devices),
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

OrtStatus* ORT_API_CALL MorphiZenEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr, const OrtMemoryInfo* /*memory_info*/,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<MorphiZenEpFactory*>(this_ptr);

  *allocator = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateAllocator should not be called as we did "
                            "not add OrtMemoryInfo to our OrtEpDevice.");
}

void ORT_API_CALL MorphiZenEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this*/, OrtAllocator* /*allocator*/) noexcept {
  // should never be called as we don't implement CreateAllocator
  // TODO : implement release allocator if needed
  LOG(FATAL) << "TODO";
}

OrtStatus* ORT_API_CALL MorphiZenEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr; // not implemented
  return nullptr;
}

bool ORT_API_CALL MorphiZenEpFactory::IsStreamAwareImpl(
    const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

OrtStatus* ORT_API_CALL MorphiZenEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr, const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  auto* factory = static_cast<MorphiZenEpFactory*>(this_ptr);

  *stream = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateSyncStreamForDevice should not be called as "
                            "IsStreamAware returned false.");
}

} // namespace morphizen
