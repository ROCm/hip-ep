/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./vitisai-ep-factory.hpp"
#include "./vitisai-ep.hpp"
#include "morphizen-utils/morphizen-utils.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP_FACTORY, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP_FACTORY) >= n)
namespace morphizen {

VitisAiEpFactory::VitisAiEpFactory(const char* ep_name, ApiPtrs apis)
    : ApiPtrs(apis), ep_name_{ep_name},
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
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
}
const char* ORT_API_CALL
VitisAiEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const VitisAiEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL
VitisAiEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const VitisAiEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}
uint32_t ORT_API_CALL
VitisAiEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const VitisAiEpFactory*>(this_ptr);
  return factory->vendor_id_;
}
const char* ORT_API_CALL
VitisAiEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const VitisAiEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
    size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<VitisAiEpFactory*>(this_ptr);

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    // C API
    const OrtHardwareDevice& device = *devices[i];
    if (factory->ort_api.HardwareDevice_Type(&device) ==
        OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
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
      // random example using made up values
      // factory->ort_api.AddKeyValuePair(ep_metadata, "version", "0.1");
      factory->ort_api.AddKeyValuePair(ep_options, "run_really_fast", "true");

      // OrtEpDevice copies ep_metadata and ep_options.
      auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(
          factory, &device, ep_metadata, ep_options,
          &ep_devices[num_ep_devices++]);
      if (status != nullptr) {
        return status;
      }
    }
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
    _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
    _In_ size_t num_devices, _In_ const OrtSessionOptions* session_options,
    _In_ const OrtLogger* logger, _Out_ OrtEp** ep) noexcept {
  MY_LOG(1) << "CreateEpImpl: ";
  auto* factory = static_cast<VitisAiEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for NPU and only expected to be selected for one NPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you
    // will get an entry for each device the EP has been selected for.
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "VitisAI EP only supports selection for one device.");
  }

  // Create the execution provider
  factory->throw_if_error(factory->ort_api.Logger_LogMessage(
      logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO, "Creating VitisAI EP",
      ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  auto vitisai_ep = std::make_unique<VitisAIEP>(
      *factory, factory->ep_name_, ep_metadata, *session_options, *logger);

  *ep = vitisai_ep.release();
  return nullptr;
}
void ORT_API_CALL VitisAiEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                                  OrtEp* ep) noexcept {
  VitisAIEP* vitisai_ep = static_cast<VitisAIEP*>(ep);
  delete vitisai_ep;
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateAllocatorImpl(
    OrtEpFactory* /*this_ptr*/, const OrtMemoryInfo* /*memory_info*/,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** /*allocator*/) noexcept {
  return nullptr; // TODO: Implement allocator creation if needed
}

void ORT_API_CALL VitisAiEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this*/, OrtAllocator* /*allocator*/) noexcept {
  // TODO: Implement allocator release if needed
  LOG(FATAL) << "TODO";
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/,
    OrtDataTransferImpl** /*data_transfer*/) noexcept {
  return nullptr; // TODO: Implement data transfer if needed
}

} // namespace morphizen
