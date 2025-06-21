/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
namespace morphizen {
struct VitisAiEpFactory : OrtEpFactory, ApiPtrs {
  VitisAiEpFactory(const char* ep_name, ApiPtrs apis);

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr);

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr);

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* p_num_ep_devices);

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
      _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
      _In_ size_t num_devices,
      _In_ const OrtSessionOptions* /*session_options*/,
      _In_ const OrtLogger* logger, _Out_ OrtEp** ep);

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                         OrtEp* /*ep*/);

  const std::string ep_name_;           // EP name
  const std::string vendor_{"Contoso"}; // EP vendor name
};
} // namespace morphizen
