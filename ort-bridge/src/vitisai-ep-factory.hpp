/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
namespace morphizen {
struct VitisAiEpFactory : OrtEpFactory, ApiPtrs {
  VitisAiEpFactory(const char* ep_name, ApiPtrs apis);

  static const char* ORT_API_CALL
  GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL
  GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static uint32_t ORT_API_CALL
  GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL
  GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr, const OrtHardwareDevice* const* devices,
      size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
      _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
      _In_ size_t num_devices,
      _In_ const OrtSessionOptions* /*session_options*/,
      _In_ const OrtLogger* logger, _Out_ OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/,
                                         OrtEp* /*ep*/) noexcept;
  static OrtStatus* ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                      const OrtKeyValuePairs* /*allocator_options*/,
                      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept;

  const std::string ep_name_;             // EP name
  const std::string vendor_{"AMD"};       // EP vendor name
  const uint32_t vendor_id_{0x1002};      // EP vendor ID
  const std::string ep_version_{"0.1.0"}; // EP version
  std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs*)>
      ep_metadata_; // EP metadata
  std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs*)>
      ep_options_; // EP metadata
};
} // namespace morphizen
