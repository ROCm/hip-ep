/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include <vector>

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
// Pull in the full HipDataTransferImpl definition so that
// std::unique_ptr<HipDataTransferImpl> in MorphiZenEpFactory below has a
// complete type when this header is included from translation units that
// instantiate the destructor. The header itself does not include any HIP
// runtime headers, so this stays cheap for callers.
#include "./morphizen-hip-gpu-allocator.hpp"
#include <memory>
#endif

namespace morphizen {

struct MorphiZenEpFactory : OrtEpFactory, ApiPtrs {
  MorphiZenEpFactory(const char *ep_name, ApiPtrs apis,
                     const OrtLogger &default_logger);

  static const char *ORT_API_CALL
  GetNameImpl(const OrtEpFactory *this_ptr) noexcept;

  static const char *ORT_API_CALL
  GetVendorImpl(const OrtEpFactory *this_ptr) noexcept;

  static uint32_t ORT_API_CALL
  GetVendorIdImpl(const OrtEpFactory *this_ptr) noexcept;

  static const char *ORT_API_CALL
  GetVersionImpl(const OrtEpFactory *this_ptr) noexcept;

  static OrtStatus *ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory *this_ptr, const OrtHardwareDevice *const *devices,
      size_t num_devices, OrtEpDevice **ep_devices, size_t max_ep_devices,
      size_t *p_num_ep_devices) noexcept;

  static OrtStatus *ORT_API_CALL CreateEpImpl(
      OrtEpFactory *this_ptr,
      _In_reads_(num_devices) const OrtHardwareDevice *const * /*devices*/,
      _In_reads_(num_devices) const OrtKeyValuePairs *const * /*ep_metadata*/,
      _In_ size_t num_devices,
      _In_ const OrtSessionOptions * /*session_options*/,
      _In_ const OrtLogger *logger, _Out_ OrtEp **ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory * /*this_ptr*/,
                                         OrtEp * /*ep*/) noexcept;

  // Complete Use Case Flow:
  //
  // COMPILE TIME (Development):
  // 1. Developer: model.compile(ep_context=True)
  // 2. MorphiZenEP processes model through backends (DPU, DD, etc.)
  // 3. Each backend: PassContext::append_compiled_model_compatibility_info()
  // 4. MorphiZenEP::GetCompiledModelCompatibilityInfo() aggregates & serializes
  // to JSON
  // 5. ORT embeds JSON in EP context model file
  //
  // RUNTIME (Production):
  // 1. User: session = ort.InferenceSession("model.onnx")
  // 2. ORT reads EP context model, extracts compatibility JSON
  // 3. ORT calls THIS METHOD → ValidateCompiledModelCompatibilityInfoImpl()
  // 4. This method:
  //    - Parses JSON
  //    - For each backend: finds plugin, calls validation function
  //    - Aggregates results (worst case wins)
  // 5. Returns compatibility status to ORT (most restrictive wins):
  //    - EP_UNSUPPORTED → Reject model, show error (highest priority)
  //    - EP_SUPPORTED_PREFER_RECOMPILATION → Load but warn user
  //    - EP_SUPPORTED_OPTIMAL → Load model (best case)
  //    - EP_NOT_APPLICABLE → ORT decides (graceful fallback)

  static OrtStatus *ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory *this_ptr,
      _In_reads_(num_devices) const OrtHardwareDevice *const *devices,
      _In_ size_t num_devices, _In_ const char *compatibility_info,
      _Out_ OrtCompiledModelCompatibility *model_compatibility) noexcept;

  static OrtStatus *ORT_API_CALL
  CreateAllocatorImpl(OrtEpFactory *this_ptr, const OrtMemoryInfo *memory_info,
                      const OrtKeyValuePairs * /*allocator_options*/,
                      OrtAllocator **allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory * /*this*/, OrtAllocator *allocator) noexcept;

  static OrtStatus *ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory *this_ptr, OrtDataTransferImpl **data_transfer) noexcept;

  static bool ORT_API_CALL
  IsStreamAwareImpl(const OrtEpFactory * /*this_ptr*/) noexcept;

  static OrtStatus *ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory *this_ptr, const OrtMemoryDevice * /*memory_device*/,
      const OrtKeyValuePairs * /*stream_options*/,
      OrtSyncStreamImpl **stream) noexcept;

  // ORT calls GetNumCustomOpDomainsImpl once to size its array, then
  // GetCustomOpDomainsImpl to fill it; both must observe the same result.
  static OrtStatus *ORT_API_CALL GetNumCustomOpDomainsImpl(
      OrtEpFactory *this_ptr, size_t *num_domains) noexcept;
  static OrtStatus *ORT_API_CALL
  GetCustomOpDomainsImpl(OrtEpFactory *this_ptr, OrtCustomOpDomain **domains,
                         size_t num_domains) noexcept;

  const OrtLogger &default_logger_; // default logger for the EP factory
  const std::string ep_name_;       // EP name
  const std::string vendor_{"AMD"}; // EP vendor name
  // PCI vendor id of the AMD GPU we serve (== OrtDevice::VendorIds::AMD).
  // The AMD CPU / NPU PCI vendor id is 0x1022 (AuthenticAMD), which is a
  // different value but is not relevant for the production GPU path.
  const uint32_t vendor_id_{0x1002};
  const std::string ep_version_{"0.1.0"}; // EP version
  std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs *)>
      ep_metadata_; // EP metadata
  std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs *)>
      ep_options_; // EP metadata

  // Custom op domains from this EP factory.
  std::vector<OrtCustomOpDomain *> custom_op_domains_;

#if defined(MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR) &&                             \
    MORPHIZEN_ENABLE_HIP_GPU_ALLOCATOR
  // Lazily-created OrtMemoryInfo describing where the EP's GPU allocator places
  // tensors. Owned by the factory; OrtEpDevice copies them when registered via
  // EpDevice_AddAllocatorInfo, so we may free them when the factory dies.
  using MemoryInfoPtr =
      std::unique_ptr<OrtMemoryInfo, void (*)(OrtMemoryInfo *)>;
  MemoryInfoPtr gpu_memory_info_{nullptr, nullptr}; // hipMalloc-backed
  MemoryInfoPtr gpu_host_accessible_memory_info_{
      nullptr, nullptr}; // hipHostMalloc-backed

  // Single shared OrtDataTransferImpl returned from CreateDataTransferImpl.
  std::unique_ptr<HipDataTransferImpl> data_transfer_impl_;
#endif
};
} // namespace morphizen
