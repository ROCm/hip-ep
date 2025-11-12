/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./vitisai-ep-factory.hpp"
#include "./vitisai-ep.hpp"
#include "model_compatibility.pb.h"
#include "morphizen-utils/morphizen-utils.hpp"
#include "morphizen-utils/vaip_plugin.hpp"
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP_FACTORY, "0")
DEF_ENV_PARAM(MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE, "0")
#define MY_LOG(n)                                                              \
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP_FACTORY) >= n)
namespace morphizen {

VitisAiEpFactory::VitisAiEpFactory(const char* ep_name, ApiPtrs apis,
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
    const OrtHardwareDevice* hardware_device = devices[i];
    const std::uint32_t vendor_id =
        factory->ort_api.HardwareDevice_VendorId(hardware_device);
    const OrtHardwareDeviceType device_type =
        factory->ort_api.HardwareDevice_Type(hardware_device);
    static constexpr std::uint32_t hardware_vendor_id{0x1022};

    if (ENV_PARAM(MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE)) {
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

// clang-format off
// ┌───────────────────────────────────────────────────────────────────────────────────────────┐
// │                Compiled Model Compatibility Validation - Use Cases                       │
// │   Dimensions: Compilation EP | Runtime EP | Feature Support | Version Compatibility      │
// └───────────────────────────────────────────────────────────────────────────────────────────┘
//
// Terminology (to avoid ambiguity):
//   Feature Capability:
//     • Legacy EP:  EP WITHOUT ValidateCompiledModelCompatibilityInfo feature
//     • Modern EP:  EP WITH ValidateCompiledModelCompatibilityInfo feature
//
//   Version Naming:
//     • EP v1.x, v2.0, v2.1, etc. - Specific version numbers
//     • Same version: Compile version == Runtime version (e.g., v2.0 → v2.0)
//     • Different versions: Compile version != Runtime version (e.g., v2.0 → v2.1)
//
//   Other:
//     • Compat Info: Compatibility metadata embedded in compiled model
//
// Return Values:
//   • EP_SUPPORTED_OPTIMAL: Fully compatible, no recompilation needed
//   • EP_SUPPORTED_PREFER_RECOMPILATION: Compatible but recompilation recommended
//   • EP_UNSUPPORTED: Incompatible, model cannot run on this runtime
//   • EP_NOT_APPLICABLE: No compatibility check performed (default ORT behavior)
//
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Case 1: Modern EP (Compile) → Modern EP (Runtime)
// ───────────────────────────────────────────────────────────────────────────────────────────
//   Compile EP:  Modern EP (has compatibility feature)
//   Runtime EP:  Modern EP (has compatibility feature)
//   Compat Info: Yes (generated and validated)
//
//   How It Works:
//     1. Compile EP embeds compatibility info into the compiled model
//     2. Runtime EP calls ValidateCompiledModelCompatibilityInfo
//     3. Backend validation logic determines compatibility
//     4. Result depends on what Runtime EP's validation logic returns
//
//   Possible Results (determined by Runtime EP's validation):
//     • EP_SUPPORTED_OPTIMAL             - Fully compatible, optimal performance
//     • EP_SUPPORTED_PREFER_RECOMPILATION - Compatible but recompilation recommended
//     • EP_UNSUPPORTED                   - Incompatible, cannot run
//     • EP_NOT_APPLICABLE                - Backend cannot determine compatibility
//
//   Example Scenarios:
//
//   ✓ Same Version (Most Common - Optimal)
//     Compile: Modern EP v2.0  →  Runtime: Modern EP v2.0
//     Result:  EP_SUPPORTED_OPTIMAL
//     Reason:  Identical versions guarantee full compatibility
//
//   ? Cross-Version (Result Varies - Backend Decides)
//     Compile: Modern EP v2.0  →  Runtime: Modern EP v1.5
//     Result:  Depends on Runtime EP v1.5's validation logic
//              - Can be EP_SUPPORTED_OPTIMAL if backward compatible
//              - Can be EP_SUPPORTED_PREFER_RECOMPILATION if suboptimal
//              - Can be EP_UNSUPPORTED if incompatible
//
//     Compile: Modern EP v1.5  →  Runtime: Modern EP v2.0
//     Result:  Depends on Runtime EP v2.0's validation logic
//              - Can be EP_SUPPORTED_OPTIMAL if forward compatible
//              - Can be EP_SUPPORTED_PREFER_RECOMPILATION if can optimize
//              - Can be EP_UNSUPPORTED if breaking changes exist
//
//   ✗ Hardware Platform Changes (Typically Unsupported)
//     Compile: Modern EP (AIE2/Phoenix)  →  Runtime: Modern EP (AIE2P/Strix)
//     Result:  EP_UNSUPPORTED
//     Reason:  Different hardware platforms, incompatible binaries
//
//   ✗ Build Configuration Mismatch (Backend Missing)
//     Compile: Modern EP (with xcompiler backend)  →  Runtime: Modern EP (no xcompiler)
//     Result:  EP_UNSUPPORTED
//     Reason:  Required backend plugin not found at runtime
//
//   ✗ Backend Name Changed (Breaking Change)
//     Compile: Modern EP (backend="old_name")  →  Runtime: Modern EP (backend="new_name")
//     Result:  EP_UNSUPPORTED
//     Reason:  Backend plugin name mismatch, required backend not found
//
//   Purpose: PRIMARY use case - enables intelligent cross-version compatibility checking
//
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Case 2: Modern EP (Compile) → Legacy EP (Runtime)
// ───────────────────────────────────────────────────────────────────────────────────────────
//   Compile EP:  Modern EP (has compatibility feature)
//   Runtime EP:  Legacy EP (lacks ValidateCompiledModelCompatibilityInfo)
//   Compat Info: Yes (generated but ignored)
//   Result:      EP_NOT_APPLICABLE
//
//   Reason: Legacy runtime doesn't have validation method, compatibility info is silently
//           ignored. Model may or may not work depending on actual compatibility.
//
//   Purpose: Backward compatibility - models from modern EP can attempt to run on legacy EP
//
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Case 3: Legacy EP (Compile) → Modern EP (Runtime)
// ───────────────────────────────────────────────────────────────────────────────────────────
//   Compile EP:  Legacy EP (no compatibility feature)
//   Runtime EP:  Modern EP (has compatibility feature)
//   Compat Info: No (not generated during compilation)
//   Result:      EP_NOT_APPLICABLE
//
//   Reason: No compatibility information available to validate. Modern runtime falls back
//           to default ORT behavior and attempts to load the model.
//
//   Purpose: Backward compatibility - models from legacy EP continue to work on modern EP
//
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Case 4: Legacy EP (Compile) → Legacy EP (Runtime)
// ───────────────────────────────────────────────────────────────────────────────────────────
//   Compile EP:  Legacy EP (no compatibility feature)
//   Runtime EP:  Legacy EP (no compatibility feature)
//   Compat Info: No (feature doesn't exist)
//   Result:      EP_NOT_APPLICABLE
//
//   Reason: Neither side has compatibility checking capability. This is the traditional
//           pre-feature behavior.
//
//   Purpose: Baseline behavior before compatibility feature was introduced
//
// ═══════════════════════════════════════════════════════════════════════════════════════════
// clang-format on
OrtStatus* ORT_API_CALL
VitisAiEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
    _In_ size_t num_devices, _In_ const char* compatibility_info,
    _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept {
  (void)this_ptr;
  // if compatibility_info is null or empty, return EP_NOT_APPLICABLE
  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }
  morphizen::ModelCompatibilityProto compatibility_proto;
  auto status = google::protobuf::util::JsonStringToMessage(
      compatibility_info, &compatibility_proto);
  if (!status.ok()) {
    MY_LOG(1) << "Failed to parse ModelCompatibilityProto: "
              << status.message();
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  // clang-format off
  // Is backend compatibility info generated? (Y/N)
  //   |
  //   +-- N --> (compile time EP not include Compatibility checking feature) Keep ORT default behavior --> Return EP_NOT_APPLICABLE
  //   |
  //   +-- Y
  //       |
  //       v
  //    Is backend `plugin` found?
  //        |
  //        +-- No --> Runtime EP not built with backend/ runtime EP `backend name` changed, but EP ctx model expected run the original backend
  //               --> Return EP_NOT_SUPPORTED
  //        |
  //        +-- Yes
  //              |
  //              v
  //         Is `morphizen_OrtCompiledModelCompatibility` function pointer (fp) found?
  //              |
  //              +-- Yes --> Return backend's own compatibility checking result
  //              |
  //              +-- No  --> Runtime EP did not register compatibility checking function --> Return EP_NOT_APPLICABLE
  // clang-format on
  auto compatibility_infos = std::vector<OrtCompiledModelCompatibility>();
  bool any_plugin_missing = false;

  for (const auto& entry : compatibility_proto.backend_compatibility()) {
    MY_LOG(2) << "Backend: " << entry.first
              << ", Compatibility Info: " << entry.second;
    auto plugin = vaip_core::Plugin::get(entry.first);
    // Check if the corresponding backend plugin exists in the current EP
    // (Execution Provider) environment.
    // if the plugin is not found , it means the compiled model was built with
    // the backend that is not available in the current runtime setup.
    // for example, when:
    // - The EP shared library in the production environment of the compiled
    // model was built with that backend enabled.
    // - The current EP shared library was not built with support for that
    // backend.
    // In such a case, the compiled model expects to run on the backend, but it
    // was not found in this backend, we should return EP_UNSUPPORTED.
    if (!plugin) {
      MY_LOG(1) << "Backend plugin " << entry.first << " is not available.";
      any_plugin_missing = true;
      continue; // Check remaining backends
    }

    auto fp =
        plugin
            ->get_method<OrtCompiledModelCompatibility,
                         const OrtHardwareDevice* const*, size_t, const char*>(
                "morphizen_OrtCompiledModelCompatibility");
    // if the fp is not found , it means that the backend has not properly
    // registered or implemented morphizen_OrtCompiledModelCompatibility.
    // In such a case , not support the CompiledModelCompatibility checking
    // and Keep ORT default behavior --> this backend returns EP_NOT_APPLICABLE
    if (!fp) {
      MY_LOG(1) << "Backend plugin " << entry.first
                << " does not support morphizen_OrtCompiledModelCompatibility.";
      compatibility_infos.push_back(
          OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
      continue; // Check remaining backends
    }

    OrtCompiledModelCompatibility backend_compatibility =
        fp(devices, num_devices, entry.second.c_str());
    compatibility_infos.push_back(backend_compatibility);
  }

  // Handle cases where some backends couldn't be validated
  if (any_plugin_missing) {
    MY_LOG(1) << "One or more required backend plugins are missing. "
              << "Successfully validated " << compatibility_infos.size()
              << " of " << compatibility_proto.backend_compatibility().size()
              << " backends.";
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    return nullptr;
  }

  if (compatibility_infos.empty()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }
  // The return value :
  // typedef enum OrtCompiledModelCompatibility {
  // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE = 0,
  // OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL,
  // OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION,
  // OrtCompiledModelCompatibility_EP_UNSUPPORTED,
  // } OrtCompiledModelCompatibility;
  if (std::find(compatibility_infos.begin(), compatibility_infos.end(),
                OrtCompiledModelCompatibility_EP_UNSUPPORTED) !=
      compatibility_infos.end()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  } else if (
      std::find(
          compatibility_infos.begin(), compatibility_infos.end(),
          OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION) !=
      compatibility_infos.end()) {
    *model_compatibility =
        OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
  } else if (std::find(compatibility_infos.begin(), compatibility_infos.end(),
                       OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) !=
             compatibility_infos.end()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  } else {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr, const OrtMemoryInfo* /*memory_info*/,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<VitisAiEpFactory*>(this_ptr);

  *allocator = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateAllocator should not be called as we did "
                            "not add OrtMemoryInfo to our OrtEpDevice.");
}

void ORT_API_CALL VitisAiEpFactory::ReleaseAllocatorImpl(
    OrtEpFactory* /*this*/, OrtAllocator* /*allocator*/) noexcept {
  // should never be called as we don't implement CreateAllocator
  // TODO : implement release allocator if needed
  LOG(FATAL) << "TODO";
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateDataTransferImpl(
    OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr; // not implemented
  return nullptr;
}

bool ORT_API_CALL
VitisAiEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

OrtStatus* ORT_API_CALL VitisAiEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr, const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {
  auto* factory = static_cast<VitisAiEpFactory*>(this_ptr);

  *stream = nullptr;
  return factory->ort_api.CreateStatus(
      ORT_INVALID_ARGUMENT, "CreateSyncStreamForDevice should not be called as "
                            "IsStreamAware returned false.");
}

} // namespace morphizen
