/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/morphizen.hpp"

#include "morphizen/custom_op_imp.hpp"
#include "morphizen/env_config.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4946)
#endif
#include "morphizen/model_compatibility.pb.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "morphizen/onnxruntime_morphizen_ep.hpp"
#include "morphizen/plugin.hpp"
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

DEF_ENV_PARAM(DEBUG_MODEL_COMPATIBILITY, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MODEL_COMPATIBILITY) >= n)

namespace morphizen {

// Thread-local storage for the result string to ensure thread safety
thread_local static std::string g_compiled_model_compatibility_info_result;

/**
 * @brief Gets the compiled model compatibility information from execution
 * providers.
 *
 * This is the core helper function that extracts compatibility information from
 * the pass context of execution providers and serializes it to JSON format.
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param graph_viewer Pointer to GraphViewer (for ORT) or OrtGraph (for
 * ort-bridge). Can be nullptr if not needed.
 * @return Pointer to the JSON string, or nullptr/empty string if unavailable.
 *         The returned pointer is valid until the next call to this function or
 * EP destruction.
 */
extern "C" MORPHIZEN_DLL_SPEC const char *get_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *eps,
    const void *graph_viewer) {
  (void)graph_viewer; // May be used in future for graph-specific compatibility
                      // info

  g_compiled_model_compatibility_info_result.clear();
  morphizen::ModelCompatibilityProto compatibility_info_proto;
  compatibility_info_proto.mutable_version()->set_major(1);
  compatibility_info_proto.mutable_version()->set_minor(0);
  compatibility_info_proto.mutable_version()->set_patch(0);
  // base64_encoding field reserved for future use when compatibility info
  // needs to be encoded (e.g. for binary data or special characters)
  compatibility_info_proto.set_base64_encoding(false);

  if (eps && !eps->empty()) {
    auto *ep_concrete = dynamic_cast<morphizen::ExecutionProviderConcrete *>(
        eps->front().get());
    if (ep_concrete) {
      auto pass_context = ep_concrete->get_context();
      if (pass_context) {
        const auto &compatibility_info_map =
            pass_context->get_compiled_model_compatibility_info();
        if (compatibility_info_map.empty()) {
          MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] "
                       "Compatibility info map is empty. No backend "
                       "compatibility info is available.";
        }
        for (const auto &entry : compatibility_info_map) {
          (*compatibility_info_proto
                .mutable_backend_compatibility())[entry.first] = entry.second;
        }
      } else {
        MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] "
                     "PassContext is null. No backend compatibility info is "
                     "available.";
      }
    } else {
      MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] "
                   "Failed to cast to ExecutionProviderConcrete. No backend "
                   "compatibility info is available.";
    }
  } else {
    MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] "
                 "ExecutionProvider is empty. No backend compatibility info is "
                 "available.";
  }

  auto status = google::protobuf::util::MessageToJsonString(
      compatibility_info_proto, &g_compiled_model_compatibility_info_result);
  if (!status.ok()) {
    MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] Failed to "
                 "serialize ModelCompatibilityProto. Error: "
              << status.message();
    return g_compiled_model_compatibility_info_result.c_str();
  }

  MY_LOG(1) << " [MorphiZen EP][GetCompiledModelCompatibilityInfo] Compiled "
               "Model Compatibility Info: "
            << g_compiled_model_compatibility_info_result;
  // Returns a pointer valid until the next call to this method or EP
  // destruction. ORT is responsible for copying this string if needed beyond
  // that scope.
  return g_compiled_model_compatibility_info_result.c_str();
}

/**
 * @brief Validates the compiled model compatibility information.
 *
 * This is the core helper function that deserializes the compatibility info
 * JSON, queries backend plugins for their compatibility status, and determines
 * the overall compatibility level.
 *
 * @param compatibility_info JSON string containing compatibility information.
 * @param devices Array of hardware devices for validation.
 * @param num_devices Number of devices in the array.
 * @param model_compatibility Output parameter for the compatibility result.
 * @return 0 on success, non-zero on failure.
 */
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
extern "C" MORPHIZEN_DLL_SPEC int validate_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *eps,
    const char *compatibility_info, const void *const *devices,
    size_t num_devices, int *model_compatibility) {
  (void)eps; // May be used in future for EP-specific validation

  MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] "
               "ValidateCompiledModelCompatibilityInfo called with "
               "compatibility_info: "
            << compatibility_info;

  // If compatibility_info is null or empty, return EP_NOT_APPLICABLE
  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] "
                 "Compatibility info is null or empty. Return NOT_APPLICABLE.";
    *model_compatibility = 0; // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
    return 0;
  }

  morphizen::ModelCompatibilityProto compatibility_proto;
  auto status = google::protobuf::util::JsonStringToMessage(
      compatibility_info, &compatibility_proto);
  if (!status.ok()) {
    MY_LOG(1)
        << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] Failed "
           "to parse ModelCompatibilityProto. Error: "
        << status.message() << ". Return NOT_APPLICABLE.";
    *model_compatibility = 0; // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
    return 0;
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
  std::vector<int> compatibility_results;
  bool any_plugin_missing = false;
  for (const auto &entry : compatibility_proto.backend_compatibility()) {
    MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] "
                 "Validating backend: "
              << entry.first << " with compatibility info: " << entry.second;
    auto plugin = morphizen::Plugin::get(entry.first);
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
      MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] "
                   "Backend plugin "
                << entry.first
                << " is not available in the current EP environment. ";
      any_plugin_missing = true; // Check remaining backends
      continue;
    }

    auto fp =
        plugin->get_method<int, const void *const *, size_t, const char *>(
            "morphizen_OrtCompiledModelCompatibility");

    // if the fp is not found , it means that the backend has not properly
    // registered or implemented morphizen_OrtCompiledModelCompatibility.
    // In such a case , not support the CompiledModelCompatibility checking
    // and Keep ORT default behavior --> this backend returns EP_NOT_APPLICABLE
    if (!fp) {
      MY_LOG(1)
          << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] Backend  "
          << entry.first
          << " does not support morphizen_OrtCompiledModelCompatibility. "
             "cannot validate the backend compatibility. Return "
             "NOT_APPLICABLE.";
      compatibility_results.push_back(
          0); // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
      continue;
    }

    try {
      int backend_compatibility =
          fp(devices, num_devices, entry.second.c_str());
      MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] "
                   "Backend "
                << entry.first
                << " compatibility check result: " << backend_compatibility;
      compatibility_results.push_back(backend_compatibility);
    } catch (const std::exception &e) {
      LOG(ERROR) << "Exception in backend plugin " << entry.first
                 << " compatibility check: " << e.what()
                 << ". Using EP_NOT_APPLICABLE.";
      compatibility_results.push_back(
          0); // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
    } catch (...) {
      LOG(ERROR) << "Unknown exception in backend plugin " << entry.first
                 << " compatibility check. Using EP_NOT_APPLICABLE.";
      compatibility_results.push_back(
          0); // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
    }
  }

  // Handle cases where some backends couldn't be validated
  if (any_plugin_missing) {
    MY_LOG(1)
        << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] One or "
           "more required backend plugins are missing. "
        << "Return UNSUPPORTED.";
    *model_compatibility = 3; // OrtCompiledModelCompatibility_EP_UNSUPPORTED
    return 0;
  }

  if (compatibility_results.empty()) {
    MY_LOG(1)
        << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] No backend "
           "compatibility info is available. Return NOT_APPLICABLE.";
    *model_compatibility = 0; // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
    return 0;
  }

  // The return value :
  // typedef enum OrtCompiledModelCompatibility {
  // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE = 0,
  // OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL,
  // OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION,
  // OrtCompiledModelCompatibility_EP_UNSUPPORTED,
  // } OrtCompiledModelCompatibility;
  if (std::find(compatibility_results.begin(), compatibility_results.end(),
                3) != compatibility_results.end()) {
    *model_compatibility = 3; // OrtCompiledModelCompatibility_EP_UNSUPPORTED
  } else if (std::find(compatibility_results.begin(),
                       compatibility_results.end(),
                       2) != compatibility_results.end()) {
    *model_compatibility =
        2; // OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION
  } else if (std::find(compatibility_results.begin(),
                       compatibility_results.end(),
                       1) != compatibility_results.end()) {
    *model_compatibility =
        1; // OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL
  } else {
    *model_compatibility = 0; // OrtCompiledModelCompatibility_EP_NOT_APPLICABLE
  }

  MY_LOG(1) << " [MorphiZen EP][ValidateCompiledModelCompatibilityInfo] Model "
               "compatibility: "
            << *model_compatibility;
  return 0;
}

} // namespace morphizen
