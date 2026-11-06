/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/ort_api_wrapper.hpp"
#include <onnxruntime_c_api.h>
/**
 * @file onnxruntime_morphizen_ep.hpp
 * @brief Header file for the MorphiZen Execution Provider integration with ONNX
 * Runtime.
 *
 * This file contains function declarations and macros for initializing,
 * deinitializing, and managing the MorphiZen Execution Provider within the ONNX
 * Runtime framework.
 * It includes functions for setting dynamic options, compiling ONNX models,
 * creating EP context nodes, and retrieving the MorphiZen EP version.
 *
 * @note This header is part of the MorphiZen Execution Provider implementation
 * and is intended for internal use within the ONNX Runtime framework.
 */

extern "C" {
/**
 * @brief Initializes the MorphiZen Execution Provider for ONNX Runtime.
 *
 * The `MorphiZen_Provider::Initialize()` is invoked during EP registration
 * using the Plugin EP V2 API:
 * `Ort::GetApi().RegisterExecutionProviderLibrary(env,
 * "MorphiZenExecutionProvider", "onnxruntime_vitisai_ep.dll")` followed by
 * `session_options.AppendExecutionProvider_V2(env, selected_devices,
 * provider_options)`, prior to session creation. Called right after loading the
 * shared library.
 *
 * Features:
 * - Initialize the global API in onnxruntime_morphizen_ep dll/so.
 * - Register custom operations
 *
 * @param api Pointer to the OrtApiForMorphizen instance.
 * @param ret_domain Reference to a vector of OrtCustomOpDomain pointers to
 * store custom operation domains.
 */
MORPHIZEN_DLL_SPEC
void initialize_onnxruntime_morphizen_ep(
    morphizen::OrtApiForMorphizen *api,
    std::vector<OrtCustomOpDomain *> &ret_domain);

/**
 * @brief Deinitializes the MorphiZen Execution Provider for ONNX Runtime.
 *
 * The `MorphiZen_Provider::Shutdown()` is invoked during
 * `ProviderLibrary::Unload()`,prior to unloading the shared library. It cleans
 * up any resources allocated during initialization.
 */
MORPHIZEN_DLL_SPEC
void deinitialize_onnxruntime_morphizen_ep();

/**
 * @brief Collects every OrtCustomOpDomain contributed by
 * "morphizen_register_ops" plugin symbols (see morphizen/op_def.hpp's
 * OpRegister), independent of the ABI surface calling it.
 *
 * Does a fresh plugin scan on every call -- a caller that needs this to run
 * at most once must cache its own call (see MorphiZenEpFactory's
 * custom_op_domains_ member in
 * morphizen/ort-bridge/src/morphizen-ep-factory.cpp).
 *
 * @param ret_domain Domains are appended to this vector.
 */
void CollectCustomOpDomains(std::vector<OrtCustomOpDomain *> &ret_domain);

/**
 * @brief Called when InferenceSession::Run() started.
 * Enable user to set proformance mode for every session run.
 * Related to ORT #19521
 *
 * Called by `MorphiZenExecutionProvider::OnRunStart()`.
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param state Pointer to the runtime state.
 * @param get_config_entry Function pointer to retrieve run_option entries.
 * @return Status code indicating success or failure.
 */
MORPHIZEN_DLL_SPEC int morphizen_ep_on_run_start(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const void *state,
    morphizen::DllSafe<std::string> (*get_config_entry)(
        const void *state, const char *entry_name));

/**
 * @brief Called when InferenceSession::Run() finished.
 *
 * Drops the run_option accessor that morphizen_ep_on_run_start() installed for
 * the calling thread. The state it captured refers to the OrtRunOptions of that
 * run, which does not outlive the run, so it must stop being reachable from
 * PassContext::get_run_option() once the run ends.
 *
 * Called by `MorphiZenEP::OnRunEndImpl()`.
 *
 * @return Status code indicating success or failure.
 */
MORPHIZEN_DLL_SPEC int morphizen_ep_on_run_end();

/**
 * @brief Set DynamicOptions for the MorphiZen Execution Provider.
 *
 * Called when InferenceSession::SetEpDynamicOptions() is called.
 * Related to ORT #22282
 *
 * Called by `MorphiZenExecutionProvider::SetEpDynamicOptions()`.
 *
 * Valid options can be found in
 * `include\onnxruntime\core\session\onnxruntime_session_options_config_keys.h`
 * static const char* const kOrtEpDynamicOptionsWorkloadType =
 * "ep.dynamic.workload_type";
 *
 *  Specify the type of workload for this session.
 * "Default": OS determines the scheduling priority and processor performance to
 * service this workload. [Default] "Efficient": OS treats this workload is
 * efficiency oriented with low scheduling priority and efficient processor
 * performance.
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param keys Array of option keys.
 * @param values Array of option values.
 * @param kv_len Length of the key-value arrays.
 * @return Status code indicating success or failure.
 */
MORPHIZEN_DLL_SPEC int morphizen_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    const char *const *keys, const char *const *values, size_t kv_len);

/**
 * @brief Compiles an ONNX model using the MorphiZen Execution Provider.
 *
 * This is the primary API for ONNX model compilation that integrates glog with
 * ORT's logging system, allowing users to control log levels through ORT APIs.
 *
 * Called by `MorphiZenExecutionProvider::GetCapability()`.
 *
 * @param model_path Path to the ONNX model file.
 * @param graph Reference to the ONNX Runtime graph.
 * @param options Provider options for compilation.
 * @param session_configs Session configuration options.
 * @param status Pointer to a status object for error handling.
 * @param func Callback function for error handling.
 * @param ort_logger Pointer to ORT logger for log integration.
 * @return Pointer to a vector of unique pointers to ExecutionProvider
 * instances.
 */
MORPHIZEN_DLL_SPEC std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *
compile_onnx_model_morphizen_ep_v4(
    const std::string &model_path, const onnxruntime::Graph &graph,
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs, void *status,
    void (*func)(void *, int, const char *), const OrtLogger *ort_logger);

/**
 * @brief Creates EPContxt Nodes for the MorphiZen Execution Provider.
 *
 * Called after compile onnx model when ep.context_enbale is set to 1.
 * The method is called after both `GetCapability()` and `Compile()`.
 *
 * Called by `MorphiZenExecutionProvider::GetEpContextNodes()`.
 *
 * Enable Ep Context feature by sessin config option :
 * kOrtSessionOptionEpContextEnable = "ep.context_enable";
 * "0": disable. (default)
 * "1": enable.
 *
 * Specify the file path for the Onnx model by session config option:
 * kOrtSessionOptionEpContextFilePath = "ep.context_file_path";
 * Default to [original_file_name]_ctx.onnx if not specified.
 *
 * Flag to specify whether to dump the EP context into the Onnx model by :
 * kOrtSessionOptionEpContextEmbedMode = "ep.context_embed_mode";
 * "0": dump the EP context into separate file, keep the file name in the Onnx
 * model. (default).
 * "1": dump the EP context into the Onnx model.
 *
 * Enable shared EP context by session config option :
 * kOrtSessionOptionShareEpContexts = "ep.share_ep_contexts";
 * "0": disable(default).
 * "1": enable.
 * Related to VAI-10864
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param ret_value Pointer to a vector of ONNXRuntime  EPContext nodes to store
 * the result.
 * @return Status code indicating success or failure.
 */
MORPHIZEN_DLL_SPEC int create_ep_context_nodes(
#if MORPHIZEN_ORT_API_MAJOR < 6
    onnxruntime::Graph & /*ep_context_graph unused to deleted*/,
#endif
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> &eps,
    morphizen::DllSafe<std::vector<onnxruntime::Node *>> *ret_value);

/**
 * @brief Gets the compiled model compatibility information from execution
 * providers.
 *
 * This function extracts compatibility information from the pass context of the
 * execution providers and serializes it to JSON format. The resulting JSON
 * string follows the ModelCompatibilityProto schema defined in
 * model_compatibility.proto.
 *
 * Called by Old ABI (ort-bridge) and New ABI implementations.
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param graph_viewer Pointer to GraphViewer (for ORT) or OrtGraph (for
 * ort-bridge). Can be nullptr if not needed.
 * @return Pointer to the JSON string, or nullptr/empty string if unavailable.
 *         The returned pointer is valid until the next call to this function or
 * EP destruction.
 */
MORPHIZEN_DLL_SPEC const char *get_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *eps,
    const void *graph_viewer);

/**
 * @brief Validates the compiled model compatibility information.
 *
 * This function deserializes the compatibility info JSON, queries backend
 * plugins for their compatibility status, and determines the overall
 * compatibility level. The function follows the same validation logic as
 * defined in the OrtCompiledModelCompatibility enum: UNSUPPORTED >
 * PREFER_RECOMPILATION > SUPPORTED_OPTIMAL > NOT_APPLICABLE.
 *
 * Called by both Old ABI (ort-bridge) and ORT Provider (morphizen).
 *
 * @param compatibility_info JSON string containing compatibility information.
 * @param devices Array of hardware devices for validation
 * (OrtHardwareDevice**). Can be nullptr if device information is not available
 * (e.g., when called from ORT provider).
 * @param num_devices Number of devices in the array. Should be 0 if devices is
 * nullptr.
 * @param model_compatibility Output parameter for the compatibility result:
 *                            0 = EP_NOT_APPLICABLE
 *                            1 = EP_SUPPORTED_OPTIMAL
 *                            2 = EP_SUPPORTED_PREFER_RECOMPILATION
 *                            3 = EP_UNSUPPORTED
 * @return 0 on success, non-zero on failure.
 */
MORPHIZEN_DLL_SPEC int validate_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *eps,
    const char *compatibility_info, const void *const *devices,
    size_t num_devices, int *model_compatibility);

/**
 * @brief Retrieves the MorphiZen version as a 32-bit unsigned integer.
 *
 * This function returns the version number of the MorphiZen ( Execution
 * Provider). in a packed 32-bit format. The version is encoded as follows:
 *
 * - Bits 31–24: Major version
 * - Bits 23–16: Minor version
 * - Bits 15–8 : Patch version
 * - Bits 7–0  : Reserved (set to 0)
 *
 * @return A 32-bit unsigned integer representing the encoded version number.
 */
MORPHIZEN_DLL_SPEC uint32_t morphizen_get_version();

/**
 * @brief Collects profiling data for API and kernel events.
 *
 * Will enable profiling and dump the profile data to the trace file when `-p`
 * option used with `onnxruntime_pref_test.exe`.
 * The trace file can be opend with Chrome trace viewer (chrome://tracing).
 *
 * Called by `MorphiZenExecutionProvider::GetProfiler()`.
 *
 * EventInfo = std::tuple<std::string, // name
 *                            int,         // pid
 *                            int,         // tid
 *                            long long,   // timestamp
 *                            long long    // duration
 *                            >;
 * @param api_events Vector to store API event information.
 * @param kernel_events Vector to store kernel event information.
 */
MORPHIZEN_DLL_SPEC
void profiler_collect(std::vector<EventInfo> &api_events,
                      std::vector<EventInfo> &kernel_events);

/**
 * @brief Retrieves the global API instance for MorphiZen Execution Provider.
 * For multiple DLLs to use the same global_api for initialization.
 *
 * It is defined in morphizen-ort-api.cpp
 *
 * The gloal_api initialized in onnxruntime_morphizen_eo.dll/so.
 * But the gloal_api not initalized in Morphizen tools (eg onnx_grep ,
 * pattern_gen) DLL/SO. so need get the gloal_api from
 * onnxruntime_morphizen_eo.dll/so and set it to Morphizen tools.
 *
 * @return Pointer to the OrtApiForMorphizen instance.
 */
MORPHIZEN_DLL_SPEC const morphizen::OrtApiForMorphizen *get_the_global_api();
MORPHIZEN_DLL_SPEC const morphizen::OrtApiForMorphizen *
get_the_global_api_unsafe();

/**
 * @brief Returns a function pointer that deletes a dynamically allocated
 * std::vector<std::unique_ptr<morphizen::ExecutionProvider>> object.
 *
 * This function returns a generic void* pointer that internally points to a
 * deleter function. The deleter expects a void* pointing to a
 * std::vector<std::unique_ptr<morphizen::ExecutionProvider>> and deletes it.
 *
 * The std::vector<std::unique_ptr<ExecutionProvider>> created in
 * onnxruntime_vitisai_ep.dll cannot be deleted in
 * onnxruntime_providers_vitisai.dll
 *
 *
 * @return void* A function pointer to a deleter function for the specified
 * type.
 */
MORPHIZEN_DLL_SPEC void *morphizen_get_execution_provider_deletor();
}
