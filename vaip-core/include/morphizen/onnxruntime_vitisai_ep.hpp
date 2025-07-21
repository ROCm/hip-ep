/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/vaip_ort.hpp"
/**
 * @file onnxruntime_vitisai_ep.hpp
 * @brief Header file for the Vitis AI Execution Provider integration with ONNX
 * Runtime.
 *
 * This file contains function declarations and macros for initializing,
 * deinitializing, and managing the Vitis AI Execution Provider within the ONNX
 * Runtime framework.
 * It includes functions for setting dynamic options, compiling ONNX models,
 * creating EP context nodes, and retrieving the Vitis AI EP version.
 *
 * @note This header is part of the Vitis AI Execution Provider implementation
 * and is intended for internal use within the ONNX Runtime framework.
 */

extern "C" {
/**
 * @brief Initializes the Vitis AI Execution Provider for ONNX Runtime.
 *
 * The `VitisAI_Provider::Initialize()` is invoked during
 * `session_options.AppendExecutionProvider_VitisAI(provider_options)`,
 * prior to session creation.
 * Called right after loading the shared library.
 *
 * Features:
 * - Initialize the global API in onnxruntime_vitisai_ep dll/so.
 * - Register custom operations
 *
 * @param api Pointer to the OrtApiForVaip instance.
 * @param ret_domain Reference to a vector of OrtCustomOpDomain pointers to
 * store custom operation domains.
 */
VAIP_DLL_SPEC
void initialize_onnxruntime_vitisai_ep(
    vaip_core::OrtApiForVaip* api, std::vector<OrtCustomOpDomain*>& ret_domain);

/**
 * @brief Deinitializes the Vitis AI Execution Provider for ONNX Runtime.
 *
 * The `VitisAI_Provider::Shutdown()` is invoked during
 * `ProviderLibrary::Unload()`,prior to unloading the shared library. It cleans
 * up any resources allocated during initialization.
 */
VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep();

/**
 * @brief Called when InferenceSession::Run() started.
 * Enable user to set proformance mode for every session run.
 * Related to ORT #19521
 *
 * Called by `VitisAIExecutionProvider::OnRunStart()`.
 *
 * @param eps Vector of unique pointers to ExecutionProvider instances.
 * @param state Pointer to the runtime state.
 * @param get_config_entry Function pointer to retrieve run_option entries.
 * @return Status code indicating success or failure.
 */
VAIP_DLL_SPEC int vitisai_ep_on_run_start(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const void* state,
    vaip_core::DllSafe<std::string> (*get_config_entry)(
        const void* state, const char* entry_name));

/**
 * @brief Set DynamicOptions for the Vitis AI Execution Provider.
 *
 * Called when InferenceSession::SetEpDynamicOptions() is called.
 * Related to ORT #22282
 *
 * Called by `VitisAIExecutionProvider::SetEpDynamicOptions()`.
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
VAIP_DLL_SPEC int vitisai_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const char* const* keys, const char* const* values, size_t kv_len);

/**
 * @brief Compiles an ONNX model using the Vitis AI Execution Provider with
 * error handling.
 *
 * Called when VitisAIExecutionProvider::GetCapabilities() is called.
 * Throw an ONNXRuntime Error (ORT_THROW(STATUS)) if compile ONNX model error.
 *
 * Calleb by `VitisAIExecutionProvider::GetCapability()`.
 *
 * @param model_path Path to the ONNX model file.
 * @param graph Reference to the ONNX Runtime graph.
 * @param options Provider options for compilation.
 * @param status Pointer to a status object for error handling.
 * @param func Callback function for error handling.
 * @return Pointer to a vector of unique pointers to ExecutionProvider
 * instances.
 */
VAIP_DLL_SPEC
std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_error_handling(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options, [[maybe_unused]] void* status,
    void (*func)(void*, int, const char*));

/**
 * @brief Compiles an ONNX model using the Vitis AI Execution Provider with
 * specified options.
 *
 * If compile_onnx_model_vitisai_ep_with_error_handing not implements, will
 * call this function. Not throw ONNXRuntime Error when compile ONNX model
 * error.
 *
 * Called by `VitisAIExecutionProvider::GetCapability()`.
 *
 * @param model_path Path to the ONNX model file.
 * @param graph Reference to the ONNX Runtime graph.
 * @param options Provider options for compilation.
 * @return Pointer to a vector of unique pointers to ExecutionProvider
 * instances.
 */
VAIP_DLL_SPEC std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_options(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options);

/**
 * @brief Creates EPContxt Nodes for the Vitis AI Execution Provider.
 *
 * Called after compile onnx model when ep.context_enbale is set to 1.
 * The method is called after both `GetCapability()` and `Compile()`.
 *
 * Called by `VitisAIExecutionProvider::GetEpContextNodes()`.
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
VAIP_DLL_SPEC int create_ep_context_nodes(
#if VAIP_ORT_API_MAJOR < 6
    onnxruntime::Graph& /*ep_context_graph unused to deleted*/,
#endif
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    vaip_core::DllSafe<std::vector<onnxruntime::Node*>>* ret_value);

/**
 * @brief Retrieves the VAIP version as a 32-bit unsigned integer.
 *
 * This function returns the version number of the VAIP (Vitis AI Execution
 * Provider). in a packed 32-bit format. The version is encoded as follows:
 *
 * - Bits 31–24: Major version
 * - Bits 23–16: Minor version
 * - Bits 15–8 : Patch version
 * - Bits 7–0  : Reserved (set to 0)
 *
 * @return A 32-bit unsigned integer representing the encoded version number.
 */
VAIP_DLL_SPEC uint32_t vaip_get_version();

/**
 * @brief Collects profiling data for API and kernel events.
 *
 * Will enable profiling and dump the profile data to the trace file when `-p`
 * option used with `onnxruntime_pref_test.exe`.
 * The trace file can be opend with Chrome trace viewer (chrome://tracing).
 *
 * Called by `VitisAIExecutionProvider::GetProfiler()`.
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
VAIP_DLL_SPEC
void profiler_collect(std::vector<EventInfo>& api_events,
                      std::vector<EventInfo>& kernel_events);

/**
 * @brief Retrieves the global API instance for Vitis AI Execution Provider.
 * For multiple DLLs to use the same global_api for initialization.
 *
 * It is defined in vaip_ort_api.cpp
 *
 * The gloal_api initialized in onnxruntime_vitisai_eo.dll/so.
 * But the gloal_api not initalized in Morphizen tools (eg onnx_grep ,
 * pattern_gen) DLL/SO. so need get the gloal_api from
 * onnxruntime_vitisai_eo.dll/so and set it to Morphizen tools.
 *
 * @return Pointer to the OrtApiForVaip instance.
 */
VAIP_DLL_SPEC const vaip_core::OrtApiForVaip* get_the_global_api();
VAIP_DLL_SPEC const vaip_core::OrtApiForVaip* get_the_global_api_unsafe();

/**
 * @brief Returns a function pointer that deletes a dynamically allocated
 * std::vector<std::unique_ptr<vaip_core::ExecutionProvider>> object.
 *
 * This function returns a generic void* pointer that internally points to a
 * deleter function. The deleter expects a void* pointing to a
 * std::vector<std::unique_ptr<vaip_core::ExecutionProvider>> and deletes it.
 *
 * The std::vector<std::unique_ptr<ExecutionProvider>> created in
 * onnxruntime_vitisai_ep.dll cannot be deleted in
 * onnxruntime_providers_vitisai.dll
 *
 *
 * @return void* A function pointer to a deleter function for the specified
 * type.
 */
VAIP_DLL_SPEC void* vaip_get_execution_provider_deletor();
}
