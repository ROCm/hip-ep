/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./vitisai-ep.hpp"
#include "./ir-converter.hpp"
#include "./ort-graph-wrapper.hpp"
#include "./vaip-deps.hpp"
#include "glog/logging.h"
#include "morphizen-utils/morphizen-utils.hpp"
#include "morphizen/onnxruntime_vitisai_ep.hpp"
#include "morphizen/vaip-ort-api-ext.hpp"
#include "morphizen/vaip.hpp"
#include <set>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP) >= n)
namespace morphizen {

/**
 * @brief Get supported nodes from graph viewer based on execution provider
 * meta definition
 *
 * @param ep The execution provider containing meta definition with
 * supported nodes
 * @param graph_viewer The ORT graph wrapper to iterate through nodes
 * @return std::vector<const OrtNode*> List of supported nodes
 */
static std::vector<const OrtNode*>
get_supported_nodes(const vaip_core::ExecutionProvider& ep,
                    OrtGraphWrapper& graph_viewer);

VitisAIEP::VitisAIEP(ApiPtrs apis, const std::string& name,
                     const OrtKeyValuePairs* const* ep_metadata,
                     const OrtSessionOptions& session_options,
                     const OrtLogger& logger)
    : ApiPtrs(apis), name_{name}, logger_{logger} {
  // Initialize the execution provider.
  auto status = ort_api.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("ExampleEp has been created with name " + name_).c_str(), ORT_FILE,
      __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;

  // Can get configurations from the session options.
  // Note: should not store a direct reference to the session options object
  // as its lifespan is not guaranteed. EP should copy any configurations or
  // settings it needs.  (void)session_options;

  // Convert ep_metadata to provider options
  update_provider_options_from_ep_metadata(ep_metadata);

  // Update provider options from session configuration
  update_provider_options_from_session_config(session_options);

  std::string ep_context_enable;
  throw_if_error(GetSessionConfigEntryOrDefault(
      session_options, "ep.context_enable", "0", ep_context_enable));
  enable_ep_context_ = ep_context_enable == "1";

  OrtEp::ort_version_supported =
      ORT_API_VERSION; // set to the ORT version we were compiled with.
  OrtEp::GetName = GetNameImpl;
  OrtEp::GetCapability = GetCapabilityImpl;
  OrtEp::Compile = CompileImpl;
  OrtEp::ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  ir_converter = morphizen::IRConverter::to_onnx_model;
}
VitisAIEP::~VitisAIEP() {}

void VitisAIEP::update_provider_options_from_session_config(
    const OrtSessionOptions& session_options) {
  MY_LOG(2) << "Updating provider options from session configuration";

  {
    OrtKeyValuePairs* kvps = nullptr;
    throw_if_error(
        ort_api.GetSessionOptionsConfigEntries(&session_options, &kvps));
    std::unique_ptr<OrtKeyValuePairs, void (*)(OrtKeyValuePairs*)>
        config_entries(kvps, ort_api.ReleaseKeyValuePairs);
    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t num_keys = 0;
    // Get keys and values from the config entries
    ort_api.GetKeyValuePairs(config_entries.get(), &keys, &values, &num_keys);

    MY_LOG(2) << "Converting " << num_keys
              << " provider options from session config entries";

    const std::string vitisai_ep_prefix = "ep.vitisaiexecutionprovider.";
    const std::string ort_session_prefix = "ort_session_config.";
    for (size_t i = 0; i < num_keys; ++i) {
      if (keys[i] != nullptr && values[i] != nullptr) {
        std::string key_str(keys[i]);
        std::string value_str(values[i]);
        if (key_str.rfind(vitisai_ep_prefix, 0) == 0) {
          std::string option_name = key_str.substr(vitisai_ep_prefix.length());
          provider_options_[option_name] = value_str;
        } else {
          provider_options_[ort_session_prefix + key_str] = value_str;
        }
      } else {
        MY_LOG(2) << "Skipping null key or value at index " << i
                  << ": key=" << (keys[i] ? keys[i] : "null")
                  << ", value=" << (values[i] ? values[i] : "null");
      }
    }

    MY_LOG(2) << "Successfully processed " << num_keys
              << " session config entries. Total provider options: "
              << provider_options_.size();
  }
  // // List of session config keys that should be copied to provider options
  // // These control EP context behavior and optimization settings
  // // Using string literals based on ORT session option keys
  // std::vector<std::string> session_config_keys = {
  //     "ep.context_enable",         // kOrtSessionOptionEpContextEnable
  //     "ep.context_file_path",      // kOrtSessionOptionEpContextFilePath
  //     "ep.context_embed_mode",     // kOrtSessionOptionEpContextEmbedMode
  //     "ep.share_ep_contexts",      // kOrtSessionOptionShareEpContexts
  //     "ep.stop_share_ep_contexts", // kOrtSessionOptionStopShareEpContexts
  //     "ep_dynamic_options.workload_type", // kOrtEpDynamicOptionsWorkloadType
  //     "disable_model_compiles" // kOrtSessionOptionsDisableModelCompiles
  //     // Note: kOrtSessionOptionEpContextNodeNamePrefix is TODO
  //     // TODO : provider options set by end user.
  //     // "ep.vitisaiexecutionprovider.enable_cache_file_io_in_mem"
  //     // "ep.vitisaiexecutionprovider.target"
  //     // ...
  // };

  // int has_config = 0;
  // size_t config_entries_found = 0;

  // for (const auto& key : session_config_keys) {
  //   auto status = ort_api.HasSessionConfigEntry(&session_options,
  //   key.c_str(),
  //                                               &has_config);
  //   if (status != nullptr) {
  //     MY_LOG(1) << "Failed to check session config entry for key: " << key
  //               << " - " << ort_api.GetErrorMessage(status);
  //     ort_api.ReleaseStatus(status);
  //     continue;
  //   }
  //   if (has_config) {
  //     // Get the config value - first call to get required size
  //     size_t config_value_size = 0;
  //     status = ort_api.GetSessionConfigEntry(&session_options, key.c_str(),
  //                                            nullptr, &config_value_size);
  //     if (status != nullptr) {
  //       MY_LOG(1) << "Failed to get session config size for key: " << key
  //                 << " - " << ort_api.GetErrorMessage(status);
  //       ort_api.ReleaseStatus(status);
  //       continue;
  //     }

  //     if (config_value_size > 0) {
  //       // Allocate buffer and get the actual value
  //       std::string config_value(config_value_size, '\0');
  //       status =
  //           ort_api.GetSessionConfigEntry(&session_options, key.c_str(),
  //                                         &config_value[0],
  //                                         &config_value_size);
  //       if (status == nullptr) {
  //         // Remove the null terminator if present
  //         if (config_value_size > 0 &&
  //             config_value[config_value_size - 1] == '\0') {
  //           config_value.resize(config_value_size - 1);
  //         }
  //         provider_options_["ort_session_config." + key] = config_value;
  //         MY_LOG(3) << "Session config: " << key << " = " << config_value;
  //         config_entries_found++;
  //       } else {
  //         MY_LOG(1) << "Failed to get session config value for key: " << key
  //                   << " - " << ort_api.GetErrorMessage(status);
  //         ort_api.ReleaseStatus(status);
  //       }
  //     } else {
  //       MY_LOG(3) << "Session config key " << key << " has empty value";
  //     }
  //   }
  //   has_config = 0; // Reset for next iteration
  // }
  // MY_LOG(2) << "Updated " << config_entries_found
  //           << " provider options from session configuration";
}

void VitisAIEP::update_provider_options_from_ep_metadata(
    const OrtKeyValuePairs* const* ep_metadata) {
  if (ep_metadata && *ep_metadata) {
    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t num_keys = 0;

    // Get keys and values from the metadata
    ort_api.GetKeyValuePairs(*ep_metadata, &keys, &values, &num_keys);

    MY_LOG(2) << "Converting " << num_keys
              << " provider options from ep_metadata";

    size_t successful_options = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (keys[i] != nullptr && values[i] != nullptr) {
        provider_options_[keys[i]] = values[i];
        MY_LOG(3) << "Provider option: " << keys[i] << " = " << values[i];
        ++successful_options;
      } else {
        MY_LOG(1) << "Skipping invalid key or value in ep_metadata at index "
                  << i << ": " << (keys[i] ? keys[i] : "null") << " = "
                  << (values[i] ? values[i] : "null");
      }
    }

    // Note: ORT API manages the memory for keys/values arrays, don't delete
    // them
    MY_LOG(2) << "Successfully loaded " << successful_options << " of "
              << num_keys << " provider options from metadata. Total options: "
              << provider_options_.size();
  } else {
    MY_LOG(2) << "No ep_metadata provided, using default provider options";
  }
}

const char* ORT_API_CALL
VitisAIEP::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const VitisAIEP*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ORT_API_CALL VitisAIEP::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {

  VitisAIEP* self = static_cast<VitisAIEP*>(this_ptr);

  if (self->ir_converter == nullptr) {
    MY_LOG(1) << " no available IR converter.";
    return nullptr;
  }
  // Use the OrtGraphWrapper class instead of manual API calls
  auto graph_wrapper = morphizen::OrtGraphWrapper(*self, *graph);
  self->GetCapability(graph_wrapper, *graph_support_info);
  return nullptr;
}

OrtStatus* ORT_API_CALL VitisAIEP::CompileImpl(
    OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
    size_t count, OrtNodeComputeInfo** node_compute_infos,
    OrtNode** ep_context_nodes) noexcept {
  VitisAIEP* self = static_cast<VitisAIEP*>(this_ptr);
  return self->Compile(graphs, fused_nodes, count, node_compute_infos,
                       ep_context_nodes);
}
void ORT_API_CALL VitisAIEP::ReleaseNodeComputeInfosImpl(
    OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  VitisAIEP* self = static_cast<VitisAIEP*>(this_ptr);
  self->ReleaseNodeComputeInfos(node_compute_infos, num_node_compute_infos);
}

OrtStatus* VitisAIEP::GetCapability(OrtGraphWrapper& graph_viewer,
                                    OrtEpGraphSupportInfo& graph_support_info) {
  auto is_subgraph = graph_viewer.is_subgraph();
  // TODO: ORT does not support Graph_IsSubgraph API yet, so we cannot check if
  // the graph is a subgraph.
  // ort_api.Graph_IsSubgraph(graph, &is_subgraph);
  if (is_subgraph) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return nullptr;
  }
  // setup API environment
  const char* backend_ir = "onnx-ir-imp";
  auto with_new_api = setup_global_vaip_ort_api(backend_ir);
  //
  auto ir_model = ir_converter(*this, graph_viewer.get());
  auto& graph = VAIP_ORT_API(model_main_graph)(*ir_model);
  auto model_path = VAIP_ORT_API(get_model_path)(graph);
  OrtStatus* status = nullptr;
  execution_providers_ = std::make_unique<my_ep_t>(
      compile_onnx_model_vitisai_ep_with_error_handling(
          model_path.u8string(), graph, provider_options_, (void*)&status,
          [](void* status, int code, const char* error_message) {
            OrtStatus** ort_status = static_cast<OrtStatus**>(status);
            *ort_status =
                Ort::GetApi().CreateStatus((OrtErrorCode)code, error_message);
          }));

  if (!execution_providers_) {
    MY_LOG(1) << "Failed to compile ONNX model to Vitis AI EP.";
    return nullptr;
  }
  // iterator over the execution providers and set the graph support info
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;
  node_fusion_options.drop_constant_initializers = true;
  auto supported_node_groups = std::vector<std::vector<const OrtNode*>>();
  for (auto& ep : **execution_providers_) {
    CHECK(ep.get() != nullptr);
    supported_node_groups.emplace_back(get_supported_nodes(*ep, graph_viewer));
  }
  for (auto& supported_nodes : supported_node_groups) {
    auto status1 = ep_api.EpGraphSupportInfo_AddNodesToFuse(
        &graph_support_info, supported_nodes.data(), supported_nodes.size(),
        &node_fusion_options);
    if (status1 != nullptr) {
      MY_LOG(1) << "Failed to add nodes to fuse in GetCapability: "
                << ort_api.GetErrorMessage(status1);
      return status1;
    }
  }
  return nullptr;
}

static void update_argument_indice(
    const google::protobuf::RepeatedPtrField<std::string>& meta_def_args,
    google::protobuf::RepeatedField<int32_t>* argument_indices,
    const std::vector<const OrtValueInfo*>& node_value_infos) {
  CHECK_EQ(meta_def_args.size(), node_value_infos.size());
  argument_indices->Clear();
  argument_indices->Reserve(meta_def_args.size());
  auto size = meta_def_args.size();
  for (size_t i = 0; i < size; ++i) {
    auto name = Ort::ConstValueInfo(node_value_infos[i]).Name();
    auto it = std::find(meta_def_args.begin(), meta_def_args.end(), name);
    CHECK(it != meta_def_args.end()) << " cannot find name: " << name;
    auto index = std::distance(meta_def_args.begin(), it);
    argument_indices->Add((int32_t)index);
  }
}
void VitisAIEP::update_input_output_argument_indice(
    vaip_core::ExecutionProvider& ep, const OrtNode* node) {
  auto ep_ext = dynamic_cast<vaip_core::ExecutionProviderConcrete*>(&ep);
  CHECK(ep_ext != nullptr) << "Execution provider does not support "
                              "ExecutionProviderExt interface.";
  auto& meta_def = ep_ext->get_meta_def();
  std::vector<const OrtValueInfo*> inputs = {};
  size_t num_of_inputs = 0;
  throw_if_error(ort_api.Node_GetNumInputs(node, &num_of_inputs));
  inputs.resize(num_of_inputs);
  throw_if_error(ort_api.Node_GetInputs(node, inputs.data(), num_of_inputs));
  std::vector<const OrtValueInfo*> outputs = {};
  size_t num_of_outputs = 0;
  throw_if_error(ort_api.Node_GetNumOutputs(node, &num_of_outputs));
  outputs.resize(num_of_outputs);
  throw_if_error(ort_api.Node_GetOutputs(node, outputs.data(), num_of_outputs));
  update_argument_indice(meta_def.inputs(),
                         meta_def.mutable_input_argument_indice(), inputs);
  update_argument_indice(meta_def.outputs(),
                         meta_def.mutable_output_argument_indice(), outputs);
}
OrtStatus* VitisAIEP::Compile(const OrtGraph** graphs,
                              const OrtNode** fused_nodes, size_t count,
                              OrtNodeComputeInfo** node_compute_infos,
                              OrtNode** ep_context_nodes) {
  CHECK(execution_providers_ != nullptr)
      << "Execution providers are not initialized.";
  CHECK_EQ((*execution_providers_)->size(), count)
      << "Number of execution providers does not match the count provided.";
  MY_LOG(1) << "VitisAIEP::Compile called with " << count
            << " execution providers.";
  throw_if_error(CreateEpContextNodes(fused_nodes, ep_context_nodes, count));
  for (auto index = 0u; index < count; ++index) {
    // here we assume that the execution_providers_ are in the same order as the
    // nodes in the graph. this is the tight coupling between the
    // execution providers and the graph nodes.
    auto& ep_ptr = (**execution_providers_)[index];
    CHECK(ep_ptr.get() != nullptr)
        << "Execution provider at index " << index << " is null.";
    auto status =
        CompileSubgraph(*ep_ptr, graphs[index], fused_nodes[index],
                        node_compute_infos[index], ep_context_nodes[index]);
    update_input_output_argument_indice(*ep_ptr, fused_nodes[index]);
    if (status != nullptr) {
      MY_LOG(1) << "Failed to compile subgraph for execution provider at index "
                << index << ": " << ort_api.GetErrorMessage(status);
      return status;
    }
  }
  return nullptr;
}
OrtStatus*
VitisAIEP::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_infos,
                                   size_t num_node_compute_infos) {
  for (auto i = 0u; i < num_node_compute_infos; ++i) {
    delete static_cast<VitisAIEP_ComputeInfo*>(node_compute_infos[i]);
  }
  return nullptr;
}

OrtStatus* VitisAIEP::CompileSubgraph(const vaip_core::ExecutionProvider& ep,
                                      const OrtGraph* graph,
                                      const OrtNode* fused_node,
                                      OrtNodeComputeInfo*& node_compute_info,
                                      OrtNode*& ep_context_node) {
  (void)graph;
  (void)fused_node;
  (void)ep_context_node; // TODO implement EP context.
  try {
    MY_LOG(2) << "VitisAIEP::CompileSubgraph called for execution provider";

    // Validate input parameters
    if (!graph) {
      MY_LOG(1) << "CompileSubgraph: graph parameter is null";
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Graph parameter cannot be null");
    }

    if (!fused_node) {
      MY_LOG(1) << "CompileSubgraph: fused_node parameter is null";
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "Fused node parameter cannot be null");
    }
    auto vitisai_node_compute_info = new VitisAIEP_ComputeInfo();
    node_compute_info = vitisai_node_compute_info;
    vitisai_node_compute_info->ort_version_supported = ORT_API_VERSION;
    vitisai_node_compute_info->vitisai_ep =
        const_cast<vaip_core::ExecutionProvider*>(&ep);
    vitisai_node_compute_info->CreateState =
        [](OrtNodeComputeInfo* this_ptr, OrtNodeComputeContext* compute_context,
           void** compute_state) -> OrtStatus* {
      (void)compute_context;
      auto self = reinterpret_cast<VitisAIEP_ComputeInfo*>(this_ptr);
      auto* p = self->vitisai_ep->compile().release();
      *compute_state = p;
      return nullptr;
    };
    vitisai_node_compute_info->ReleaseState = [](OrtNodeComputeInfo* this_ptr,
                                                 void* compute_state) -> void {
      auto self = reinterpret_cast<VitisAIEP_ComputeInfo*>(this_ptr);
      (void)self;
      if (compute_state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(compute_state);
      }
    };
    vitisai_node_compute_info->Compute =
        [](OrtNodeComputeInfo* this_ptr, void* compute_state,
           OrtKernelContext* kernel_context) -> OrtStatus* {
      auto self = reinterpret_cast<VitisAIEP_ComputeInfo*>(this_ptr);
      (void)self;
      reinterpret_cast<vaip_core::CustomOp*>(compute_state)
          ->Compute(&Ort::GetApi(), kernel_context);
      return nullptr;
    };
    MY_LOG(2) << "CompileSubgraph: Successfully compiled subgraph";
    return nullptr;

  } catch (const std::exception& e) {
    MY_LOG(1) << "CompileSubgraph: Exception caught: " << e.what();
    return Ort::GetApi().CreateStatus(
        ORT_FAIL,
        ("CompileSubgraph failed with exception: " + std::string(e.what()))
            .c_str());
  } catch (...) {
    MY_LOG(1) << "CompileSubgraph: Unknown exception caught";
    return Ort::GetApi().CreateStatus(
        ORT_FAIL, "CompileSubgraph failed with unknown exception");
  }
}

static bool vaip_and_ort_have_same_names(std::set<std::string>& vaip_names,
                                         std::set<std::string>& ort_names) {
  auto join = [](const auto& set) {
    std::ostringstream oss;
    for (const auto& name : set) {
      if (!oss.str().empty()) {
        oss << ", ";
      }
      oss << name;
    }
    return oss.str();
  };
  // Check if both sets have the same names
  auto ret =
      std::equal(vaip_names.begin(), vaip_names.end(), ort_names.begin());
  if (!ret) {
    MY_LOG(1) << "VitisAI EP and ORT have different names: "
              << "VitisAI EP names: " << join(vaip_names)
              << ", ORT names: " << join(ort_names);
  } else {
    MY_LOG(1) << "VitisAI EP and ORT have the same names: "
              << "VitisAI EP names: " << join(vaip_names)
              << ", ORT names: " << join(ort_names);
  }
  return ret;
}
OrtNode* VitisAIEP::convert_vaip_node_to_ort_node(
    const OrtModelEditorApi* model_editor_api,
    const onnxruntime::Node* vaip_node, //
    const OrtNode* fused_node) const {
  if (!vaip_node) {
    return nullptr;
  }
  auto input_node_args = VAIP_ORT_API(node_get_inputs_unsafe)(*vaip_node);
  auto output_node_args =
      VAIP_ORT_API(node_get_output_node_args_unsafe)(*vaip_node);
  std::vector<std::string> input_names;
  std::set<std::string> vaip_input_names;
  for (const auto& arg : *input_node_args) {
    auto name = VAIP_ORT_API(node_arg_get_name_unsafe)(*arg.node_arg);
    input_names.push_back(name);
    vaip_input_names.insert(name);
  }
  std::vector<std::string> output_names;
  std::set<std::string> vaip_output_names;
  output_names.reserve(output_node_args->size());
  for (const auto& arg : *output_node_args) {
    auto name = VAIP_ORT_API(node_arg_get_name_unsafe)(*arg);
    output_names.push_back(name);
    vaip_output_names.insert(name);
  }
  auto& vaip_attributes = VAIP_ORT_API(node_get_attributes)(
      const_cast<onnxruntime::Node&>(*vaip_node));
  auto attributes = std::unique_ptr<std::vector<OrtOpAttr*>,
                                    void (*)(std::vector<OrtOpAttr*>*)>(
      new std::vector<OrtOpAttr*>(), [](std::vector<OrtOpAttr*>* ptr) {
        for (auto& attr : *ptr) {
          Ort::GetApi().ReleaseOpAttr(attr);
        }
        delete ptr;
      });
  auto keys = VAIP_ORT_API(node_attributes_get_keys)(vaip_attributes);
  attributes->reserve(keys->size());
  for (const auto& name : *keys) {
    auto proto1 = VAIP_ORT_API(node_attributes_get)(vaip_attributes, name);
    CHECK(proto1 != nullptr) << "Attribute proto is null for name: " << name;
    auto& proto = *proto1;
    auto type = VAIP_ORT_API(attr_proto_get_type)(proto);
    switch (type) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType::
        AttributeProto_AttributeType_INT: {
      int64_t value = VAIP_ORT_API(attr_proto_get_int)(proto);
      OrtOpAttr* ort_attr = nullptr;
      throw_if_error(ort_api.CreateOpAttr(
          name.c_str(), &value, 1, OrtOpAttrType::ORT_OP_ATTR_INT, &ort_attr));
      attributes->push_back(ort_attr);
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType::
        AttributeProto_AttributeType_STRING: {
      std::string value = VAIP_ORT_API(attr_proto_get_string)(proto);
      OrtOpAttr* ort_attr = nullptr;
      MY_LOG(1) << " add " << name << " size = " << value.size();
      throw_if_error(ort_api.CreateOpAttr(
          name.c_str(), value.data(), static_cast<int>(value.size()),
          OrtOpAttrType::ORT_OP_ATTR_STRING, &ort_attr));
      attributes->push_back(ort_attr);
      break;
    }
    default:
      LOG(FATAL) << "Unsupported attribute type: " << type;
      continue;
    }
  }

  // Create the ORT node using the model editor API
  auto op_type = VAIP_ORT_API(node_op_type)(*vaip_node);
  auto domain = VAIP_ORT_API(node_op_domain)(*vaip_node);

  // NOTE: we must use the fused_node name, see graph_partitioner.cc:898
  const char* ort_node_name = nullptr;
  std::vector<const OrtValueInfo*> fused_node_inputs;
  size_t num_of_inputs = 0;
  std::vector<const OrtValueInfo*> fused_node_outputs;
  size_t num_of_outputs = 0;
  throw_if_error(ort_api.Node_GetName(fused_node, &ort_node_name));
  throw_if_error(ort_api.Node_GetNumInputs(fused_node, &num_of_inputs));
  throw_if_error(ort_api.Node_GetNumOutputs(fused_node, &num_of_outputs));
  fused_node_inputs.resize(num_of_inputs);
  fused_node_outputs.resize(num_of_outputs);
  throw_if_error(ort_api.Node_GetInputs(fused_node, fused_node_inputs.data(),
                                        fused_node_inputs.size()));
  throw_if_error(ort_api.Node_GetOutputs(fused_node, fused_node_outputs.data(),
                                         fused_node_outputs.size()));
  std::vector<std::string> input_names_2;
  std::vector<std::string> output_names_2;
  std::transform(fused_node_inputs.begin(), fused_node_inputs.end(),
                 std::back_inserter(input_names_2),
                 [](const OrtValueInfo* input) {
                   return Ort::ConstValueInfo(input).Name();
                 });
  std::transform(fused_node_outputs.begin(), fused_node_outputs.end(),
                 std::back_inserter(output_names_2),
                 [](const OrtValueInfo* output) {
                   return Ort::ConstValueInfo(output).Name();
                 });
  std::set<std::string> ort_input_names(input_names_2.begin(),
                                        input_names_2.end());
  std::set<std::string> ort_output_names(output_names_2.begin(),
                                         output_names_2.end());
  CHECK(vaip_and_ort_have_same_names(vaip_input_names, ort_input_names))
      << "VitisAI EP and ORT have different input names: ";
  CHECK(vaip_and_ort_have_same_names(vaip_output_names, ort_output_names))
      << "VitisAI EP and ORT have different output names: ";
  std::vector<const char*> input_names_c;
  std::vector<const char*> output_names_c;
  input_names_c.reserve(input_names_2.size());
  output_names_c.reserve(output_names_2.size());
  std::transform(input_names_2.begin(), input_names_2.end(),
                 std::back_inserter(input_names_c),
                 [](const std::string& name) { return name.c_str(); });
  std::transform(output_names_2.begin(), output_names_2.end(),
                 std::back_inserter(output_names_c),
                 [](const std::string& name) { return name.c_str(); });
  OrtNode* ort_added_node = nullptr;
  auto status = model_editor_api->CreateNode(
      "EPContext", "com.microsoft", ort_node_name, input_names_c.data(),
      input_names_c.size(), output_names_c.data(), output_names_c.size(),
      attributes->data(), attributes->size(), &ort_added_node);

  if (status != nullptr) {
    MY_LOG(1) << "Failed to create ORT node: "
              << ort_api.GetErrorMessage(status);
    return nullptr;
  }

  return ort_added_node;
}
OrtStatus* VitisAIEP::CreateEpContextNodes(const OrtNode** fused_nodes,
                                           OrtNode** ep_context_nodes,
                                           size_t count) const {
  if (!enable_ep_context_) {
    return nullptr;
  }
  vaip_core::DllSafe<std::vector<onnxruntime::Node*>> vaip_ep_context_nodes;
  auto error_code =
      create_ep_context_nodes(**execution_providers_, &vaip_ep_context_nodes);

  if (error_code != 0) {
    MY_LOG(1) << "Failed to create EP context nodes, error code: "
              << error_code;
  }
  CHECK_EQ(count, vaip_ep_context_nodes->size())
      << "Count of EP context nodes does not match the expected count";

  const OrtModelEditorApi* model_editor_api = ort_api.GetModelEditorApi();
  if (!model_editor_api) {
    LOG(ERROR) << "Model editor API is not available";
    return ort_api.CreateStatus(ORT_FAIL, "Model editor API is not available");
  }
  for (size_t i = 0; i < count && i < vaip_ep_context_nodes->size(); ++i) {
    ep_context_nodes[i] = convert_vaip_node_to_ort_node(
        model_editor_api, (*vaip_ep_context_nodes)[i], fused_nodes[i]);
  }
  return nullptr;
}
/**
 * @brief Get supported nodes from graph viewer based on execution provider
 * meta definition
 *
 * @param ep The execution provider containing meta definition with supported
 * nodes
 * @param graph_viewer The ORT graph wrapper to iterate through nodes
 * @return std::vector<const OrtNode*> List of supported nodes
 */
static std::vector<const OrtNode*>
get_supported_nodes(const vaip_core::ExecutionProvider& ep,
                    OrtGraphWrapper& graph_viewer) {

  std::vector<const OrtNode*> supported_nodes;

  try {
    auto meta_def_outputs =
        std::vector<std::string>(std::move(*ep.get_meta_def_nodes().get()));
    std::unordered_set<std::string> ep_supported_outputs(
        meta_def_outputs.begin(), meta_def_outputs.end());
    if (ep_supported_outputs.empty()) {
      MY_LOG(2) << "ExecutionProvider meta_def has no supported nodes";
      return supported_nodes;
    }
    MY_LOG(2) << "ExecutionProvider supports " << ep_supported_outputs.size()
              << " node outputs";
    // Iterate through all nodes in the graph
    auto nodes = graph_viewer.nodes();
    for (const OrtNode* node : nodes) {
      if (!node)
        continue;

      try {
        // Get node outputs using ORT API
        std::vector<const OrtValueInfo*> outputs;
        size_t num_outputs = 0;
        graph_viewer.throw_if_error(
            graph_viewer.ort_api.Node_GetNumOutputs(node, &num_outputs));
        outputs.resize(num_outputs);
        graph_viewer.throw_if_error(graph_viewer.ort_api.Node_GetOutputs(
            node, outputs.data(), num_outputs));

        // Check if any output of this node is supported by the EP
        auto outputs_span = gsl::span(outputs);
        bool node_supported = false;

        for (const OrtValueInfo* output : outputs_span) {
          if (!output)
            continue;

          // Get the output name
          auto output_value_info = Ort::ConstValueInfo(output);
          std::string output_name = output_value_info.Name();
          size_t node_id = 0u;
          graph_viewer.throw_if_error(
              graph_viewer.ort_api.Node_GetId(node, &node_id));
          // Check if this output is in the EP's supported nodes
          if (ep_supported_outputs.count(output_name) > 0) {
            MY_LOG(3) << "Node with output '" << output_name
                      << "' is supported by EP, ID=" << node_id;
            node_supported = true;
            break;
          }
        }

        if (node_supported) {
          supported_nodes.push_back(node);
        }

      } catch (const std::exception& e) {
        MY_LOG(3) << "Exception while processing node: " << e.what();
        // Continue with next node
      }
    }

    MY_LOG(1) << "Found " << supported_nodes.size()
              << " supported nodes for ExecutionProvider";
    CHECK_EQ(supported_nodes.size(), ep_supported_outputs.size())
        << "Mismatch between supported nodes count and EP meta definition "
           "size";
  } catch (const std::exception& e) {
    MY_LOG(1) << "Exception in get_supported_nodes: " << e.what();
    // Return empty vector on error
  }

  return supported_nodes;
}

OrtStatus* VitisAIEP::GetSessionConfigEntryOrDefault(
    const OrtSessionOptions& session_options, const char* config_key,
    const std::string& default_val,
    /*out*/ std::string& config_val) {
  int has_config = 0;
  throw_if_error(
      ort_api.HasSessionConfigEntry(&session_options, config_key, &has_config));
  if (has_config != 1) {
    config_val = default_val;
    return nullptr;
  }

  size_t size = 0;
  throw_if_error(ort_api.GetSessionConfigEntry(&session_options, config_key,
                                               nullptr, &size));

  config_val.resize(size);
  throw_if_error(ort_api.GetSessionConfigEntry(&session_options, config_key,
                                               config_val.data(), &size));
  config_val.resize(size - 1); // remove the terminating '\0'
  return nullptr;
}

} // namespace morphizen
