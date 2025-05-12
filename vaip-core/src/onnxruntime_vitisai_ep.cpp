/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// #include "symbols.hpp"

// typedef void* void_ptr_t;
// #define DECLARE_SYMBOL(sym) extern "C" void_ptr_t sym;
// SYMBOLS(DECLARE_SYMBOL)
// #if defined(_WIN32)
// SYMBOLS_WIN(DECLARE_SYMBOL)
// #endif

// #define DEFINE_SYMBOL(sym) sym,

// static void_ptr_t reserved_symbols[] = {SYMBOLS(DEFINE_SYMBOL)};
#include "morphizen/onnxruntime_vitisai_ep.hpp"
#include "./cleanup.hpp"
#include "./stat.hpp"
#include "./xir_ops/xir_ops_defs.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include "morphizen/op_def.hpp"
#include "morphizen/vaip.hpp"
#include <fstream>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_SUPRRESS_DEPRECATED_WARNG, "1")
DEF_ENV_PARAM(DEBUG_OP_REGISTER, "0")
extern void* BuildInOPs__hook; // prevent xir_opes_defs.obj symbol gc
extern "C" {
class OpHolder {
  struct Opdef {
    std::string domain;
    OrtCustomOp* op_ptr; // we want call deleter instead of release()
    void (*deleter)(OrtCustomOp*);
  };

public:
  ~OpHolder() {
    // 1. all_ops_
    for (auto& ops : all_ops_) {
      ops.deleter(ops.op_ptr);
    }
    all_ops_.clear();
  }

  // caller code: add_op("com.xilinx", op.release(), ...);
  void add_op(const char* domain1, OrtCustomOp* op,
              void (*deleter)(OrtCustomOp*)) {
    std::string custom_op_name = op->GetName(op);
    std::string domain = domain1;
    LOG_IF(INFO, ENV_PARAM(DEBUG_OP_REGISTER))
        << " register op(domain: " << domain << ", op_name: " << custom_op_name
        << ") ";
    if (all_ops_.end() !=
        std::find_if(all_ops_.begin(), all_ops_.end(),
                     [domain, custom_op_name](const Opdef& op) {
                       return op.domain == domain &&
                              op.op_ptr->GetName(op.op_ptr) == custom_op_name;
                     })) {
      // Do we allow duplicates? -> Yes, see opdef_main.cpp
      // do nothing
    }
    vaip_core::get_vitis_ep_custom_ops().insert(domain + ":" + custom_op_name);

    all_ops_.push_back({domain, op, deleter});

    if (domains.count(domain) == 0) {
      domains.emplace(domain, Ort::CustomOpDomain(domain1));
    }
    auto& domain_obj = domains.at(domain);
    domain_obj.Add(
        op); // This does not take ownership of the op, simply registers it.
    // it is owned by all_ops_.
  }
  OrtCustomOp* get_op(const std::string& domain,
                      const std::string& custom_op_name) const {
    auto it =
        std::find_if(all_ops_.begin(), all_ops_.end(),
                     [domain, custom_op_name](const Opdef& op) {
                       return op.domain == domain &&
                              op.op_ptr->GetName(op.op_ptr) == custom_op_name;
                     });
    if (all_ops_.end() != it) {
      return it->op_ptr;
    }
    return nullptr;
  }
  std::vector<OrtCustomOpDomain*> get_domains() const {
    std::vector<OrtCustomOpDomain*> ret;
    for (auto it = domains.begin(); it != domains.end(); it++) {
      OrtCustomOpDomain* ptr = it->second;
      ret.push_back(ptr);
    }
    return ret;
  }

private:
  std::map<std::string, Ort::CustomOpDomain> domains;
  std::vector<Opdef> all_ops_;
};

static OpHolder& get_global_op_holder() {
  static auto instance = std::make_unique<OpHolder>();
  static bool init = false;
  if (init == false) {
    init = true;
#if VAIP_ORT_API_MAJOR >= 17
    vaip_core::add_cleanup_function("cleanup global plugin store",
                                    []() { instance.reset(); });
#endif
    /*vaip_core::StaticPluginRegister(
        "onnxruntime_vitisai_ep", "morphizen_get_registered_custom_op",
        (void*)local_morphizen_get_registered_custom_op);*/
  }
  return *instance;
}

static void
intialize_op_defs_old(std::vector<OrtCustomOpDomain*>& contrib_domains,
                      std::vector<OrtCustomOpDomain*>& ret_domain) {
  // This function is used to initialize the op_def_map
  typedef vaip_core::OpDefInfo* (*vaip_op_def_info_t)();
  auto op_def_info_ptrs =
      vaip_core::Plugin::get_all_symbols("vaip_op_def_info");
  for (const auto& op_def_into_ptr : op_def_info_ptrs) {
    LOG_IF(ERROR, ENV_PARAM(MORPHIZEN_SUPRRESS_DEPRECATED_WARNG))
        << " vaip_op_def_info() is depreated, please update plugin \""
        << op_def_into_ptr.first << "\", there is potential memory leak";
    auto op_def_info_func =
        reinterpret_cast<vaip_op_def_info_t>(op_def_into_ptr.second);
    auto op_def_info = op_def_info_func();
    std::vector<Ort::CustomOpDomain> domains;
    op_def_info->get_domains(domains);
    for (auto& domain : domains) {
      // Memory leak, passing data across dlls
      contrib_domains.push_back(domain.release());
      ret_domain.push_back(*contrib_domains.rbegin());
      CHECK_LE(ret_domain.size(), 100)
          << "ret_domain applied for 100 in onnxruntime";
    }
  }
  std::set<std::string> vitis_ep_custom_ops;
  // todo
  // for (const auto& domain : contrib_domains) {
  //  for (const auto* op : domain->custom_ops_) {
  //    vitis_ep_custom_ops.insert(domain->domain_ + "::" + op->GetName(op));
  //  }
  //}
  vaip_core::get_vitis_ep_custom_ops().insert("::DequantizeLinear");
  vaip_core::get_vitis_ep_custom_ops().insert("::QuantizeLinear");
  vaip_core::get_vitis_ep_custom_ops().insert(
      "com.microsoft::DequantizeLinear");
  vaip_core::get_vitis_ep_custom_ops().insert("com.microsoft::QuantizeLinear");
}
static void intialize_op_defs(std::vector<OrtCustomOpDomain*>& ret_domain) {
  LOG_IF(INFO, BuildInOPs__hook == nullptr)
      << " built in ops empty"; // don't modify it
  // ret_domain.emplace_back(vaip_core::register_xir_ops());

  // This function is used to initialize the op_def_map
  typedef void (*register_ops_t)(void*, add_op_t);
  auto& op_holder = get_global_op_holder();
  auto add_op = [](void* state, const char* domain, OrtCustomOp* op,
                   void (*deleter)(OrtCustomOp*)) {
    auto op_holder = static_cast<OpHolder*>(state);
    op_holder->add_op(domain, op,
                      deleter); // op ownership has been transfer to op_holder.
  };
  auto register_ops_all =
      vaip_core::Plugin::get_all_symbols("morphizen_register_ops");
  LOG_IF(INFO, ENV_PARAM(DEBUG_OP_REGISTER))
      << " register op find " << register_ops_all.size() << " symbols";
  for (const auto& register_ops : register_ops_all) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_OP_REGISTER))
        << " ----------------------" << register_ops.first
        << "-------------------------- ";
    auto register_ops_func =
        reinterpret_cast<register_ops_t>(register_ops.second);
    register_ops_func(&op_holder, add_op);
  }
  auto tmp = op_holder.get_domains();
  ret_domain.insert(ret_domain.end(), tmp.begin(), tmp.end());
}
VAIP_DLL_SPEC
const ::OrtCustomOp* morphizen_get_registered_custom_op(const char* domain,
                                                        const char* op_name) {
  /* static auto plugin = vaip_core::Plugin::get("onnxruntime_vitisai_ep");
   static auto func =
       plugin->get_method<const ::OrtCustomOp*, const char*, const char*>(
           "morphizen_get_registered_custom_op");*/
  return get_global_op_holder().get_op(domain, op_name);
  // return func(domain.c_str(), op_name.c_str());
}
// The interface exported below is used by onnxruntime_providers_vitisai.so
VAIP_DLL_SPEC
void initialize_onnxruntime_vitisai_ep(
    vaip_core::OrtApiForVaip* api,
    std::vector<OrtCustomOpDomain*>& ret_domain) {
  vaip_core::set_the_global_api(api);
  {
    static std::vector<OrtCustomOpDomain*> contrib_domains;
    // contrib_domains is used to hold the raw pointers, however,there is no way
    // to to delete deconstruct objects, so the memory leak here.
    intialize_op_defs_old(contrib_domains, ret_domain);
  }
  intialize_op_defs(ret_domain);
  vaip_core::add_cleanup_function("protobuf shutdown", []() {
    google::protobuf::ShutdownProtobufLibrary();
  });
}

VAIP_DLL_SPEC
void deinitialize_onnxruntime_vitisai_ep() {
  vaip_core::deinitialize_onnxruntime_vitisai_ep();
}

VAIP_DLL_SPEC
std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_options(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options) {
  auto json_config = vaip_core::get_config_json_str(options);
  return new std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>(
      vaip_core::compile_onnx_model_3(model_path, graph, json_config.c_str()));
}

VAIP_DLL_SPEC
std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>*
compile_onnx_model_vitisai_ep_with_error_handling(
    const std::string& model_path, const onnxruntime::Graph& graph,
    const onnxruntime::ProviderOptions& options, [[maybe_unused]] void* status,
    [[maybe_unused]] void (*func)(void*, int, const char*)) {
  auto json_config = vaip_core::get_config_json_str(options);
  return new std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>(
      vaip_core::compile_onnx_model_3(model_path, graph, json_config.c_str()));
}

VAIP_DLL_SPEC
void profiler_collect(std::vector<EventInfo>& api_events,
                      std::vector<EventInfo>& kernel_events) {
  vaip_core::profiler_collect(api_events, kernel_events);
}
}
