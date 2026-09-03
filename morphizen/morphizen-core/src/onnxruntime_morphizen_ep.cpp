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
#include "morphizen/onnxruntime_morphizen_ep.hpp"
#include "./cleanup.hpp"
#include "./logger_adapter.hpp"
#include "morphizen/config_reader.hpp"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include "morphizen/onnxruntime_api.hpp"
#include "morphizen/op_def.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_SUPRRESS_DEPRECATED_WARNG, "1")
DEF_ENV_PARAM(DEBUG_OP_REGISTER, "0")
DEF_ENV_PARAM_2(DEBUG_LOG_LEVEL, "", std::string)

static inline std::string to_lower(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

static void SetGlogMinLogLevel(const std::string &log_level) {
  std::string level_lower = to_lower(log_level);

  if (level_lower == "info") {
    FLAGS_minloglevel = google::GLOG_INFO;
  } else if (level_lower == "warning") {
    FLAGS_minloglevel = google::GLOG_WARNING;
  } else if (level_lower == "error") {
    FLAGS_minloglevel = google::GLOG_ERROR;
  } else if (level_lower == "fatal") {
    FLAGS_minloglevel = google::GLOG_FATAL;
  } else {
    FLAGS_minloglevel = google::GLOG_ERROR;
  }
}

extern "C" {
class OpHolder {
  struct Opdef {
    std::string domain;
    OrtCustomOp *op_ptr; // we want call deleter instead of release()
    void (*deleter)(OrtCustomOp *);
  };

public:
  ~OpHolder() {
    // 1. all_ops_
    for (auto &ops : all_ops_) {
      ops.deleter(ops.op_ptr);
    }
    all_ops_.clear();
  }

  // caller code: add_op("com.xilinx", op.release(), ...);
  void add_op(const char *domain1, OrtCustomOp *op,
              void (*deleter)(OrtCustomOp *)) {
    std::string custom_op_name = op->GetName(op);
    std::string domain = domain1;
    LOG_IF(INFO, ENV_PARAM(DEBUG_OP_REGISTER))
        << " register op(domain: " << domain << ", op_name: " << custom_op_name
        << ") ";
    if (all_ops_.end() !=
        std::find_if(all_ops_.begin(), all_ops_.end(),
                     [domain, custom_op_name](const Opdef &op) {
                       return op.domain == domain &&
                              op.op_ptr->GetName(op.op_ptr) == custom_op_name;
                     })) {
      // Do we allow duplicates? -> Yes, see opdef_main.cpp
      // do nothing
    }
    all_ops_.push_back({domain, op, deleter});

    if (domains.count(domain) == 0) {
      domains.emplace(domain, Ort::CustomOpDomain(domain1));
    }
    auto &domain_obj = domains.at(domain);
    domain_obj.Add(
        op); // This does not take ownership of the op, simply registers it.
    // it is owned by all_ops_.
  }
  OrtCustomOp *get_op(const std::string &domain,
                      const std::string &custom_op_name) const {
    auto it =
        std::find_if(all_ops_.begin(), all_ops_.end(),
                     [domain, custom_op_name](const Opdef &op) {
                       return op.domain == domain &&
                              op.op_ptr->GetName(op.op_ptr) == custom_op_name;
                     });
    if (all_ops_.end() != it) {
      return it->op_ptr;
    }
    return nullptr;
  }
  std::vector<OrtCustomOpDomain *> get_domains() const {
    std::vector<OrtCustomOpDomain *> ret;
    for (auto it = domains.begin(); it != domains.end(); it++) {
      OrtCustomOpDomain *ptr = it->second;
      ret.push_back(ptr);
    }
    return ret;
  }

private:
  std::map<std::string, Ort::CustomOpDomain> domains;
  std::vector<Opdef> all_ops_;
};

static OpHolder &get_global_op_holder() {
  static std::unique_ptr<OpHolder> opholder_instance = nullptr;
  if (!opholder_instance) {
    opholder_instance = std::make_unique<OpHolder>();
#if MORPHIZEN_ORT_API_MAJOR >= 17
    morphizen::add_cleanup_function("cleanup global plugin store",
                                    []() { opholder_instance.reset(); });
#endif
    /*morphizen::StaticPluginRegister(
        "onnxruntime_morphizen_ep", "morphizen_get_registered_custom_op",
        (void*)local_morphizen_get_registered_custom_op);*/
  }
  return *opholder_instance;
}

static void
intialize_op_defs_old(std::vector<OrtCustomOpDomain *> &contrib_domains,
                      std::vector<OrtCustomOpDomain *> &ret_domain) {
  // This function is used to initialize the op_def_map
  typedef morphizen::OpDefInfo *(*morphizen_op_def_info_t)();
  auto op_def_info_ptrs =
      morphizen::Plugin::get_all_symbols("morphizen_op_def_info");
  for (const auto &op_def_into_ptr : op_def_info_ptrs) {
    LOG_IF(ERROR, ENV_PARAM(MORPHIZEN_SUPRRESS_DEPRECATED_WARNG))
        << " morphizen_op_def_info() is depreated, please update plugin \""
        << op_def_into_ptr.first << "\", there is potential memory leak";
    auto op_def_info_func =
        reinterpret_cast<morphizen_op_def_info_t>(op_def_into_ptr.second);
    auto op_def_info = op_def_info_func();
    std::vector<Ort::CustomOpDomain> domains;
    op_def_info->get_domains(domains);
    for (auto &domain : domains) {
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
}
// Collects every OrtCustomOpDomain contributed by "morphizen_register_ops"
// plugin symbols (see morphizen/op_def.hpp's OpRegister) into ret_domain.
// Does a fresh plugin scan on every call -- callers that need this to run
// at most once must cache their own call (see MorphiZenEpFactory's
// custom_op_domains_ member in morphizen-ep-factory.cpp).
void CollectCustomOpDomains(std::vector<OrtCustomOpDomain *> &ret_domain) {
  typedef void (*register_ops_t)(void *, add_op_t);
  auto &op_holder = get_global_op_holder();
  auto add_op = [](void *state, const char *domain, OrtCustomOp *op,
                   void (*deleter)(OrtCustomOp *)) {
    auto op_holder = static_cast<OpHolder *>(state);
    op_holder->add_op(domain, op,
                      deleter); // op ownership has been transfer to op_holder.
  };
  auto register_ops_all =
      morphizen::Plugin::get_all_symbols("morphizen_register_ops");
  LOG_IF(INFO, ENV_PARAM(DEBUG_OP_REGISTER))
      << " register op find " << register_ops_all.size() << " symbols";
  for (const auto &register_ops : register_ops_all) {
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

MORPHIZEN_DLL_SPEC
const ::OrtCustomOp *morphizen_get_registered_custom_op(const char *domain,
                                                        const char *op_name) {
  /* static auto plugin = morphizen::Plugin::get("onnxruntime_morphizen_ep");
   static auto func =
       plugin->get_method<const ::OrtCustomOp*, const char*, const char*>(
           "morphizen_get_registered_custom_op");*/
  return get_global_op_holder().get_op(domain, op_name);
  // return func(domain.c_str(), op_name.c_str());
}
// The interface exported below is used by onnxruntime_providers_morphizen.so
MORPHIZEN_DLL_SPEC
void initialize_onnxruntime_morphizen_ep(
    morphizen::OrtApiForMorphizen *api,
    std::vector<OrtCustomOpDomain *> &ret_domain) {
  morphizen::set_the_global_api(api);
  {
    static std::vector<OrtCustomOpDomain *> contrib_domains;
    // contrib_domains is used to hold the raw pointers, however,there is no way
    // to to delete deconstruct objects, so the memory leak here.
    intialize_op_defs_old(contrib_domains, ret_domain);
  }
  CollectCustomOpDomains(ret_domain);
  morphizen::add_cleanup_function("protobuf shutdown", []() {
#ifdef _WIN32
    google::protobuf::ShutdownProtobufLibrary();
#endif
  });
}

MORPHIZEN_DLL_SPEC
void deinitialize_onnxruntime_morphizen_ep() {
  morphizen::deinitialize_onnxruntime_morphizen_ep();
}

MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *
compile_onnx_model_morphizen_ep_v4(
    const std::string &model_path, const onnxruntime::Graph &graph,
    const onnxruntime::ProviderOptions &options,
    const std::map<std::string, std::string> &session_configs,
    [[maybe_unused]] void *status,
    [[maybe_unused]] void (*func)(void *, int, const char *),
    const OrtLogger *ort_logger) {
  auto set_ort_status = [&](int error_code, const char *error_message) {
    if (func != nullptr) {
      func(status, error_code, error_message);
    }
  };

  std::unique_ptr<morphizen::LoggerAdapter> logger_adapter = nullptr;

  if (ENV_PARAM(DEBUG_LOG_LEVEL).empty()) {
    // Create logger and logger_adapter early to ensure they're available
    // They will be stored in PassContext to prolong their lifetime
    auto logger = std::make_unique<Ort::Logger>(ort_logger);
    logger_adapter =
        std::make_unique<morphizen::LoggerAdapter>(std::move(logger));
  } else {
    SetGlogMinLogLevel(ENV_PARAM(DEBUG_LOG_LEVEL));
    logger_adapter = nullptr;
  }

  return new std::vector<std::unique_ptr<morphizen::ExecutionProvider>>(
      morphizen::compile_onnx_model_3_internal(
          model_path, graph, options, session_configs,
          std::move(logger_adapter), set_ort_status));
}

MORPHIZEN_DLL_SPEC
void profiler_collect(std::vector<EventInfo> &api_events,
                      std::vector<EventInfo> &kernel_events) {
  morphizen::profiler_collect(api_events, kernel_events);
}
MORPHIZEN_DLL_SPEC
void *morphizen_get_execution_provider_deletor() {
  void (*ret)(void *) = nullptr;
  ret = [](void *p) {
    auto ep = reinterpret_cast<
        std::vector<std::unique_ptr<morphizen::ExecutionProvider>> *>(p);
    if (ep != nullptr) {
      delete ep;
    }
  };
  return reinterpret_cast<void *>(ret);
}
}
#if _WIN32
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE /*hinstDLL*/, // handle to DLL module
                    DWORD fdwReason,        // reason for calling function
                    LPVOID lpvReserved)     // reserved
{
  // Perform actions based on the reason for calling.
  switch (fdwReason) {
  case DLL_PROCESS_ATTACH:
    // Initialize once for each new process.
    // Return FALSE to fail DLL load.
    break;

  case DLL_THREAD_ATTACH:
    // Do thread-specific initialization.
    break;

  case DLL_THREAD_DETACH:
    // Do thread-specific cleanup.
    break;

  case DLL_PROCESS_DETACH:
    // it is not safe to call glog() any longer
    // deinitialize_onnxruntime_morphizen_ep might be called again.
    morphizen::MorphizenOrtApi2::cleanup_morphizen();
    deinitialize_onnxruntime_morphizen_ep();
    if (lpvReserved != nullptr) {
      break; // do not do cleanup if process termination scenario
    }
    // Perform any necessary cleanup.
    break;
  }
  return TRUE; // Successful DLL_PROCESS_ATTACH.
}
#endif
