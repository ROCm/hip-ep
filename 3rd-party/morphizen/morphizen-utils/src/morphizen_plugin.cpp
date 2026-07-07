/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <sstream>
// clang-format off
// we must include glog before morphizen headers
#include <glog/logging.h>
// clang-format on
#include "morphizen-foundation/env_config.hpp"
#include "morphizen-utils/morphizen_plugin.hpp"
#include "morphizen-utils/weak_refs.hpp"
// include local headers
#include "morphizen-utils/cleanup.hpp"
DEF_ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN) >= n)
namespace morphizen {

struct Plugin_Func_Set {
  std::pair<plugin_t, bool> (*open_plugin)(const std::string& name,
                                           scope_t scope);
  void* (*plugin_sym)(plugin_t plugin, const std::string& name);
  void (*close_plugin)(plugin_t plugin);
};

static std::pair<plugin_t, bool> open_plugin_static(const std::string& name,
                                                    scope_t scope);
static void* plugin_sym_static(plugin_t plugin, const std::string& symbol);
static void close_plugin_static(plugin_t plugin);

Plugin_Func_Set g_static_plugin_func_set = {
    open_plugin_static, plugin_sym_static, close_plugin_static};

Plugin_Func_Set g_dynamic_plugin_func_set = {morphizen::open_plugin_dyn,
                                             morphizen::plugin_sym_dyn,
                                             morphizen::close_plugin_dyn};

static Plugin_Func_Set* g_static_plugin_func_set_ptr =
    &g_static_plugin_func_set;
static Plugin_Func_Set* g_dynamic_plugin_func_set_ptr =
    &g_dynamic_plugin_func_set;

std::string Plugin::guess_name(const char* name) {
  std::string name_str(name);
#ifdef _WIN32
  // Only add .dll if not already present
  if (name_str.size() >= 4 && name_str.substr(name_str.size() - 4) == ".dll") {
    return name_str;
  }
  return name_str + ".dll";
#else
  // Only add lib/.so if not already present
  if (name_str.size() >= 3 && name_str.substr(name_str.size() - 3) == ".so") {
    return name_str;
  }
  // Absolute or relative path (contains '/') — treat as a real filesystem path,
  // append ".so" without the "lib" prefix. Otherwise dlopen("/tmp/foo") would
  // become dlopen("lib/tmp/foo.so") which is a different (non-existent) file.
  if (name_str.find('/') != std::string::npos) {
    return name_str + ".so";
  }
  if (name_str.size() >= 3 && name_str.substr(0, 3) == "lib") {
    return name_str;
  }
  return "lib" + name_str + ".so";
#endif
}

// Factory method - returns nullptr if loading fails
std::unique_ptr<Plugin> Plugin::create(const char* name) {
  std::string name_str(name);
  std::string so_name = guess_name(name);

  void* plugin = nullptr;
  bool owned = false;
  Plugin_Func_Set* func_set = nullptr;

  auto try_load = [&](const std::string& tag, Plugin_Func_Set* fs) -> bool {
    MY_LOG(1) << "trying load from " << tag;
    std::tie(plugin, owned) = fs->open_plugin(so_name, scope_t::PUBLIC);
    if (plugin) {
      func_set = fs;
      MY_LOG(1) << " load plugin from " << tag << " name=" << name
                << " so_name=" << so_name;
      return true;
    }
    return false;
  };

  // Try static first, then dynamic
  if (!try_load("static", g_static_plugin_func_set_ptr)) {
    if (!try_load("dynamic", g_dynamic_plugin_func_set_ptr)) {
      // Both failed - return nullptr
      MY_LOG(1) << "Failed to load plugin: " << name << " (tried: " << so_name
                << ")";
      return nullptr;
    }
  }

  // Success - create Plugin object using PrivateTag
  return std::make_unique<Plugin>(PrivateTag{}, name, plugin, func_set, owned);
}

// Constructor - only callable via factory method (requires PrivateTag)
Plugin::Plugin(PrivateTag, const char* name, void* plugin,
               Plugin_Func_Set* func_set, bool owned)
    : name_{name}, so_name_{guess_name(name)}, func_set_{func_set},
      plugin_{plugin}, owned_{owned} {
  // Invariant: plugin_ is never nullptr (enforced by factory method)
}

Plugin::~Plugin() {
  if (func_set_ && owned_) {
    MY_LOG(1) << "  -- close plugin: " << name_ << " " << so_name_
              << " this=" << (void*)this;
    func_set_->close_plugin((plugin_t)plugin_);
  } else {
    MY_LOG(1) << "  -- do not close plugin because it is owned handled: "
              << name_ << " " << so_name_ << " this=" << (void*)this;
  }
}
// TODO: for now all Plugin::get store loaded plugin in a global
// store. it is not a good solution, it is possible to put all plugin
// into pass_context?
static std::unordered_map<std::string, std::shared_ptr<Plugin>>&
get_global_plugin_store() {
  static std::unordered_map<std::string, std::shared_ptr<Plugin>> store_;
  static bool init = false;
  if (init == false) {
    init = true;
    add_cleanup_function("cleanup global plugin store",
                         []() { store_.clear(); });
  }
  return store_;
}

Plugin* Plugin::get(const std::string& plugin_name) {
  auto& store_ = get_global_plugin_store();
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] =
        morphizen::utils::WeakStore<std::string, Plugin>::create(
            plugin_name, plugin_name.c_str());
  }
  it = store_.find(plugin_name);
  CHECK(it != store_.end())
      << "cannot load plugin. plugin_name=" << plugin_name;
  if (it->second && it->second->plugin_ == nullptr) {
    MY_LOG(1) << "cannot load plugin: " << plugin_name;
    return nullptr;
  }
  return it->second.get();
}
void* Plugin::my_plugin_sym(void* handle, const char* name) const {
  if (!func_set_)
    return nullptr;
  return func_set_->plugin_sym((plugin_t)handle, name);
}

static std::unordered_map<std::string, std::unordered_map<std::string, void*>>&
get_store() {
  static std::unordered_map<std::string, std::unordered_map<std::string, void*>>
      store_;
  return store_;
}

std::vector<std::pair<std::string, void*>>
Plugin::get_all_symbols(const char* name) {
  auto& store = get_store();
  auto ret = std::vector<std::pair<std::string, void*>>();
  for (auto it = store.begin(); it != store.end(); ++it) {
    for (auto it_sym = it->second.begin(); it_sym != it->second.end();
         ++it_sym) {
      if (it_sym->first == name) {
        ret.emplace_back(it->first, it_sym->second);
      }
    }
  }
  return ret;
}

static std::pair<plugin_t, bool> open_plugin_static(const std::string& name,
                                                    scope_t /*scope*/) {
  auto& store = get_store();
  auto it = store.find(name);
  if (it == store.end()) {
    MY_LOG(1) << " open_plugin_static cannot find plugin: " << name;
    MY_LOG(1) << " valid plugins are: ";
    for (auto& x : store) {
      MY_LOG(1) << "  plugin=" << x.first << " " << x.second.size()
                << " symbols";
    }
    return {nullptr, false};
  }
  MY_LOG(1) << " found plugin: " << name;
  return {reinterpret_cast<plugin_t>(new std::string(name)), true};
}

static void* plugin_sym_static(plugin_t plugin, const std::string& symbol) {
  auto& name = *reinterpret_cast<std::string*>(plugin);
  auto& store = get_store();
  auto it_lib = store.find(name);
  if (it_lib == store.end()) {
    MY_LOG(1) << "cannot find lib:" << name;
    MY_LOG(1) << "usually this should not happened, did you forget to register "
                 "the plugin?";
    MY_LOG(1) << "valid libs are: ";
    for (auto& x : store) {
      MY_LOG(1) << "  libs=" << x.first << "\n";
    }
    return nullptr;
  }
  auto it_sym = it_lib->second.find(symbol);
  if (it_sym == it_lib->second.end()) {
    MY_LOG(1) << "cannot find symbol " << symbol << " in " << name;
    MY_LOG(1) << "valid symbols are: ";
    for (auto& x : it_lib->second) {
      MY_LOG(1) << "  symbols=" << x.first << "\n";
    }
    return nullptr;
  }
  return it_sym->second;
}

static void close_plugin_static(plugin_t plugin) {
  delete reinterpret_cast<std::string*>(plugin);
}

std::string plugin_error_static(plugin_t /*plugin*/) { return "N/A"; }

void register_plugin_static(const std::string& name, const std::string& symbol,
                            void* addr) {
  MY_LOG(1) << "register: " << name << " " << symbol << " " << addr;
  get_store()[name][symbol] = addr;
}

void unregister_plugin_static(const std::string& name,
                              const std::string& symbol, void* addr) {
  MY_LOG(1) << "unregister: " << name << " " << symbol << " " << addr;
  auto it = get_store().find(name);
  if (it == get_store().end()) {
    MY_LOG(1) << "cannot find lib: " << name;
    return;
  }
  auto it_sym = it->second.find(symbol);
  if (it_sym == it->second.end()) {
    MY_LOG(1) << "cannot find symbol: " << symbol;
    return;
  }
  it->second.erase(it_sym);
  return;
}

StaticPluginRegister::StaticPluginRegister(const char* name, const char* symbol,
                                           void* addr)
    : name_(name), symbol_(symbol), addr_(addr) {
#if _WIN32
  std::string lib_name = std::string(name) + ".dll";
#else
  std::string lib_name = std::string("lib") + name + ".so";
#endif
  register_plugin_static(lib_name, symbol, addr);
  std::ostringstream str;
  str << "unload " << symbol << " @ " << name;
  add_cleanup_function(str.str(), [lib_name, symbol, addr]() {
    register_plugin_static(lib_name, symbol, addr);
  });
}
StaticPluginRegister::~StaticPluginRegister() {}

extern "C" void morphizen_register_static_plugin(const char* name,
                                                 const char* symbol,
                                                 void* addr) {
  register_plugin_static(name, symbol, addr);
}

void StaticPluginRegister::sync_static_plugin_into_module(
    const char* module_name) {
  // this function try to sync with onnxruntime_vitisai_ep.dll
  auto morphizen_register_static_plugin_func =
      Plugin::get(module_name)
          ->get_method<void, const char*, const char*, void*>(
              "morphizen_register_static_plugin");
  CHECK(morphizen_register_static_plugin_func)
      << "cannot find morphizen_register_static_plugin in module: "
      << module_name;
  auto& store = get_store();
  for (auto& x : store) {
    for (auto& y : x.second) {
      morphizen_register_static_plugin_func(x.first.c_str(), y.first.c_str(),
                                            y.second);
    }
  }
}

} // namespace morphizen
