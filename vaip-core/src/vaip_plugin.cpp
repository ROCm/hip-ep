/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include <sstream>
// clang-format off
// we must include glog before morphizen headers
#include <glog/logging.h>
// clang-format on
#include "morphizen/env_config.hpp"
#include "morphizen/vaip_plugin.hpp"
#include "morphizen/weak.hpp"
// include local headers
#include "./cleanup.hpp"
DEF_ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_PLUGIN) >= n)
namespace vaip_core {

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

Plugin_Func_Set g_dynamic_plugin_func_set = {vaip_core::open_plugin_dyn,
                                             vaip_core::plugin_sym_dyn,
                                             vaip_core::close_plugin_dyn};

static Plugin_Func_Set* g_static_plugin_func_set_ptr =
    &g_static_plugin_func_set;
static Plugin_Func_Set* g_dynamic_plugin_func_set_ptr =
    &g_dynamic_plugin_func_set;

std::string Plugin::guess_name(const char* name) {
#ifdef _WIN32
  return std::string("") + name + ".dll";
#else
  return std::string("lib") + name + ".so";
#endif
}

Plugin::Plugin(const char* name)
    : name_{name}, so_name_{guess_name(name)}, func_set_{nullptr},
      plugin_{nullptr}, owned_{false} {

  auto load = [this](const std::string& tag, Plugin_Func_Set* func_set) {
    MY_LOG(1) << "trying load from " << tag;
    std::tie(plugin_, owned_) =
        func_set->open_plugin(so_name_, scope_t::PUBLIC);
    if (plugin_) {
      func_set_ = func_set;
      MY_LOG(1) << " load plugin from " << tag << " name=" << name_
                << " so_name=" << so_name_;
    }
  };
  load("static", g_static_plugin_func_set_ptr);
  if (!plugin_) {
    load("dynamic", g_dynamic_plugin_func_set_ptr);
  }
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

VAIP_DLL_SPEC
Plugin* Plugin::get(const std::string& plugin_name) {
  auto& store_ = get_global_plugin_store();
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] = morphizen::WeakStore<std::string, Plugin>::create(
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
                                                    scope_t scope) {
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

std::string plugin_error_static(plugin_t plugin) { return "N/A"; }

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

} // namespace vaip_core
