/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <cstdlib>
#include <iostream>
#include <memory>
#include <morphizen/export.h>
#include <string>
#include <unordered_map>

namespace morphizen {
using plugin_t = void*;
enum class scope_t { PUBLIC, PRIVATE };
struct Plugin_Func_Set; // Forward declaration
std::pair<plugin_t, bool> open_plugin_dyn(const std::string& name,
                                          scope_t scope);
void* plugin_sym_dyn(plugin_t plugin, const std::string& name);
std::string plugin_error_dyn(plugin_t plugin);
void close_plugin_dyn(plugin_t plugin);
void register_plugin_static(const std::string& name, const std::string& symbol,
                            void* addr);
class StaticPluginRegister {
public:
  StaticPluginRegister(const char* name, const char* symbol, void* addr);
  ~StaticPluginRegister();
  /**
   * @brief Synchronizes a static plugin into the specified module.
   *
   * This function copies all registered static plugin in the current module
   * into a module identified by its name. It ensures that the plugin is
   * properly synchronized and available for use within the specified module.
   *
   * @param module_name The name of the module into which the static plugin will
   * be synchronized.
   */
  static void sync_static_plugin_into_module(const char* module_name);

private:
  const char* name_;
  const char* symbol_;
  void* addr_;
};

extern "C" void morphizen_register_static_plugin(const char* name,
                                                 const char* symbol,
                                                 void* addr);

struct Plugin {

  // Factory method - returns nullptr if plugin fails to load
  static std::unique_ptr<Plugin> create(const char* name);

private:
  struct PrivateTag {}; // See docs/technical/privatetag-factory-pattern.md

public:
  // Constructor requires PrivateTag - only callable from factory method
  Plugin(PrivateTag, const char* name, void* plugin, Plugin_Func_Set* func_set,
         bool owned);
  ~Plugin();
  template <typename R, typename... Args>
  static R invoke(const char* plugin_name, const char* symbol, Args&&... args) {
    auto plugin = Plugin::get(plugin_name);
    if (plugin == nullptr) {
      std::cerr << "no such plugin: " << plugin_name << std::endl;
      std::abort();
    }
    return plugin->invoke<R, Args...>(symbol, std::forward<Args>(args)...);
  }
  template <typename R, typename... Args>
  R invoke(const char* name, Args&&... args) {
    auto method = get_method<R, Args&&...>(name);
    if (method == nullptr) {
      std::cerr << "no such function: " << name << "; " //
                << "libname " << name_ << " "           //
                << "so_name " << so_name_ << " "        //
                << std::endl;
      std::abort();
    }
    return method(std::forward<Args>(args)...);
  }

  MORPHIZEN_DLL_SPEC static Plugin* get(const std::string& name);

  bool has_method(const char* name) const {
    return my_plugin_sym(plugin_, name) != nullptr;
  };

  // Check if the plugin DLL was successfully loaded
  bool is_loaded() const { return plugin_ != nullptr; }

  // Get the actual DLL path that was attempted (useful for error reporting)
  const std::string& get_so_name() const { return so_name_; }
  /**
   * @brief Retrieves all symbols associated with the given name.
   *
   * This function returns a vector of pairs, where each pair consists of a
   * string representing the plugin name and a void pointer to the symbol's
   * associated data or function.
   *
   * @param name The name of the symbol group to retrieve.
   * @return A vector of pairs containing plugin names and their corresponding
   * pointers.
   */
  static std::vector<std::pair<std::string, void*>>
  get_all_symbols(const char* name);
  template <typename R, typename... Args> using method_t = R (*)(Args...);

  template <typename R, typename... Args>
  method_t<R, Args...> get_method(const char* name) const {
    auto sym = my_plugin_sym(plugin_, name);
    if (sym == nullptr) {
      return nullptr;
    }
    typedef R (*fun_type_t)(Args...);
    fun_type_t f = reinterpret_cast<fun_type_t>(sym);
    return f;
  }

private:
  std::string name_;
  std::string so_name_;
  struct Plugin_Func_Set* func_set_;
  void* plugin_;
  bool owned_;

private:
  static std::string guess_name(const char* name);

  MORPHIZEN_DLL_SPEC void* my_plugin_sym(void*, const char*) const;
};

template <typename T, typename... Args> class WithPlugin {
public:
  static std::unique_ptr<T> create(const std::string& plugin_name,
                                   Args&&... args) {
    auto plugin = Plugin::get(plugin_name);
    if (plugin == nullptr) {
      std::cerr << "no such plugin: " << plugin_name << std::endl;
      std::abort();
    }
    auto ret = plugin->invoke<T*>(T::entry_point, std::forward<Args>(args)...);
    return std::unique_ptr<T>(ret);
  }
};
} // namespace morphizen
