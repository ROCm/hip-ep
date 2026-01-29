/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <dlfcn.h>
#include <sstream>
#include <string>
#include <utility>

namespace morphizen {
using plugin_t = void*;
enum class scope_t { PUBLIC, PRIVATE };

std::pair<plugin_t, bool> open_plugin_dyn(const std::string& name,
                                          scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  return {dlopen(name.c_str(),
                 scope == scope_t::PUBLIC ? flag_public : flag_private),
          true};
}
void* plugin_sym_dyn(plugin_t plugin, const std::string& name) {
  dlerror(); // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error_dyn(plugin_t /*plugin*/) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin_dyn(plugin_t plugin) { dlclose(plugin); }
} // namespace morphizen
