#include <dlfcn.h>
plugin_t open_plugin_dyn(const std::string& name, scope_t scope) {
  auto flag_public = (RTLD_LAZY | RTLD_GLOBAL);
  auto flag_private = (RTLD_LAZY | RTLD_LOCAL);
  return dlopen(name.c_str(),
                scope == scope_t::PUBLIC ? flag_public : flag_private);
}
void* plugin_sym_dyn(plugin_t plugin, const std::string& name) {
  dlerror(); // clean up error;
  return dlsym(plugin, name.c_str());
}
std::string plugin_error_dyn(plugin_t plugin) {
  std::ostringstream str;
  str << "ERROR CODE: " << dlerror();
  return str.str();
}
void close_plugin_dyn(plugin_t plugin) { dlclose(plugin); }
