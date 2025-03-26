#pragma once
#include <functional>
#include <string>
namespace vaip_core {
// NOTE: must not add VAIP_DLL_SPEC
// this function cannot be shared between DLLs
void add_cleanup_function(const std::string& name,
                          std::function<void()> cleanup_function);
} // namespace vaip_core
