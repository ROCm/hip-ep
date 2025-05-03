/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "onnxruntime_api.hpp"
#include <morphizen/vaip_plugin.hpp>
#include <set>
#include <vector>

namespace vaip_core {
struct OpDefInfo {
  void (*get_domains)(std::vector<Ort::CustomOpDomain>& domains);
};

template <typename T> struct ProcessorOpDefInfo {
  static OpDefInfo* op_fef_info() { return &info; }
  static OpDefInfo info;
};

template <typename T> OpDefInfo ProcessorOpDefInfo<T>::info = {&T::process};
void set_vitis_ep_custom_ops(const std::set<std::string>&);
} // namespace vaip_core

#ifndef _WIN32
#  include <vaip/export.h>
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    extern "C" VAIP_PASS_ENTRY vaip_core::OpDefInfo* vaip_op_def_info() {      \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    extern "C" {                                                               \
    void* /* a hook var*/ id##__hook = nullptr;                                \
    }
#else
#  define DEFINE_VAIP_OPDEF(cls, id)                                           \
    static ::vaip_core::OpDefInfo* vaip_op_def_info() {                        \
      return ProcessorOpDefInfo<cls>::op_fef_info();                           \
    }                                                                          \
    namespace {                                                                \
    static ::vaip_core::StaticPluginRegister                                   \
        __register(OUTPUT_NAME, "vaip_op_def_info", (void*)&vaip_op_def_info); \
    }

#endif

typedef void (*add_op_t)(void*, const char*, OrtCustomOp*,
                         void (*)(OrtCustomOp*));
struct OpRegister {
public:
  OpRegister(void* state, add_op_t add_op) : state_(state), add_op_(add_op) {}
  virtual ~OpRegister() = default;
  virtual int register_ops() = 0;

protected:
  template <typename T>
  void AddOp(const std::string& domain, std::unique_ptr<T> op) {
    add_op_(state_, domain.c_str(), op.release(),
            [](OrtCustomOp* p) { delete static_cast<T*>(p); });
  }
  void* state_;
  add_op_t add_op_;
};
