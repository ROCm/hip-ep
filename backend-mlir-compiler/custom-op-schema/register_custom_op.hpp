/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <morphizen/morphizen.hpp>
#include <morphizen/op_def.hpp>

#include <memory>

// Registers `OpClass` (an hipep::SchemaOnlyCustomOpBase<OpClass> subclass,
// see schema_only_custom_op_base.hpp) under `domain` with a single
// declaration at the bottom of an ops/*_schema.cpp file:
//
//   HIPEP_REGISTER_CUSTOM_OP(QMoESchemaOp, "com.amd")
//
// Each invocation expands to a self-contained OpRegister subclass +
// StaticPluginRegister, matching VAIP's per-op registration pattern
// (vaip_op_def_gqa/src/register_gqa.cpp) but without hand-writing that
// boilerplate at every call site. There is no shared registry/singleton
// here: morphizen-core discovers every registration independently through
// Plugin::get_all_symbols("morphizen_register_ops"), which already
// aggregates across translation units/libraries, so nothing in this macro
// depends on static-initialization order across *_schema.cpp files.
//
// The StaticPluginRegister name must be process-wide unique: morphizen's
// plugin store is keyed by (name, symbol), and a second registration under
// the same name silently overwrites the first instead of erroring (see
// morphizen-utils/src/morphizen_plugin.cpp register_plugin_static). __FILE__
// is unique per translation unit by construction (one op per ops/*.cpp
// file), so it is used here instead of a hand-picked tag that two ops could
// accidentally collide on.
#define HIPEP_REGISTER_CUSTOM_OP(OpClass, domain)                              \
  namespace {                                                                  \
  class OpClass##_Register : public OpRegister {                               \
  public:                                                                      \
    OpClass##_Register(void *state, add_op_t add_op)                           \
        : OpRegister(state, add_op) {}                                         \
    int register_ops() override {                                              \
      AddOp(domain, std::make_unique<OpClass>());                              \
      return 0;                                                                \
    }                                                                          \
  };                                                                           \
  int OpClass##_register_ops(void *state, add_op_t add_op) {                   \
    return OpClass##_Register(state, add_op).register_ops();                   \
  }                                                                            \
  static ::morphizen::StaticPluginRegister                                     \
      OpClass##_g_register(__FILE__, "morphizen_register_ops",                 \
                           (void *)&OpClass##_register_ops);                   \
  } // namespace
