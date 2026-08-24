/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Schema-only registration for ORT custom ops that only need to let ORT
// load/Resolve a graph containing them (see hipep::SchemaOnlyCustomOp in
// schema_only_custom_op.hpp for why one class covers every op here).
//
// Add a new op by adding one (domain, name) entry to kSchemaOnlyOps below --
// no new file, no CMakeLists.txt edit. If an op ever needs real type/shape
// validation, give it its own real ONNX-to-HIP conversion pattern instead
// of a schema-only entry here.
#include "./schema_only_custom_op.hpp"
#include <morphizen/morphizen.hpp>
#include <morphizen/op_def.hpp>

#include <memory>

namespace {

struct SchemaOnlyOpSpec {
  const char *domain;
  const char *name;
};

constexpr SchemaOnlyOpSpec kSchemaOnlyOps[] = {
    {"com.amd", "QMoE"},
    // {"domain", "OpName"},  // add new schema-only ops here
};

int RegisterSchemaOnlyOps(void *state, add_op_t add_op) {
  struct Register : OpRegister {
    Register(void *s, add_op_t a) : OpRegister(s, a) {}
    int register_ops() override {
      for (const auto &spec : kSchemaOnlyOps) {
        AddOp(spec.domain,
              std::make_unique<hipep::SchemaOnlyCustomOp>(spec.name));
      }
      return 0;
    }
  } reg(state, add_op);
  return reg.register_ops();
}

static ::morphizen::StaticPluginRegister
    g_register("hipep-custom-op-schema", "morphizen_register_ops",
               (void *)&RegisterSchemaOnlyOps);

} // namespace
