/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "core/graph/contrib_ops/contrib_defs.h"

#include "onnx-deps.hpp"
#include <cmath>

// From onnxruntime/core/graph/constants.h
namespace onnxruntime {
constexpr const char *kMSDomain = "com.microsoft";
} // namespace onnxruntime

// Suppress a warning: global initializer calls a non-constexpr function
// 'symbol' which is from ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in
// debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif

namespace onnxruntime {

namespace contrib {
using namespace ONNX_NAMESPACE;
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
void RegisterContribSchemas() {
  auto &domainToVersionRangeInstance =
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domainToVersionRangeInstance.Map().find(onnxruntime::kMSDomain) ==
      domainToVersionRangeInstance.Map().end()) {
    // External shared providers may have already added kMSDomain
    domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSDomain, 1,
                                                    1);
  }
  static morphizen_onnx::OpSchemaRegistry::OpSchemaRegisterOnce
      op_schema_register_onceEPContext14(
          morphizen_onnx::OpSchema(
              "EPContext",
              "C:\\Users\\chunye.wang\\workspace\\cp-"
              "dev\\source\\MorphiZen\\onnx-ir-imp\\src\\core\\graph\\contrib_"
              "ops\\contrib_defs.cc",
              29)
              .SetDomain(kMSDomain)
              .SinceVersion(1)
              .SetDoc("Onnx node container for EP context.")
              .Attr("main_context",
                    "Usually each single EPContext associate with a graph "
                    "partition."
                    "But for some case like QNN, it has single EPContext "
                    "contains all "
                    "partitions."
                    "In that case, the node with ep_cache_context should set "
                    "main_context=1. Other nodes set main_context=0 and skip "
                    "ep_cache_context."
                    "The path is relative to this Onnx file. Default is 1.",
                    AttributeProto::INT, static_cast<int64_t>(1))
              .Attr("ep_cache_context",
                    "payload of the execution provider context if "
                    "embed_mode=1, or "
                    "path to the context file if embed_mode=0.",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("embed_mode",
                    "1: indicate ep_cache_context is the context content. 0: "
                    "indicate "
                    "ep_cache_context is the file path to the context content."
                    "The path is relative to this Onnx file. Default is 1.",
                    AttributeProto::INT, static_cast<int64_t>(1))
              .Attr("ep_sdk_version",
                    "(Optional) SDK version used to convert the model.",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("onnx_model_filename",
                    "(Optional) Filename of the original ONNX model.",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("hardware_architecture",
                    "(Optional) Hardware architecture.", AttributeProto::STRING,
                    OPTIONAL_VALUE)
              .Attr("partition_name", "(Optional) partitioned graph name.",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("source",
                    "(Optional) the source used to generate the engine/context "
                    "cache "
                    "file. Ort EP or native SDK tool chain",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("notes", "(Optional) Some notes for the model",
                    AttributeProto::STRING, OPTIONAL_VALUE)
              .Attr("max_size",
                    "max size in the context. Usage depend on the EP.",
                    AttributeProto::INT, static_cast<int64_t>(0))
              .AllowUncheckedAttributes()
              .Input(0, "inputs", "List of tensors for inputs", "T",
                     OpSchema::Variadic, false, 1, OpSchema::NonDifferentiable)
              .Output(0, "outputs",
                      "One or more outputs, list of tensors for outputs", "T",
                      OpSchema::Variadic, false, 1, OpSchema::NonDifferentiable)
              .TypeConstraint("T",
                              {"tensor(bool)", "tensor(int8)", "tensor(int16)",
                               "tensor(int32)", "tensor(int64)",
                               "tensor(uint8)", "tensor(uint16)",
                               "tensor(uint32)", "tensor(uint64)",
                               "tensor(float16)", "tensor(float)",
                               "tensor(double)", "tensor(bfloat16)"},
                              "Constrain input and output types."));
}
} // namespace contrib
} // namespace onnxruntime
