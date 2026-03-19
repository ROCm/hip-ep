/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

/// Schema-driven JSON <-> FlatBuffers native struct helpers.
///
/// Both helpers go through FlatBuffers binary as an intermediate:
///   toJson:   NativeT -> Pack() -> binary -> GenerateText() -> JSON string
///   fromJson: JSON string -> ParseJson() -> binary -> UnPackTo() -> NativeT
///
/// Template parameter: NativeT (e.g. CompilationOptionsT, HipModelMetaInfoT)
/// FlatT (the binary accessor class) is derived via NativeT::TableType so
/// callers never need to mention it.

#include "flatbuffers/idl.h"

#include <string>

namespace mlir {
namespace hip {

template <typename NativeT>
bool toJson(const NativeT &native, const char *schema, std::string &result,
            std::string &error) {
  using FlatT = typename NativeT::TableType;

  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(FlatT::Pack(fbb, &native));

  flatbuffers::Parser parser;
  // strict_json=true outputs quoted field names (standard JSON).
  // Without this, GenerateText produces FlatBuffers relaxed JSON with
  // unquoted keys, which standard JSON parsers reject.
  parser.opts.strict_json = true;
  if (!parser.Parse(schema)) {
    error = "Schema parse error: " + std::string(parser.error_);
    return false;
  }

  flatbuffers::GenerateText(parser, fbb.GetBufferPointer(), &result);
  return true;
}

template <typename NativeT>
bool fromJson(const std::string &json, const char *schema, NativeT &result,
              std::string &error) {
  flatbuffers::Parser parser;
  if (!parser.Parse(schema)) {
    error = "Schema parse error: " + std::string(parser.error_);
    return false;
  }
  if (!parser.ParseJson(json.c_str())) {
    error = "JSON parse error: " + std::string(parser.error_);
    return false;
  }
  flatbuffers::GetRoot<typename NativeT::TableType>(
      parser.builder_.GetBufferPointer())
      ->UnPackTo(&result);
  return true;
}

} // namespace hip
} // namespace mlir
