/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {

::ONNX_NAMESPACE::OpSchema&
RegisterRangeOpSchema(::ONNX_NAMESPACE::OpSchema&& op_schema);

} // namespace contrib
} // namespace onnxruntime
