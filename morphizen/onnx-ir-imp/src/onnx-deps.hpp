/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// Include the base ONNX protobuf definitions first
#include <onnx/onnx_pb.h>
#ifdef ONNX_NAMESPACE
#undef ONNX_NAMESPACE // for some unknown reasons, we have to redefine it.
#endif
#define ONNX_NAMESPACE morphizen_onnx

// Include additional ONNX headers that shape inference depends on
#include <onnx/common/constants.h>
#include <onnx/defs/schema.h>
#include <onnx/defs/tensor_proto_util.h>
// Now include shape inference implementation
#include <onnx/shape_inference/implementation.h>
