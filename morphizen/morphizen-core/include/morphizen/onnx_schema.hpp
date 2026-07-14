/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <string>

// Define ONNX namespace and include schema definitions
#ifdef ONNX_NAMESPACE
#undef ONNX_NAMESPACE
#endif
#define ONNX_NAMESPACE morphizen_onnx
#include <onnx/defs/schema.h>

namespace morphizen {

using OpSchema = ONNX_NAMESPACE::OpSchema;

/**
 * @brief Retrieves the operation schema for a given ONNX operator.
 *
 * This function looks up and returns the schema definition for a specified
 * ONNX operator type within a given domain. The schema contains metadata
 * about the operator including its inputs, outputs, attributes, and type
 * constraints.
 *
 * @param op_type The name of the ONNX operator (e.g., "Conv", "Relu", "Add")
 * @param op_domain The domain namespace for the operator. Defaults to the
 *                  standard ONNX domain (ONNX_NAMESPACE::ONNX_DOMAIN)
 *
 * @return const OpSchema* Pointer to the operator schema if found, nullptr
 * otherwise
 *
 * @note The returned pointer should not be deleted as it points to a statically
 *       managed schema registry entry.
 */
const OpSchema *
GetOpSchema(const std::string &op_type,
            const std::string &op_domain = ONNX_NAMESPACE::ONNX_DOMAIN);

} // namespace morphizen
