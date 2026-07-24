/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

namespace morphizen {

// Forward declaration
class Graph;

/**
 * @brief In-memory representation of ONNX NodeArg
 *
 * This class provides operations on ONNX node argument structure.
 * Currently serves as a placeholder for node argument operations.
 */
class NodeArg {
public:
  // Empty class for now - placeholder for future node argument operations

  /**
   * @brief Get constant data as tensor (placeholder implementation)
   * @param graph The parent graph
   * @return void* Pointer to tensor data (currently returns nullptr)
   */
  void *get_const_data_as_tensor(const Graph &graph) const;
};

} // namespace morphizen
