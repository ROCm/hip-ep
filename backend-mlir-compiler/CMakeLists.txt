##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
cmake_minimum_required(VERSION 3.29)

message(STATUS "Configuring MLIR Compiler Backend")

# Protobuf library for metadata serialization (must be built first)
add_subdirectory(proto)

# Level-1 Pass (MorphiZen integration - always build)
add_subdirectory(level-1-pass)

# Custom Op MLIR (MorphiZen runtime - always build)
add_subdirectory(custom-op-mlir)

# E2E Tests (requires ONNX Runtime and GTest)
if(morphizen_ENABLE_UNIT_TEST)
  add_subdirectory(test)
endif()
