##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

#=============================================================================
# Dialect unit tests (GPU-free)
#=============================================================================
# Native C++ tests that link the HIP dialect library and exercise dialect-level
# C++ APIs (attributes, type predicates) that LIT cannot reach directly.
# Registered with CTest.

message(STATUS "Configuring dialect unit tests")

add_executable(test-hip-memory-space
  Dialect/test_memory_space.cpp
)

target_compile_features(test-hip-memory-space PRIVATE cxx_std_17)

# Generated dialect headers (HipDialect.h pulls in the enum + attr .inc files).
add_dependencies(test-hip-memory-space
  HipDialectIncGen
  HipTypesIncGen
  HipAttributesIncGen
  HipDpsOpInterfaceIncGen
  HipOpStateInterfaceIncGen
  HipOpsIncGen
)

target_link_libraries(test-hip-memory-space PRIVATE
  HipDialectIR
  MLIRIR
  MLIRAsmParser
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  ${llvm_libs}
)

add_test(NAME HipMemorySpaceUnitTest COMMAND test-hip-memory-space)

set_target_properties(test-hip-memory-space PROPERTIES FOLDER "dialect-tests")
