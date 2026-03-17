##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Create test-model-dll executable
add_executable(test-model-dll
  test-model-dll.cpp
)

# Use LLVM for cross-platform DLL loading (DynamicLibrary) and JSON parsing
llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(test-model-dll
  PRIVATE
  ${llvm_libs}
)

target_include_directories(test-model-dll
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include   # For hip/Support/DiskFileSystem.h
)

# Install target
install(TARGETS test-model-dll
  RUNTIME DESTINATION bin
)
