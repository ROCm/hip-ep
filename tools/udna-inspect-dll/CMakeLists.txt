##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(udna-inspect-dll
  udna-inspect-dll.cpp
)

# Use LLVM for cross-platform DLL loading (DynamicLibrary) and JSON parsing
llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(udna-inspect-dll
  PRIVATE
  ${llvm_libs}
)

# Install target
install(TARGETS udna-inspect-dll
  RUNTIME DESTINATION bin
)
