##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generate protobuf sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS metadata.proto)

# Create test-model-dll executable
add_executable(test-model-dll
  test-model-dll.cpp
  ${PROTO_SRCS}
)

# Link full protobuf library (needed for json_util)
target_link_libraries(test-model-dll
  PRIVATE
  protobuf::libprotobuf  # Use full library, not lite
)

# Add protobuf include directory
target_include_directories(test-model-dll
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
)

# Link platform-specific dynamic linking library (Linux only)
if(NOT WIN32)
  target_link_libraries(test-model-dll PRIVATE ${CMAKE_DL_LIBS})
endif()

# Install target
install(TARGETS test-model-dll
  RUNTIME DESTINATION bin
)
