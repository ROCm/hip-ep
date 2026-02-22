##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generate C++ code from proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS compilation_options.proto)

# Create static library for proto
add_library(MorphizenProto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(MorphizenProto PUBLIC protobuf::libprotobuf)
target_include_directories(MorphizenProto PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)

# Install proto library
install(TARGETS MorphizenProto
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
