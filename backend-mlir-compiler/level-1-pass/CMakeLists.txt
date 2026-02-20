##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring level-1-pass (MLIR compiler plugin)")

find_package(glog CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Generate protobuf files
set(PROTO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/proto/metadata.proto")
set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC})

set(LIB_NAME morphizen-level1-pass-mlir-compiler)
add_library(${LIB_NAME} STATIC
    src/pass_main.cpp
    src/MlirCompiler.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
    morphizen::morphizen-core-static
    glog::glog
    protobuf::libprotobuf
    # NOTE: morphizen-mlir-compiler.dll is loaded dynamically at runtime, not linked
)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROTO_OUT_DIR}  # For generated proto headers
)

target_compile_definitions(${LIB_NAME} PRIVATE
    -DOUTPUT_NAME="morphizen-level1-pass-mlir-compiler"
)

# Link to morphizen EP target with whole archive
if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
    target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
        PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>"
    )
endif()

set_target_properties(${LIB_NAME} PROPERTIES
    FOLDER "mlir"
)

message(STATUS "level-1-pass will dynamically load morphizen-mlir-compiler.dll at runtime (morphizen plugin pattern)")
