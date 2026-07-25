##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring level-1-pass (MLIR compiler plugin)")

# glog::glog and protobuf::libprotobuf are provided by cmake/deps.cmake or the
# morphizen subdirectory (find_package or from-source fallback). Only resolve
# standalone when the targets are absent.
if(NOT TARGET glog::glog)
  find_package(glog CONFIG REQUIRED)
endif()
if(NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf CONFIG REQUIRED)
endif()

# Use shared protobuf library (no local generation needed)
# Proto is generated in parent CMakeLists.txt as mlir_compilation_proto

set(LIB_NAME morphizen-level1-pass-mlir-compiler)
add_library(${LIB_NAME} STATIC
    src/pass_main.cpp
    src/MlirCompiler.cpp
    ${CMAKE_SOURCE_DIR}/lib/Support/CrashHandler.cpp
)

# Link dependencies
target_link_libraries(${LIB_NAME} PRIVATE
    morphizen::morphizen-core-static
    glog::glog
    mlir_compilation_proto  # Shared protobuf library
    cpptrace::cpptrace
    # NOTE: hip-compiler.dll is loaded dynamically at runtime, not linked
)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/..  # For generated proto headers from parent
    ${CMAKE_SOURCE_DIR}/include     # Shared project headers (timing.h, etc.)
    ${CMAKE_SOURCE_DIR}/lib/Support # CrashHandler.h
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

message(STATUS "level-1-pass will dynamically load hip-compiler.dll at runtime (morphizen plugin pattern)")
