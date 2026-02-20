##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring level-1-pass (MLIR compiler plugin)")

find_package(glog CONFIG REQUIRED)

set(LIB_NAME morphizen-level1-pass-mlir-compiler)
add_library(${LIB_NAME} STATIC
    src/pass_main.cpp
    src/MlirCompiler.cpp
)

# Link dependencies (NO morphizen-mlir-compiler - loaded at runtime)
target_link_libraries(${LIB_NAME} PRIVATE
    morphizen::morphizen-core-static
    glog::glog
    # NOTE: morphizen-mlir-compiler.dll is loaded dynamically at runtime, not linked
    # NOTE: Protobuf removed - metadata is now simple JSON string building
)

# Include directories - NO morphizen-mlir-compiler headers needed (types defined locally)
target_include_directories(${LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # Note: morphizen-mlir-compiler types are defined in src/MlirCompilerTypes.h
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
