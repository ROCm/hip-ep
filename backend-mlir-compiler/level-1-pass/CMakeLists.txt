##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring level-1-pass (MLIR compiler plugin)")

find_package(glog CONFIG REQUIRED)

set(LIB_NAME morphizen-level1-pass-mlir-compiler)
add_library(${LIB_NAME} STATIC
    src/pass_main.cpp
    src/CompilationConfig.cpp
    src/MetadataBuilder.cpp
    src/FusionManager.cpp
    src/MlirCompiler.cpp  # NEW - simplified compiler using C API
    # Removed: MlirPipelineBuilder.cpp, MlirParser.cpp, MlirTransformer.cpp, LlvmCompiler.cpp
    # These are no longer needed - all MLIR compilation is done via morphizen-mlir-compiler.dll C API
)

# Link dependencies (NO morphizen-mlir-compiler - loaded at runtime)
target_link_libraries(${LIB_NAME} PRIVATE
    morphizen::morphizen-core-static
    glog::glog
    mlir_compilation_proto  # Keep for metadata
    # NOTE: morphizen-mlir-compiler.dll is loaded dynamically at runtime, not linked
)

# Include directories - ONLY C API headers (for type definitions), NO LLVM/MLIR headers
target_include_directories(${LIB_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-mlir-compiler/include  # C API types only
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
