##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(MorphizenCompiler STATIC
    CompilerPipeline.cpp
)

target_include_directories(MorphizenCompiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../proto  # For compilation_options.pb.h
)

target_link_libraries(MorphizenCompiler PUBLIC
    MorphizenProto  # For compilation_options.pb.h/.cc
    MorphizenHipDialect
    MorphizenBackend
    # Note: MorphizenRuntime is a custom target, not a library - dependency handled via MorphizenBackend
    MLIRParser
    MLIRPass
    MLIRIR
    MLIRSupport
    MLIRTransforms
    MLIRLLVMToLLVMIRTranslation
    OMONNXOps
)

# Install library
install(TARGETS MorphizenCompiler
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
