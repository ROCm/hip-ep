##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(HipCInterface STATIC
    CompilerAPI.cpp
)

target_include_directories(HipCInterface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(HipCInterface PRIVATE
    HIP_COMPILER_EXPORTS
)

target_link_libraries(HipCInterface PUBLIC
    LibHipCompiler
    HipSchemasLib
)

install(TARGETS HipCInterface
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
