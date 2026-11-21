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

target_link_libraries(HipCInterface PUBLIC
    LibHipCompiler
    HipSchemasLib
    morphizen-utils
)

if(TARGET hipdnn_graph_runtime)
    target_link_libraries(HipCInterface PUBLIC hipdnn_graph_runtime)
    target_compile_definitions(HipCInterface PRIVATE
        HIPDNN_GRAPH_RUNTIME_AVAILABLE
    )
endif()

install(TARGETS HipCInterface
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
    target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
        PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,HipCInterface>")
else()
    message(WARNING
        "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined or not a "
        "target - the compiler will not be linked into the EP")
endif()
