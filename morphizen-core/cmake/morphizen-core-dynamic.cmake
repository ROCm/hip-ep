##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generate version info header for cross-platform version information
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_version_header.cmake)
generate_version_info_header()

add_library(${morphizen_CORE_DYNAMIC_UNIQUE_ID} SHARED src/main.cpp ${ryzenai_version_rc_file})
message(STATUS "create target ${morphizen_CORE_DYNAMIC_UNIQUE_ID} for onnxruntime_vitisai_ep.dll")
add_library (morphizen::${morphizen_CORE_DYNAMIC_UNIQUE_ID} ALIAS ${morphizen_CORE_DYNAMIC_UNIQUE_ID})
set_target_properties(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PROPERTIES FOLDER morphizen)
# set output name of ${morphizen_CORE_DYNAMIC_UNIQUE_ID}, it is required by MorphiZen EP.
set_target_properties(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PROPERTIES OUTPUT_NAME ${morphizen_OUTPUT_NAME})

if(MSVC)
  if(morphizen_ENABLE_ORT_BRIDGE)
    target_sources(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PRIVATE onnxruntime_morphizen_ep_with_ort_bridge.def)
  else()
    target_sources(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PRIVATE onnxruntime_morphizen_ep.def)
  endif()
endif(MSVC)


# NOTE: do not use $<LINK_LIBRARY:WHOLE_ARCHIVE, morphizen-core-static
#
# WHY, it seems that WHOLE_ARCHIVE can be only marked once. It means
# if we add the mark here, then, all targets dependes on
# morphizen-core-static directly or indirectly, must be marked with
# WHOLE_ARCHIVE. In stead, we manually maintain
# onnxruntime_morphizen_ep.def file. tools/parse_cl_link_error.py is
# used to parse the link error and update onnxruntime_morphizen_ep.def
# automatically, see tools/parse_cl_link_error.py for more details.
#
# morphizen-graph and morphizen-pattern are already linked transitively
# through morphizen-core-static, so we don't link them directly here
# to avoid "link item occurred with different features" errors.
target_link_libraries(${morphizen_CORE_DYNAMIC_UNIQUE_ID}
  PRIVATE
  morphizen-core-static
)

target_include_directories(${morphizen_CORE_DYNAMIC_UNIQUE_ID}
  PUBLIC
  $<BUILD_INTERFACE:$<TARGET_PROPERTY:morphizen-core-static,INTERFACE_INCLUDE_DIRECTORIES>>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}  # For generated version_info_config.h
)
target_compile_features(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PUBLIC cxx_std_17)
if(MSVC)
  target_compile_options(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PUBLIC "/Zc:__cplusplus")
else(MSVC)
  target_compile_options(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PUBLIC "-fPIC")
endif(MSVC)
target_compile_definitions(${morphizen_CORE_DYNAMIC_UNIQUE_ID}
  PRIVATE
    "-DMORPHIZEN_USE_DLL=1"
    "-DMORPHIZEN_EXPORT_DLL=1"
    "-DHAVE_VERSION_INFO_CONFIG=1"  # Enable generated version header
  PUBLIC
    "-DONNX_NAMESPACE=onnx"
)
target_compile_options(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_COMPILER_OPTIONS}>")
target_link_options(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MORPHIZEN_LINKER_OPTIONS}>")

# Localize the statically-linked LLVM symbols (version script) so ROCm's
# libamd_comgr.so cannot bind its own @LLVM_22.0 llvm:: references to our
# ABI-incompatible copy and segfault its in-process device-code compile.
# Windows controls the same surface via onnxruntime_morphizen_ep.def.
if(NOT MSVC)
    target_link_options(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_morphizen_ep.exports")
endif()
set_target_properties(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PROPERTIES
  VS_DEBUGGER_COMMAND "${CMAKE_INSTALL_PREFIX}\\bin\\test_onnx_runner.exe"
  VS_DEBUGGER_COMMAND_ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}\\..\\..\\test_onnx_runner\\data\\pt_resnet50.onnx"
  VS_DEBUGGER_ENVIRONMENT "XLNX_ONNX_EP_VERBOSE=2
DEBUG_LOG_LEVEL=info
DEBUG_MORPHIZEN_PASS=1
MORPHIZEN_DEBUG_TAR_ENTRY=1
MORPHIZEN_DEBUG_TAR_FILE=1
DEBUG_TAR_CACHE=1
"
)

target_link_libraries(${morphizen_CORE_DYNAMIC_UNIQUE_ID} PUBLIC glog::glog protobuf::libprotobuf )
