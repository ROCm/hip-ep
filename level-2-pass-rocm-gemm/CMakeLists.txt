##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(glog CONFIG REQUIRED)
include(cmake/generate_pattern_inc.cmake)

set(LIB_NAME morphizen-level2-pass-rocm-gemm)
add_library(${LIB_NAME} STATIC src/pass_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/gemm_pattern_json.hpp)
target_link_libraries(${LIB_NAME} PUBLIC
	rocm_proto
	morphizen::morphizen-core-static
	glog::glog
)
target_include_directories(${LIB_NAME} PRIVATE
	include
	${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(${LIB_NAME} PRIVATE -DOUTPUT_NAME="morphizen-level2-pass-rocm-gemm")
target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "rocm-gemm"
)
