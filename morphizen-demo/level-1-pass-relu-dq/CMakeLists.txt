##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package (glog CONFIG REQUIRED)
include(cmake/generate_pattern_inc.cmake)
set(LIB_NAME morphizen-level1-pass-relu-dq)
add_library(${LIB_NAME} STATIC src/pass_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/relu_dq_pattern_json.hpp)
target_link_libraries(${LIB_NAME} PUBLIC
	relu_dq_proto
	morphizen::morphizen-core-static
	glog::glog
)
target_include_directories(${LIB_NAME} PRIVATE
	include
	${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(${LIB_NAME} PRIVATE -DOUTPUT_NAME="morphizen-level1-pass-relu-dq")
target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "relu-dq"
)
