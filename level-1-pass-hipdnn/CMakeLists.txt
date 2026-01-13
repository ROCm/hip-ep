##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# TheRock installation root - contains HIP, hipDNN, and other ROCm components
set(THEROCK_DIST "$ENV{THEROCK_DIST}" CACHE PATH "Path to TheRock dist/rocm directory")
if(NOT THEROCK_DIST)
  message(FATAL_ERROR "THEROCK_DIST must be specified. Set it via -DTHEROCK_DIST=<path> or THEROCK_DIST environment variable.")
endif()

# Add TheRock to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")

# Third-party dependencies from TheRock (nlohmann-json is required by hipdnn_sdk)
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/../../third-party/nlohmann-json/dist/share/cmake")

# Find hipDNN from TheRock
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)

find_package (glog CONFIG REQUIRED)
include(cmake/generate_pattern_inc.cmake)
set(LIB_NAME morphizen-level1-pass-hipdnn)
add_library(${LIB_NAME} STATIC src/pass_main.cpp ${CMAKE_CURRENT_BINARY_DIR}/hipdnn_pattern_json.hpp)
target_link_libraries(${LIB_NAME} PUBLIC
	hipdnn_proto
	morphizen::morphizen-core-static
	glog::glog
	hipdnn_frontend
	hipdnn_backend
)
target_include_directories(${LIB_NAME} PRIVATE
	include
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../external/hipDNNEP/include
	${THEROCK_DIST}/include
)
target_compile_definitions(${LIB_NAME} PRIVATE 
	-DOUTPUT_NAME="morphizen-level1-pass-hipdnn"
	-D__HIP_PLATFORM_AMD__
)
target_compile_options(${LIB_NAME} PRIVATE /utf-8)
target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "hipdnn"
)
