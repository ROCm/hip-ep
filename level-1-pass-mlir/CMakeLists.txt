##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(glog CONFIG REQUIRED)

# find_package works for both pre-installed and FetchContent LLVM:
# - Pre-installed: imports targets via MLIRTargets.cmake
# - FetchContent: targets already exist, MLIRConfig.cmake skips import
# See doc/FETCHCONTENT_LLVM_ISSUE.md for detailed explanation
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

set(LIB_NAME morphizen-level1-pass-mlir)
add_library(${LIB_NAME} STATIC src/pass_main.cpp)
target_link_libraries(${LIB_NAME} PUBLIC
	morphizen::morphizen-core-static
	glog::glog
	MLIRIR
	MLIRFuncDialect
	MLIRArithDialect
	MLIRParser
	MLIRSupport
)
target_include_directories(${LIB_NAME} PRIVATE
	include
	${LLVM_INCLUDE_DIRS}
	${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(${LIB_NAME} PRIVATE -DOUTPUT_NAME="morphizen-level1-pass-mlir")
target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "mlir"
)
