##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
find_package(glog CONFIG REQUIRED)

# Only use find_package for LLVM/MLIR when using pre-installed version
# When using FetchContent, targets are already available in the CMake scope
# See doc/FETCHCONTENT_LLVM_ISSUE.md for detailed explanation
if(MORPHIZEN_LLVM_PREINSTALLED)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
endif()

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
target_link_libraries(${morphizen_ONNXRUNTIME_VITISAI_EP_TARGET} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
set_target_properties(${LIB_NAME} PROPERTIES
	FOLDER "mlir"
)
