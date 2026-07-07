##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

find_path(ONNX_MLIR_SOURCE_TREE
  NAMES "src/Dialect/ONNX/ONNXDialect.hpp"
  PATHS
  "${CMAKE_SOURCE_DIR}/../onnx-mlir"
  "${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir"
  "${CMAKE_SOURCE_DIR}/thirdparty/onnx-mlir"
  "${CMAKE_SOURCE_DIR}/onnx-mlir"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT ONNX_MLIR_SOURCE_TREE)
  message(WARNING "cannot find ONNX-MLIR source tree. ONNX_MLIR_SOURCE_TREE=${ONNX_MLIR_SOURCE_TREE}, search in "
    "\n\t${CMAKE_SOURCE_DIR}/../onnx-mlir"
    "\n\t${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir"
    "\n\t${CMAKE_SOURCE_DIR}/thirdparty/onnx-mlir"
    "\n\t${CMAKE_SOURCE_DIR}/onnx-mlir"
    "\n\tMLIR backend will be disabled."
  )
  return()
else()
  message(STATUS "ONNX_MLIR_SOURCE_TREE: ${ONNX_MLIR_SOURCE_TREE}")
endif()


find_path(ONNX_MLIR_BINARY_DIR
  NAMES "src/Dialect/ONNX/ONNXDialect.hpp.inc"
  PATHS
  "${CMAKE_BINARY_DIR}/../onnx-mlir"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT ONNX_MLIR_BINARY_DIR)
  message(WARN "cannot find ONNX-MLIR source tree. ONNX_MLIR_BINARY_DIR=${ONNX_MLIR_BINARY_DIR}, search in "
    "\n\t${CMAKE_BINARY_DIR}/../onnx-mlir"
  )
else()
  message(STATUS "ONNX_MLIR_BINARY_DIR: ${ONNX_MLIR_BINARY_DIR}")
endif()

# find_package(ONNX REQUIRED CONFIG)
# message("DOWNLOADING MLIR from ${DEP_URL_onnx-mlir} with SHA1 ${DEP_SHA1_onnx-mlir}")
# set(ONNX_MLIR_BUILD_TESTS OFF CACHE BOOL "Build ONNX-MLIR tests")
# set(ONNX_MLIR_ENABLE_JAVA OFF CACHE BOOL "Enable ONNX-MLIR Java bindings")
# set(MLIR_DIR "${llvm_BINARY_DIR}/lib/cmake/mlir" CACHE STRING "the llvm dir")
# list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/cmake/modules")
# list(APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/../mlir/cmake/modules")

# find_path(LOCAL_ONNX-MLIR
#   NAMES CMakeLists.txt
#   PATHS
#   "${CMAKE_SOURCE_DIR}/../onnx-mlir"
#   "${CMAKE_SOURCE_DIR}/onnx-mlir"
#   "${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir"
#   "${CMAKE_SOURCE_DIR}/../onnx-mlir-project/onnx-mlir"
#   "${CMAKE_SOURCE_DIR}/onnx-mlir-project/onnx-mlir"
#   "${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir-project/onnx-mlir"
#   NO_DEFAULT_PATH)
# if(LOCAL_ONNX-MLIR)
#   message(STATUS "found onnx-mlir source in local working directory")
#   message(STATUS "onnx-mlir SOURCE_DIR ${LOCAL_ONNX-MLIR}")
#   FetchContent_Declare(
#    onnx-mlir
#    SOURCE_DIR ${LOCAL_ONNX-MLIR})
#   FetchContent_MakeAvailable(onnx-mlir)
# else()
#   message(STATUS "cannot find onnx-mlir in ${CMAKE_SOURCE_DIR}/../onnx-mlir or ${CMAKE_SOURCE_DIR}/../Onnx-Mlir")
#   message(STATUS "fetch onnx-mlir source from ${DEP_URL_onnx-mlir} @ ${DEP_SHA1_onnx-mlir}")
#   FetchContent_Declare(
#   onnx-mlir
#   GIT_REPOSITORY ${DEP_URL_onnx-mlir}
#   GIT_TAG ${DEP_SHA1_onnx-mlir}
#   GIT_SUBMODULES_RECURSE
#   DOWNLOAD_EXTRACT_TIMESTAMP TRUE
#   EXCLUDE_FROM_ALL
# )
#   FetchContent_MakeAvailable(onnx-mlir)
# endif()
