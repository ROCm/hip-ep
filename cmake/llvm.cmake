##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring LLVM/MLIR for morphizen-mlir")

# LLVM configuration options
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI in LLVM")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Enable libedit in LLVM")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "Install LLVM utilities")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "Build LLVM tests")
set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "Use static zlib")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Enable zstd compression")

# LLVM commit hash
set(DEP_SHA1_llvm "f8cb7987c64dcffb72414a40560055cb717dbf74")
set(DEP_URL_llvm "https://github.com/llvm/llvm-project.git")

# Try to find pre-installed LLVM/MLIR first
find_package(LLVM QUIET CONFIG)
find_package(MLIR QUIET CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
  message(STATUS "Found pre-installed LLVM and MLIR")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
  set(MORPHIZEN_LLVM_PREINSTALLED ON CACHE BOOL "Using pre-installed LLVM" FORCE)
else()
  message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent and build inline")
  set(MORPHIZEN_LLVM_PREINSTALLED OFF CACHE BOOL "Using FetchContent LLVM" FORCE)
  
  # Try to find LLVM in local directories first
  find_path(LOCAL_LLVM
    NAMES CMakeLists.txt
    PATHS
    "${CMAKE_SOURCE_DIR}/../llvm/llvm"
    "${CMAKE_SOURCE_DIR}/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm/llvm"
    "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
    "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
    NO_DEFAULT_PATH)
  
  if(LOCAL_LLVM)
    message(STATUS "Found LLVM source in local working directory")
    message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
    FetchContent_Declare(
      llvm-project
      SOURCE_DIR ${LOCAL_LLVM}/..)
  else()
    message(STATUS "Cannot find LLVM in local directories")
    message(STATUS "Fetching LLVM source from ${DEP_URL_llvm} @ ${DEP_SHA1_llvm}")
    message(STATUS "WARNING: This will download and build LLVM, which may take a long time")
    FetchContent_Declare(
      llvm-project
      GIT_REPOSITORY ${DEP_URL_llvm}
      GIT_TAG ${DEP_SHA1_llvm}
      GIT_SUBMODULES_RECURSE
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      EXCLUDE_FROM_ALL
    )
  endif()
  
  # Make LLVM available - this adds LLVM as a subdirectory
  FetchContent_MakeAvailable(llvm-project)
  
  # Set include directories for downstream targets
  set(LLVM_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/llvm/include"
    "${llvm-project_BINARY_DIR}/include"
    CACHE PATH "LLVM include directories" FORCE)
  set(MLIR_INCLUDE_DIRS 
    "${llvm-project_SOURCE_DIR}/mlir/include"
    "${llvm-project_BINARY_DIR}/tools/mlir/include"
    CACHE PATH "MLIR include directories" FORCE)
  set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
  set(MLIR_DIR "${llvm-project_BINARY_DIR}/lib/cmake/mlir" CACHE PATH "" FORCE)
  
  message(STATUS "LLVM source dir: ${llvm-project_SOURCE_DIR}")
  message(STATUS "LLVM binary dir: ${llvm-project_BINARY_DIR}")
  message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
  message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
endif()

message(STATUS "LLVM/MLIR configuration complete")
