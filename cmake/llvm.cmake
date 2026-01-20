##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Configuring LLVM/MLIR for morphizen-mlir")

# LLVM configuration
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

# LLVM commit hash from build_llvm.bat
set(DEP_SHA1_llvm "f8cb7987c64dcffb72414a40560055cb717dbf74")
set(DEP_URL_llvm "https://github.com/llvm/llvm-project.git")

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

# Check if LLVM is already installed
find_package(LLVM QUIET CONFIG)
find_package(MLIR QUIET CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
  message(STATUS "Found pre-installed LLVM and MLIR")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
  message(STATUS "MLIR_DIR: ${MLIR_DIR}")
else()
  message(STATUS "LLVM/MLIR not found in CMAKE_PREFIX_PATH, will use FetchContent and build")
  
  if(LOCAL_LLVM)
    message(STATUS "Found LLVM source in local working directory")
    message(STATUS "LLVM SOURCE_DIR: ${LOCAL_LLVM}")
    FetchContent_Declare(
      llvm
      SOURCE_DIR ${LOCAL_LLVM})
  else()
    message(STATUS "Cannot find LLVM in local directories")
    message(STATUS "Fetching LLVM source from ${DEP_URL_llvm} @ ${DEP_SHA1_llvm}")
    message(STATUS "WARNING: This will download and build LLVM, which may take several hours")
    FetchContent_Declare(
      llvm
      GIT_REPOSITORY ${DEP_URL_llvm}
      GIT_TAG ${DEP_SHA1_llvm}
      GIT_SUBMODULES_RECURSE
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      EXCLUDE_FROM_ALL
    )
  endif()
  
  FetchContent_MakeAvailable(llvm)
  
  message(STATUS "LLVM configured at: ${llvm_BINARY_DIR}")
  message(FATAL_ERROR 
    "LLVM/MLIR FetchContent is configured but not yet built and installed.\n"
    "Please run build_llvm.bat first to build and install LLVM/MLIR to ../local\n"
    "Then run build.bat again to build morphizen-mlir.\n"
    "\n"
    "Reason: MorphiZen's mlir-imp uses find_package(MLIR) which requires\n"
    "MLIRTargets.cmake file that is only generated during 'cmake --install'.\n"
    "FetchContent alone cannot provide this file.")
endif()

message(STATUS "LLVM/MLIR configuration complete")
