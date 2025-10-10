##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# message("DOWNLOADING LLVM from ${DEP_URL_llvm} with SHA1 ${DEP_SHA1_llvm}")
# set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "LLVM projects to build")
# set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "LLVM targets to build")
# set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "Enable LLVM assertions")
# set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI in LLVM")
# set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Enable libedit in LLVM")
# set(LLVM_BUILD_TOOLS ON CACHE BOOL "Build LLVM tools")
# set(LLVM_INSTALL_UTILS OFF CACHE BOOL "Install LLVM utilities")
# set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "Build LLVM tests")
# set(LLVM_DISABLE_ASSEMBLY_FILES OFF CACHE BOOL "disable assembly")
# find_path(LOCAL_LLVM
#   NAMES CMakeLists.txt
#   PATHS
#   "${CMAKE_SOURCE_DIR}/../llvm/llvm"
#   "${CMAKE_SOURCE_DIR}/llvm/llvm"
#   "${CMAKE_SOURCE_DIR}/3rd-party/llvm/llvm"
#   "${CMAKE_SOURCE_DIR}/../llvm-project/llvm"
#   "${CMAKE_SOURCE_DIR}/llvm-project/llvm"
#   "${CMAKE_SOURCE_DIR}/3rd-party/llvm-project/llvm"
#   NO_DEFAULT_PATH)
# if(LOCAL_LLVM)
#   message(STATUS "found llvm source in local working directory")
#   message(STATUS "llvm SOURCE_DIR ${LOCAL_LLVM}")
#   FetchContent_Declare(
#     llvm
#     SOURCE_DIR ${LOCAL_LLVM})
#   FetchContent_MakeAvailable (llvm)
# else()                          #
#   message(STATUS "cannot find llvm in ${CMAKE_SOURCE_DIR}/../llvm or ${CMAKE_SOURCE_DIR}/../Llvm")
#   message(STATUS "fetch llvm source from ${DEP_URL_llvm} @ ${DEP_SHA1_llvm}")
#   FetchContent_Declare(
#     llvm
#     GIT_REPOSITORY ${DEP_URL_llvm}
#     GIT_TAG ${DEP_SHA1_llvm}
#     GIT_SUBMODULES_RECURSE
#     DOWNLOAD_EXTRACT_TIMESTAMP TRUE
#     EXCLUDE_FROM_ALL
#   )
#   FetchContent_MakeAvailable (llvm)
# endif()
