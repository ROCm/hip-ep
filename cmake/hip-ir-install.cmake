##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Install/export rules for the `hip-ir` package. Included from
# lib/Dialect/IR/CMakeLists.txt when HIP_IR_INSTALL is ON (default: only when
# hip-ir is the top-level project, so a full hip-ep build is unaffected).
#
# Requires, from the dialect subdirectories:
#   HIP_IR_SOURCE_INCLUDE_ROOT    <hip-ep>/include
#   HIP_IR_GENERATED_INCLUDE_ROOT TableGen output root (embedder-dependent)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT HIP_IR_VERSION)
  set(HIP_IR_VERSION "1.0.0")
endif()

install(TARGETS HipDialectIR
        EXPORT  hip-ir-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Public headers plus the .td sources: downstreams run mlir_tablegen against
# HipOps.td, which includes its siblings by "hip/Dialect/IR/..." path.
install(DIRECTORY "${HIP_IR_SOURCE_INCLUDE_ROOT}/hip/Dialect/IR/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hip/Dialect/IR"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td")

# HipDialect.h includes the generated .inc files, so they are part of the
# public surface and must ship too.
install(DIRECTORY "${HIP_IR_GENERATED_INCLUDE_ROOT}/hip/Dialect/IR/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hip/Dialect/IR"
        FILES_MATCHING PATTERN "*.inc")

install(EXPORT hip-ir-targets
        FILE        hip-ir-targets.cmake
        NAMESPACE   hip::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip-ir")

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/hip-ir-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-ir-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip-ir"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

# Additive ops are a minor bump, op-semantics changes are a major bump.
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hip-ir-config-version.cmake"
  VERSION       "${HIP_IR_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-ir-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/hip-ir-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hip-ir")
