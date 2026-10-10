##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##

# Package-facing standalone HIP kernel unit tests. Keeping this file beside
# the leaf sources makes example/ the single test entry.
#
# Built when BUILD_HIP_KERNEL_UNIT_TESTS is ON (the default), on Windows with
# the real runtime. lib/Runtime/Kernels/CMakeLists.txt adds this directory, so
# hip::host and the hip_utils.cmake variables are in scope here. Outputs:
# bin/kernel-tests/<arch>/hipdnn-kernel-ut-*.exe plus the bin/hipdnn-kernel-ut
# entry point that runs every leaf for the live GPU arch.
#
# Each target compiles only its leaf-specific host driver. GPU implementations
# are compiled once in custom_kernels_<arch> and dynamically linked by every
# test executable, so no test executable embeds a duplicate kernel fatbin or
# autotune-LUT resolver.
if(NOT WIN32)
  message(STATUS "Kernel package unit-test executables are Windows-only; skipping")
  return()
endif()

# The leaves convert fp16 <-> fp32 on the host, which clang lowers to
# compiler-rt calls (__extendhfsf2 / __truncsfhf2). hip::host carries
# clang_rt.builtins only when HIP_CXX_COMPILER is clang, and here it is MSVC,
# so ask hipcc for the library the way hip-config-amd.cmake asks clang.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env HIPCC_VERBOSE=0
          ${HIPCC_EXECUTABLE} -print-libgcc-file-name --rtlib=compiler-rt
  OUTPUT_VARIABLE _kernel_ut_clang_rt
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _kernel_ut_clang_rt_result
)
file(TO_CMAKE_PATH "${_kernel_ut_clang_rt}" _kernel_ut_clang_rt)
if(NOT _kernel_ut_clang_rt_result EQUAL 0 OR NOT EXISTS "${_kernel_ut_clang_rt}")
  message(FATAL_ERROR
    "Kernel package UTs need clang_rt.builtins, but ${HIPCC_EXECUTABLE} "
    "-print-libgcc-file-name returned '${_kernel_ut_clang_rt}'")
endif()

function(add_kernel_package_ut arch executable_name source_file)
  set(kernel_target custom_kernels_${arch})
  if(NOT TARGET ${kernel_target})
    message(FATAL_ERROR
      "Kernel package UT ${executable_name} requires target ${kernel_target}")
  endif()

  # On Windows hip_add_executable compiles the sources with a hipcc custom
  # command, so defines and include paths must go through its own
  # COMPILE_OPTIONS / INCLUDE_DIRECTORIES; target_compile_definitions and
  # target_include_directories would never reach hipcc.
  set(ut_includes ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/include)
  set(ut_options
    -O3
    -std=c++17
    -Wno-c23-extensions
    -DHIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS=1
  )
  if(source_file MATCHES "^gqa/(decode|prefill)/Xfp16_Wfp16_Yfp16/")
    # The fp16 GQA leaves exercise the production lookup-only entry points.
    # Their resolver and LUT blobs stay in custom_kernels_<arch>; the source
    # guard suppresses the old per-executable #embed payload.
    list(APPEND ut_options -DHIPDNN_LUT_LINKED_EXTERNALLY=1)
    list(APPEND ut_includes ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa)
  endif()

  set(target ${executable_name}-${arch})
  hip_add_executable(${target}
    ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
    INCLUDE_DIRECTORIES
      ${ut_includes}
    COMPILE_OPTIONS
      ${ut_options}
    LINK_LIBRARIES
      ${kernel_target}
      ${_kernel_ut_clang_rt}
    OFFLOAD_ARCHS
      ${arch}
  )
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${executable_name}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/kernel-tests/${arch}
    FOLDER "kernel-package-tests"
  )
  list(APPEND KERNEL_PACKAGE_UT_TARGETS ${target})
  set(KERNEL_PACKAGE_UT_TARGETS ${KERNEL_PACKAGE_UT_TARGETS} PARENT_SCOPE)

  install(TARGETS ${target}
    RUNTIME DESTINATION bin/kernel-tests/${arch}
  )

  if(BUILD_TESTING)
    string(REPLACE "-" "_" ctest_name ${target})
    add_test(NAME ${ctest_name} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${ctest_name} PROPERTIES
      WORKING_DIRECTORY $<TARGET_FILE_DIR:${kernel_target}>
      LABELS "gpu;kernel-ut;${arch}"
    )
  endif()
endfunction()

foreach(_arch IN LISTS HIP_ARCHITECTURES)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu2-yfp16
    matmul_nbits/Xfp16_Wu2_Yfp16/test_matmul_nbits_u2.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu3-yfp16
    matmul_nbits/Xfp16_Wu3_Yfp16/test_matmul_nbits_u3.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-matmul-nbits-xfp16-wu4-yfp16
    matmul_nbits/Xfp16_Wu4_Yfp16/test_matmul_nbits.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-matmul-nbits-xfp16-wi8-yfp16
    matmul_nbits/Xfp16_Wi8_Yfp16/test_matmul_nbits_i8.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gqa-decode-xfp16-wfp16-yfp16
    gqa/decode/Xfp16_Wfp16_Yfp16/test_gqa_decode.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gqa-decode-xfp16-wi8-yfp16
    gqa/decode/Xfp16_Wi8_Yfp16/test_gqa_decode_i8.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gqa-prefill-xfp16-wfp16-yfp16
    gqa/prefill/Xfp16_Wfp16_Yfp16/test_gqa_prefill.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gqa-prefill-xfp16-wi8-yfp16
    gqa/prefill/Xfp16_Wi8_Yfp16/test_gqa_prefill_i8.cpp)
endforeach()

# One package-facing entry point finds the live GPU arch, invokes every leaf
# executable, and aggregates their per-case rows into bin/out/results.csv.
list(GET HIP_ARCHITECTURES 0 _kernel_ut_primary_arch)
hip_add_executable(hipdnn-kernel-ut
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_ut_main.cpp
  COMPILE_OPTIONS
    -O3
    -std=c++17
  OFFLOAD_ARCHS
    ${_kernel_ut_primary_arch}
)
add_dependencies(hipdnn-kernel-ut ${KERNEL_PACKAGE_UT_TARGETS})
set_target_properties(hipdnn-kernel-ut PROPERTIES
  OUTPUT_NAME hipdnn-kernel-ut
  FOLDER "kernel-package-tests"
)
install(TARGETS hipdnn-kernel-ut RUNTIME DESTINATION bin)
