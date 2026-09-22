##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##

# Package-facing standalone HIP kernel unit tests. Keeping this file beside
# the leaf sources makes example/ the single test entry.
#
# Each target compiles only its leaf-specific host driver. GPU implementations
# are compiled once in custom_kernels_<arch> and dynamically linked by every
# test executable, so no test executable embeds a duplicate kernel fatbin or
# autotune-LUT resolver.
if(NOT WIN32)
  message(STATUS "Kernel package unit-test executables are Windows-only; skipping")
  return()
endif()

function(add_kernel_package_ut arch executable_name source_file)
  set(kernel_target custom_kernels_${arch})
  if(NOT TARGET ${kernel_target})
    message(FATAL_ERROR
      "Kernel package UT ${executable_name} requires target ${kernel_target}")
  endif()

  set(target ${executable_name}-${arch})
  hip_add_executable(${target}
    ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
    INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/include
    COMPILE_OPTIONS
      -O3
      -std=c++17
      -Wno-c23-extensions
    LINK_LIBRARIES
      ${kernel_target}
    OFFLOAD_ARCHS
      ${arch}
  )
  target_compile_definitions(${target} PRIVATE
    HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS=1
  )
  if(source_file MATCHES "^gqa/(decode|prefill)/Xfp16_Wfp16_Yfp16/")
    # The fp16 GQA leaves exercise the production lookup-only entry points.
    # Their resolver and LUT blobs stay in custom_kernels_<arch>; the source
    # guard suppresses the old per-executable #embed payload.
    target_compile_definitions(${target} PRIVATE
      HIPDNN_LUT_LINKED_EXTERNALLY=1
    )
    target_include_directories(${target} PRIVATE
      ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa
    )
  endif()
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
    hipdnn-kernel-ut-gemm-xfp16-wfp16-yfp16
    gemm/Xfp16_Wfp16_Yfp16/test_gemm.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gemm-xbf16-wbf16-ybf16
    gemm/Xbf16_Wbf16_Ybf16/test_gemm.cpp)
  add_kernel_package_ut(${_arch}
    hipdnn-kernel-ut-gemm-xfp32-wfp32-yfp32
    gemm/Xfp32_Wfp32_Yfp32/test_gemm.cpp)
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
