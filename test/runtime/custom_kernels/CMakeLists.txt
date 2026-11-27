##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# The K-quant, Q8_1 and token-major GQA kernels of custom_kernels_<arch>
# against host references, on a gfx1151 GPU (exit 77: no such GPU, skipped).
# Each slice a gfx1151 loads is tested: its own and the gfx11-generic one
# release builds ship.

include("${PROJECT_SOURCE_DIR}/lib/Runtime/Kernels/cmake/hip_utils.cmake")

foreach(arch IN ITEMS gfx1151 gfx11-generic)
  if(NOT "${arch}" IN_LIST HIP_ARCHITECTURES OR NOT TARGET custom_kernels_${arch})
    continue()
  endif()
  foreach(source IN ITEMS
      kquant_matmul_test.hip
      kquant_swiglu_test.hip
      skip_rms_norm_q8_1_test.hip
      token_major_gqa_test.hip)
    get_filename_component(name "${source}" NAME_WE)
    set(target "${name}-${arch}")
    hip_add_executable(${target} ${source} KQuantReference.cpp
      INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/lib/Runtime/Kernels/include"
        "${PROJECT_SOURCE_DIR}/include"
      LINK_LIBRARIES
        custom_kernels_${arch}
      OFFLOAD_ARCHS ${arch})
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
      LABELS "gpu;custom-kernels-gpu"
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77)
    set_target_properties(${target} PROPERTIES FOLDER "runtime-tests")
  endforeach()
endforeach()
