##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Compile-time guard for the arch-gated LDS barrier in matmul_nbits_kernel.hip.
# GPU-free: only needs the HIP-capable compiler to emit device assembly and to
# preprocess the barrier gate. See check_barrier_arch_gate.py for details.
find_package(Python3 COMPONENTS Interpreter)

# Resolve a HIP-capable compiler for the guard. CMAKE_HIP_COMPILER is set by
# enable_language(HIP) in the kernels subdirectory but is directory-scoped and
# not visible here, so also consider the cache handles hip_utils.cmake and the
# HIP package leave behind: amdclang++ (_HIP_CLANGXX) and hipcc
# (HIPCC_EXECUTABLE / hip_HIPCC_EXECUTABLE -- the compiler available on Windows
# CI), then a plain PATH lookup. The script drives whichever it is handed
# (amdclang++ or the hipcc wrapper) identically. CMAKE_CXX_COMPILER is the host
# compiler and cannot target amdgcn, so it is deliberately not used.
set(_lds_barrier_hip_cc "")
foreach(_cand IN ITEMS
    "${CMAKE_HIP_COMPILER}" "${_HIP_CLANGXX}"
    "${HIPCC_EXECUTABLE}" "${hip_HIPCC_EXECUTABLE}")
  if(_cand AND NOT _lds_barrier_hip_cc)
    set(_lds_barrier_hip_cc "${_cand}")
  endif()
endforeach()
if(NOT _lds_barrier_hip_cc)
  find_program(_lds_barrier_hip_cc NAMES amdclang++ clang++ hipcc)
endif()

if(Python3_Interpreter_FOUND AND _lds_barrier_hip_cc)
  add_test(
    NAME LdsBarrierArchGate
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_barrier_arch_gate.py
  )
  set_tests_properties(LdsBarrierArchGate PROPERTIES
    ENVIRONMENT "HIP_CLANG=${_lds_barrier_hip_cc}"
    LABELS "kernels")
else()
  message(STATUS
    "Skipping LdsBarrierArchGate test "
    "(python3=${Python3_Interpreter_FOUND}, hip_cc=${_lds_barrier_hip_cc})")
endif()
