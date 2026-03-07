##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_executable(test_mul_f32 test_mul_f32.cpp)
target_compile_definitions(test_mul_f32 PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(test_mul_f32 PRIVATE ${THEROCK_DIST}/include)
target_link_directories(test_mul_f32 PRIVATE ${THEROCK_DIST}/lib)
target_link_libraries(test_mul_f32 PRIVATE amdhip64)

# add_mlir_to_gpu_test(<name> <mlir_file> <symbol> [extra_run_args...])
#   Adds a compile + run test pair with fixture dependency.
function(add_mlir_to_gpu_test name mlir_file symbol)
  set(dll "${CMAKE_CURRENT_BINARY_DIR}/${name}.dll")

  add_test(NAME mlir_to_gpu_compile_${name}
    COMMAND $<TARGET_FILE:hip-compiler>
      "${CMAKE_CURRENT_SOURCE_DIR}/${mlir_file}" -o "${dll}")
  set_tests_properties(mlir_to_gpu_compile_${name} PROPERTIES
    FIXTURES_SETUP ${name}_dll
    ENVIRONMENT "THEROCK_DIST=${THEROCK_DIST}")

  add_test(NAME mlir_to_gpu_run_${name}
    COMMAND $<TARGET_FILE:test_mul_f32> "${dll}" "${symbol}" ${ARGN})
  set_tests_properties(mlir_to_gpu_run_${name} PROPERTIES
    FIXTURES_REQUIRED ${name}_dll
    ENVIRONMENT "PATH=${THEROCK_DIST}/bin\;$ENV{PATH}")
endfunction()

add_mlir_to_gpu_test(mul_f32       mul_f32.mlir       mul)
add_mlir_to_gpu_test(mul_f32_const mul_f32_const.mlir mul_const --const)
