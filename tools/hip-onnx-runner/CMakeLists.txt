##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

find_package(onnxruntime CONFIG QUIET)
if(NOT onnxruntime_FOUND)
  message(STATUS "hip-onnx-runner: onnxruntime not found, skipping")
  return()
endif()

add_executable(hip-onnx-runner hip-onnx-runner.cpp)

target_compile_features(hip-onnx-runner PRIVATE cxx_std_17)

target_link_libraries(hip-onnx-runner PRIVATE
  onnxruntime::onnxruntime
)

install(TARGETS hip-onnx-runner
  RUNTIME DESTINATION bin
)

# Copy onnxruntime.dll to output directory on Windows
if(MSVC)
  add_custom_command(
    TARGET hip-onnx-runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:onnxruntime::onnxruntime>
      $<TARGET_FILE_DIR:hip-onnx-runner>
  )
endif()
