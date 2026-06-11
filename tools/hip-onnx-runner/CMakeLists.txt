##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# onnxruntime::onnxruntime may already be provided by cmake/deps.cmake (either
# find_package against an installed prefix or the release-zip fallback). Only
# run find_package here when the target is absent, and gate on the target -- not
# the onnxruntime_FOUND variable -- so the synthesized fallback target counts.
if(NOT TARGET onnxruntime::onnxruntime)
  find_package(onnxruntime CONFIG QUIET)
endif()
if(NOT TARGET onnxruntime::onnxruntime)
  message(STATUS "hip-onnx-runner: onnxruntime not found, skipping")
  return()
endif()

add_executable(hip-onnx-runner
  hip-onnx-runner.cpp
  ${CMAKE_SOURCE_DIR}/lib/Support/CrashHandler.cpp
)

target_compile_features(hip-onnx-runner PRIVATE cxx_std_17)

target_include_directories(hip-onnx-runner PRIVATE
  ${CMAKE_SOURCE_DIR}/include  # Shared project headers (timing.h, etc.)
  ${CMAKE_SOURCE_DIR}/lib/Support  # CrashHandler.h
)

target_link_libraries(hip-onnx-runner PRIVATE
  onnxruntime::onnxruntime
  cpptrace::cpptrace
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
