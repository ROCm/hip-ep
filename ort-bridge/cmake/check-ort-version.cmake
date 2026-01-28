##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Check ONNX Runtime version for plugin EP support
# According to ONNX Runtime release notes, Plugin-based Execution Provider (Plugin EP)
# official API support was introduced in v1.23.0
# Reference: https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0
# This version includes required APIs:
#   - OpAttr_GetName (PR 25224)
#   - GetSessionOptionsConfigEntries (PR 25277)

set(MINIMUM_ONNXRUNTIME_VERSION "1.23.0")

if(DEFINED onnxruntime_VERSION)
  if(onnxruntime_VERSION VERSION_LESS ${MINIMUM_ONNXRUNTIME_VERSION})
    message(FATAL_ERROR
      "ONNX Runtime version ${onnxruntime_VERSION} is not supported.\n"
      "Minimum required version: ${MINIMUM_ONNXRUNTIME_VERSION} (for plugin EP support)\n"
      "Please install ONNX Runtime >= ${MINIMUM_ONNXRUNTIME_VERSION}")
  else()
    message(STATUS "✓ ONNX Runtime version ${onnxruntime_VERSION} meets requirements (>= ${MINIMUM_ONNXRUNTIME_VERSION})")
  endif()
else()
  # Version not available from find_package - warn but continue
  # Compilation will fail naturally if required APIs are missing
  message(WARNING
    "ONNX Runtime version could not be determined from find_package().\n"
    "Minimum required version: ${MINIMUM_ONNXRUNTIME_VERSION} (for plugin EP support)\n"
    "Build will fail if ONNX Runtime < ${MINIMUM_ONNXRUNTIME_VERSION}")
endif()
