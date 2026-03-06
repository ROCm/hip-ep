##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# HIP Compiler library (core MLIR -> DLL compilation)
add_subdirectory(compiler)

# Standalone CLI tool
add_subdirectory(tools)

# EP integration (only when morphizen is available)
if(TARGET morphizen::morphizen-core-static)
  add_subdirectory(level-1-pass)
  add_subdirectory(custom-op-hip)

  # E2E test (requires morphizen EP + ORT + GTest)
  if(morphizen_ENABLE_UNIT_TEST AND DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET)
    add_subdirectory(test)
  endif()
endif()
