##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(CheckCXXSymbolExists)
include(CheckSourceCompiles)


set(HEADERS "${ONNXRUNTIME_SOURCE_TREE_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
set(CODE_HEAD "#define ORT_API_MANUAL_INIT 1\n
#include \"${HEADERS}\"\n
int main() {
")
set(CODE_END "return 0;}")
## checking version of ONNX Runtime
check_source_compiles(CXX  "${CODE_HEAD}  (void)Ort::GetApi().OpAttr_GetName; ${CODE_END}"
   HAVE_ORT_OpAttr_GetName)
check_source_compiles(CXX  "${CODE_HEAD}  (void)Ort::GetApi().GetSessionOptionsConfigEntries; ${CODE_END}"
   HAVE_ORT_GetSessionOptionsConfigEntries)




if (NOT HAVE_ORT_OpAttr_GetName)
  message(FATAL_ERROR "OpAttr_GetName is not found in onnxruntime_c_api.h, please check your ONNX Runtime version includes PR 25224")
endif()

if (NOT HAVE_ORT_GetSessionOptionsConfigEntries)
  message(FATAL_ERROR "GetSessionOptionsConfigEntries is not found in onnxruntime_c_api.h, please check your ONNX Runtime version includes PR 25277")
endif()
