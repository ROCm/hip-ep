set(ONNXRUNTIME_SOURCE_TREE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime" CACHE PATH "Path to the root of the ONNX Runtime source tree" )
find_path(VAIP_ORT_API_H
  NAMES vaip/vaip_ort_api.h
  PATHS "${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(NOT VAIP_ORT_API_H)
  message(FATAL_ERROR "cannot find vaip_ort_api.h in ${ONNXRUNTIME_SOURCE_TREE_DIR}/onnxruntime/core/providers/vitisai/include")
else()
  message(STATUS "VAIP_ORT_API_H: ${VAIP_ORT_API_H}")
endif()
# get directory of vaip_ort_api.h
get_filename_component(VAIP_ORT_API_DIR "${VAIP_ORT_API_H}/.." DIRECTORY)
