##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generic ORT custom-op schema registration module. Every schema-only op
# (one that only needs to let ORT load/Resolve a graph containing it, with
# no real kernel or shape inference) is a one-line entry in the
# kSchemaOnlyOps table in schema_only_custom_op.cpp
set(LIB_NAME hipep-custom-op-schema)
add_library(${LIB_NAME} STATIC
  schema_only_custom_op.hpp
  schema_only_custom_op.cpp
)
target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)
target_link_libraries(${LIB_NAME} PRIVATE
  onnxruntime::onnxruntime
  morphizen::morphizen-core-static
)

# Must be WHOLE_ARCHIVE-linked into the EP target: otherwise the
# StaticPluginRegister static object in schema_only_custom_op.cpp is
# referenced by nothing, so the linker drops the whole translation unit and
# registration silently becomes a no-op.
if(DEFINED morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET AND
   TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(
    ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>"
  )
else()
  message(WARNING
    "morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET not defined or not a target - "
    "skipping ${LIB_NAME} registration")
endif()

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "mlir")
