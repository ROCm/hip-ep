##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Generic ORT custom-op schema registration module. Adding a new op only
# requires a new ops/*_schema.cpp file added below -- no other file in this
# CMakeLists.txt or elsewhere needs to change.
set(LIB_NAME morphizen-custom-op-schema)
add_library(${LIB_NAME} STATIC
  register_custom_op.hpp
  schema_only_custom_op_base.hpp
  ops/qmoe_schema.cpp
  # ops/xxx_schema.cpp        # future ops only add a line here
)
target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)
target_link_libraries(${LIB_NAME} PRIVATE
  onnxruntime::onnxruntime
  morphizen::morphizen-core-static
)

# Must be WHOLE_ARCHIVE-linked into the EP target: otherwise the
# StaticPluginRegister static object each HIPEP_REGISTER_CUSTOM_OP
# (register_custom_op.hpp) expands to in every ops/*_schema.cpp is
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
