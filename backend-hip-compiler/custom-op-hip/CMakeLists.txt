##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(hip-compiler-custom-op STATIC
  HipCustomOp.cpp
  HipInferenceState.cpp
  plugin_registration.cpp
)

target_include_directories(hip-compiler-custom-op PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hip-compiler-custom-op PUBLIC
  morphizen::morphizen-core-static
)

# Link into the EP target via WHOLE_ARCHIVE so static registration runs
if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,hip-compiler-custom-op>
  )
endif()
