##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

add_library(hip-compiler-level1-pass STATIC
  pass_main.cpp
)

target_link_libraries(hip-compiler-level1-pass PUBLIC
  hip-compiler-lib
  morphizen::morphizen-core-static
)

# Link into the EP target via WHOLE_ARCHIVE so pass registration runs
if(TARGET ${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET})
  target_link_libraries(${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET} PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,hip-compiler-level1-pass>
  )
endif()
