##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

option(HIP_EP_GENERATE_TQ_CODEC "Generate experimental TurboQuant codec assets on CPU" OFF)
if(NOT HIP_EP_GENERATE_TQ_CODEC)
  return()
endif()

# Optional because ordinary HIP-EP builds must not require Torch/SciPy/NumPy.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(HIP_EP_TQ_CODEC_DIR "${CMAKE_BINARY_DIR}/turboquant" CACHE PATH
    "Generated TurboQuant codec and reference-array directory")
set(_tq_generator "${CMAKE_CURRENT_SOURCE_DIR}/generate_codec.py")
add_custom_command(
  OUTPUT "${HIP_EP_TQ_CODEC_DIR}/codec.bin"
         "${HIP_EP_TQ_CODEC_DIR}/constants.npz"
         "${HIP_EP_TQ_CODEC_DIR}/manifest.json"
  COMMAND "${Python3_EXECUTABLE}" "${_tq_generator}"
          --output-dir "${HIP_EP_TQ_CODEC_DIR}"
  DEPENDS "${_tq_generator}" "${CMAKE_CURRENT_SOURCE_DIR}/write_codec.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/reference.py"
  COMMENT "Generating TurboQuant QR codec on CPU"
  VERBATIM
)
add_custom_target(turboquant-codec ALL
  DEPENDS "${HIP_EP_TQ_CODEC_DIR}/codec.bin"
          "${HIP_EP_TQ_CODEC_DIR}/constants.npz"
          "${HIP_EP_TQ_CODEC_DIR}/manifest.json")
install(FILES "${HIP_EP_TQ_CODEC_DIR}/codec.bin" "${HIP_EP_TQ_CODEC_DIR}/manifest.json"
        DESTINATION share/hip-ep/turboquant COMPONENT turboquant-codec)
