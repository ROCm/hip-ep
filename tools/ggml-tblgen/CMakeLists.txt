##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# The generator runs at build time on the host, so it is an ordinary tool
# rather than part of the plugin.
add_executable(ggml-tblgen Main.cpp)
target_compile_features(ggml-tblgen PRIVATE cxx_std_17)
target_include_directories(ggml-tblgen PRIVATE ${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(_ggml_tblgen_libraries TableGen Support)
target_link_libraries(ggml-tblgen PRIVATE ${_ggml_tblgen_libraries})
if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    target_compile_options(ggml-tblgen PRIVATE /GR-)
  else()
    target_compile_options(ggml-tblgen PRIVATE -fno-rtti)
  endif()
endif()
