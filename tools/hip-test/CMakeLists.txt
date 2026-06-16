##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-test loads a per-model artifact -- LLVM bitcode via LlvmIrJit or a
# native .dll/.so via morphizen::Plugin, selected by file extension (the same
# two loaders the EP uses at runtime) -- resolves the inference_* entry points,
# runs the model end-to-end, and optionally validates outputs.
add_executable(hip-test
  hip-test.cpp
)

# LLVM Support is still needed for JSON parsing of the embedded metadata.
# The heavy ORC stack comes in transitively via morphizen-custom-op-mlir.
llvm_map_components_to_libnames(llvm_libs Support)

target_link_libraries(hip-test
  PRIVATE
  morphizen-custom-op-mlir       # Provides LlvmIrJit and its LLVM ORC deps.
  morphizen::morphizen-core-static # morphizen::Plugin (native .dll/.so loader)
  HipSupport
  ${llvm_libs}
)

# LlvmIrJit (linked in via morphizen-custom-op-mlir above) dlopens the
# matching custom_kernels_<arch>.{dll,so} at JIT init, anchoring the
# lookup on the containing module's directory. So hip-test only needs
# the per-arch DLLs co-located with hip-test.exe (which the kernel
# install rule already arranges on Windows; Linux needs $ORIGIN/../lib in
# the RPATH because the .so installs to lib/ while the exe installs to
# bin/). Skipped on mock / GPU-less builds, where the per-arch targets
# are not registered.
if(NOT BUILD_MOCK_RUNTIME AND HIP_ARCHITECTURES)
  foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(TARGET custom_kernels_${_arch})
      add_dependencies(hip-test custom_kernels_${_arch})
    endif()
  endforeach()
  if(NOT WIN32)
    set_property(TARGET hip-test APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(hip-test PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
endif()
if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip CONFIG REQUIRED)
  target_link_libraries(hip-test PRIVATE hip::host)
endif()

target_include_directories(hip-test
  BEFORE PRIVATE
  ${CMAKE_SOURCE_DIR}/include   # For hip/Support/DiskFileSystem.h (must precede local/include)
  ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-foundation/include  # For morphizen-foundation/file_io.hpp
  ${CMAKE_SOURCE_DIR}/3rd-party/morphizen/morphizen-core/include         # For morphizen/plugin.hpp (native loader)
  ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src           # For LlvmIrJit.h
)

install(TARGETS hip-test
  RUNTIME DESTINATION bin
)
