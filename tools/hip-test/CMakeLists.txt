##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

include(${CMAKE_SOURCE_DIR}/cmake/amdmlss_stage.cmake)

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

# LlvmIrJit (linked in via morphizen-custom-op-mlir above) dlopens the matching
# custom_kernels_<arch>.{dll,so} at JIT init using an EXPLICIT path built from
# its own module directory (thisModuleDirectory() == the hip-test binary's dir).
# An explicit-path dlopen bypasses RPATH search, so INSTALL_RPATH does NOT help
# the kernel load -- the .so must physically sit next to hip-test. The kernel
# target emits into lib/Runtime/Kernels/, so co-locate a copy beside hip-test
# after each build or the JIT load fails at run time (LlvmIrJit::create ->
# "Load(<dir>/libcustom_kernels_<arch>.so) failed"). copy_if_different is
# idempotent and a no-op when the two dirs already coincide. Skipped on mock /
# GPU-less builds, where the per-arch targets are not registered.
if(NOT BUILD_MOCK_RUNTIME AND HIP_ARCHITECTURES)
  foreach(_arch IN LISTS HIP_ARCHITECTURES)
    if(TARGET custom_kernels_${_arch})
      add_dependencies(hip-test custom_kernels_${_arch})
      add_custom_command(TARGET hip-test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:custom_kernels_${_arch}>"
          "$<TARGET_FILE_DIR:hip-test>"
        VERBATIM)
    endif()
  endforeach()
  # Separate concern from the kernel co-location above: this lets the INSTALLED
  # hip-test resolve its own DT_NEEDED shared libs (LLVM, ORT, ...) from ../lib.
  if(NOT WIN32)
    set_property(TARGET hip-test APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(hip-test PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  endif()
endif()
if(HIPDNN_EP_AMDMLSS_DLL)
  hip_ep_stage_amdmlss_dll(hip-test)
endif()
if(NOT BUILD_MOCK_RUNTIME)
  find_package(hip CONFIG REQUIRED)
  target_link_libraries(hip-test PRIVATE hip::host)
  # Signals the output-allocator callback to hand the model.dll real device
  # memory (hipMalloc) and copy it back D2H for validation. Mock builds omit
  # this and the callback returns host buffers the CPU stubs write directly.
  target_compile_definitions(hip-test PRIVATE HIPDNN_EP_LINK_HIP_HOST=1)

  # Hide static-archive symbols so hip-test's statically-linked LLVM can't clash
  # with a different shared libLLVM the HIP runtime may pull in at load time
  # (which can SIGSEGV before main). GNU ld / lld only.
  if(NOT WIN32)
    target_link_options(hip-test PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
endif()

target_include_directories(hip-test
  BEFORE PRIVATE
  ${CMAKE_SOURCE_DIR}/include   # For hip/Support/DiskFileSystem.h (must precede local/include)
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-foundation/include  # For morphizen-foundation/file_io.hpp
  ${CMAKE_SOURCE_DIR}/morphizen/morphizen-core/include         # For morphizen/plugin.hpp (native loader)
  ${CMAKE_SOURCE_DIR}/backend-mlir-compiler/custom-op-mlir/src           # For LlvmIrJit.h
)

install(TARGETS hip-test
  RUNTIME DESTINATION bin
)
