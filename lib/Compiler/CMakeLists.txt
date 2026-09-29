##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building Compiler Library")

# Create compiler library
add_library(LibHipCompiler STATIC
  CompilerDriver.cpp
  PluginRegistry.cpp
  # Static compiler-plugin dispatch. Includes the CMake-generated
  # ${CMAKE_BINARY_DIR}/include/hip/Compiler/StaticLinkedPlugins.inc (see
  # cmake/HipEpPlugins.cmake); empty => dispatch is a no-op.
  StaticPlugins.cpp
)

# rocMLIR kernel compilation. Attached here rather than to
# hip-rocmlir-compiler so that everything linking LibHipCompiler -- including
# hipgpu.dll, which picks it up transitively through morphizen-core-static --
# can compile rock.kernel funcs in process. Without it the EP can outline
# kernels but never fills in hip.rocmlir's kernel_binary, and every dispatch
# fails at hipModuleLoadData.
if(ENABLE_ROCMLIRTRITON)
  target_sources(LibHipCompiler PRIVATE RocMlirKernelCompiler.cpp)
  # rocMLIR/Triton's per-library static archives have circular dependencies,
  # so a single-pass link leaves undefined symbols. GNU/BSD-style linkers need
  # an explicit RESCAN group; MSVC-style linkers (link.exe / lld-link) rescan
  # automatically and do not support the feature.
  if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_rocmlir_link_closure "$<LINK_GROUP:RESCAN,${HIPDNN_ROCMLIR_LINK_LIBS}>")
  else()
    set(_rocmlir_link_closure "${HIPDNN_ROCMLIR_LINK_LIBS}")
  endif()
  target_link_libraries(LibHipCompiler PUBLIC
    HipToTosa
    # rocMLIR/Triton reference this MLIR extension (registerInlinerExtension)
    # but it is a leaf target not pulled in transitively by the rock/triton
    # closure.
    MLIRFuncInlinerExtension
    # CompilerDriver's rocMLIR arm calls createDuplicateFunctionEliminationPass.
    # It resolves transitively today, but through the rocMLIR closure -- name
    # it so a standard MLIR dependency does not hinge on that.
    MLIRFuncTransforms
    "${_rocmlir_link_closure}"
  )
  # Same name as the CMake option deliberately: one flag for the whole
  # feature, rather than a separate macro that can drift out of sync with it.
  target_compile_definitions(LibHipCompiler PUBLIC ENABLE_ROCMLIRTRITON=1)

  # resolveRocMlirArch asks the live GPU which chip to build for. Without this
  # it can only guess, and a code object for the wrong chip fails at module
  # load. Same gate the rest of the EP uses for HIP host APIs.
  #
  # Headers only -- deliberately NOT linking hip::host. This library also ends
  # up in hip-compiler and hip-rocmlir-compiler, which only compile, and a
  # load-time import of the HIP runtime makes those tools unable to start at all
  # where no GPU driver is installed. resolveRocMlirArch loads the runtime on
  # demand instead and falls back to the build's target architecture, so pass it
  # the real library name rather than hardcoding a versioned one (Windows ships
  # amdhip64_<major>.dll). Without a name the device query is compiled out.
  if(HIPDNN_EP_LINK_HIP_HOST)
    find_package(hip CONFIG REQUIRED)
    target_include_directories(LibHipCompiler SYSTEM PRIVATE
      $<TARGET_PROPERTY:hip::host,INTERFACE_INCLUDE_DIRECTORIES>)
    # Definitions as well as includes: the HIP headers #error out unless
    # __HIP_PLATFORM_AMD__ is set, and hip::host is what normally supplies it.
    target_compile_definitions(LibHipCompiler PRIVATE
      $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_DEFINITIONS>)
    target_compile_definitions(LibHipCompiler PRIVATE HIPDNN_EP_LINK_HIP_HOST=1)
    set(_hip_runtime_name "")
    if(TARGET hip::amdhip64)
      get_target_property(_hip_runtime_cfgs hip::amdhip64 IMPORTED_CONFIGURATIONS)
      foreach(_cfg IN LISTS _hip_runtime_cfgs)
        get_target_property(_hip_runtime_loc hip::amdhip64 IMPORTED_LOCATION_${_cfg})
        if(_hip_runtime_loc)
          get_filename_component(_hip_runtime_name "${_hip_runtime_loc}" NAME)
          break()
        endif()
      endforeach()
    endif()
    if(_hip_runtime_name)
      target_compile_definitions(LibHipCompiler PRIVATE
        "HIPDNN_HIP_RUNTIME_LIB=\"${_hip_runtime_name}\"")
    else()
      message(STATUS "[onnx-hipdnn-ep] HIP runtime library name unknown; "
                     "rocMLIR arch will come from HIP_ARCHITECTURES")
    endif()
  endif()
  # Last resort when there is no device to ask: whatever this build targets.
  if(HIP_ARCHITECTURES)
    list(GET HIP_ARCHITECTURES 0 _rocmlir_default_arch)
    target_compile_definitions(LibHipCompiler PRIVATE
      "HIPDNN_ROCMLIR_DEFAULT_ARCH=\"${_rocmlir_default_arch}\"")
  endif()
endif()

# When hipBLASLt is disabled, the model link must not add -lhipblaslt
# (CompilerDriver gates that behind this define).
if(HIPDNN_EP_DISABLE_VENDOR_BLAS)
  target_compile_definitions(LibHipCompiler PRIVATE HIPDNN_EP_DISABLE_VENDOR_BLAS)
endif()

# Link against required libraries
target_link_libraries(LibHipCompiler PUBLIC
  HipConversion
  HipDialectIR
  HipsrDialectIR
  HipsrDialectTransforms
  HipsrDialectPipelines
  HipDialectTransforms
  HipTargetLLVM
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFToControlFlow
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRIndexToLLVM
  MLIRControlFlowToLLVM
  MLIRUBToLLVM
  MLIRComplexToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  HipSchemasLib
)

# Link LLVM components
llvm_map_components_to_libnames(compiler_llvm_libs
  Support
  Core
  IRReader
  Passes
  Target
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
)

target_link_libraries(LibHipCompiler PUBLIC ${compiler_llvm_libs})

target_include_directories(LibHipCompiler PUBLIC
  ${CMAKE_BINARY_DIR}/lib
  ${CMAKE_BINARY_DIR}/schemas
  ${CMAKE_SOURCE_DIR}/3rd-party/onnx-mlir
  ${CMAKE_BINARY_DIR}/3rd-party/onnx-mlir
  # For StaticPlugins.cpp's #include "hip/Compiler/StaticLinkedPlugins.inc"
  # (generated by cmake/HipEpPlugins.cmake at configure time).
  ${CMAKE_BINARY_DIR}/include
)

# Native artifact path (CompilationOptions.output_mode == NATIVE):
# CompilerDriver::discoverLibraries links the per-arch custom-kernels import
# library (custom_kernels_<arch>.{lib,so}) into the generated model DLL. Tell
# it where that lib lives (build-tree dir of the first configured arch) and
# which arch to link by default; both are overridable at runtime via the
# HIP_CUSTOM_KERNELS_DIR / HIP_CUSTOM_KERNELS_ARCH env vars.
if(HIP_ARCHITECTURES)
    list(GET HIP_ARCHITECTURES 0 _ck_first_arch)
    if(NOT BUILD_MOCK_RUNTIME AND TARGET custom_kernels_${_ck_first_arch})
        target_compile_definitions(LibHipCompiler PRIVATE
            "HIPDNN_CK_LIB_DIR=\"$<TARGET_LINKER_FILE_DIR:custom_kernels_${_ck_first_arch}>\""
            "HIPDNN_CK_DEFAULT_ARCH=\"${_ck_first_arch}\""
        )
        message(STATUS
            "Custom kernels (native link): arch=${_ck_first_arch}, "
            "dir=$<TARGET_LINKER_FILE_DIR:custom_kernels_${_ck_first_arch}>")
    endif()
endif()

# Native current-stream accessor (tls_stream.cpp). runtime.bc references but no
# longer defines hipdnn_ep_get/set_current_stream, so every native model DLL
# must link this static lib. Point the driver at its build-tree location.
if(TARGET hipdnn-ep-tls)
    target_compile_definitions(LibHipCompiler PRIVATE
        "HIPDNN_EP_TLS_LIB_DIR=\"$<TARGET_LINKER_FILE_DIR:hipdnn-ep-tls>\""
        "HIPDNN_EP_TLS_LIB_NAME=\"$<TARGET_LINKER_FILE_BASE_NAME:hipdnn-ep-tls>\""
    )
endif()

# OnnxToHipDNN provides the OutlineOnnxToHipDNN / CompileHipDNNGraphs passes,
# which CompilerDriver uses when a hipDNN handle is set. Gated on the runtime
# target so a hipDNN-less build (no TheRock SDK) still configures.
# The native path also links hipdnn_graph_runtime into the model DLL; pass its
# import-library path via a generator expression.
if(TARGET hipdnn_graph_runtime)
    target_compile_definitions(LibHipCompiler PRIVATE
        "HIPDNN_GRAPH_RUNTIME_LIB_PATH=\"$<TARGET_LINKER_FILE:hipdnn_graph_runtime>\""
    )
    target_link_libraries(LibHipCompiler PUBLIC OnnxToHipDNN)
endif()

# Set folder for IDE organization
set_target_properties(LibHipCompiler PROPERTIES FOLDER "mlir/Compiler")
