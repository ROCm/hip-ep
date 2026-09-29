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
  # RocMlirAutotune.cpp replaces the default perfConfig choice with a
  # benchmarked winner. It sits here rather than in hip-rocmlir-compiler for
  # the same reason RocMlirKernelCompiler.cpp does: otherwise the EP can only
  # ever compile position 0 of the tuning space. It resolves HIP at call time
  # (see the file header), so it adds no import-table dependency.
  target_sources(LibHipCompiler PRIVATE
    AdaptiveBenchmark.cpp
    RocMlirKernelCompiler.cpp
    RocMlirAutotune.cpp
    ${CMAKE_SOURCE_DIR}/tools/problem-cache/nearest_conv_tune.cpp
  )
  target_include_directories(LibHipCompiler PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/problem-cache)
  # Benchmarking calls HIP host APIs, so the file compiles to a stub unless
  # this build both permits them and knows the runtime library name to resolve
  # them from. Published rather than left implicit because hip-rocmlir-compiler
  # has to key its own autotune flag off what was actually built here: keying
  # it off BUILD_MOCK_RUNTIME let the tool offer tuning the library had stubbed
  # out, then fail as though the machine had no GPU.
  set(HIPDNN_EP_ROCMLIR_AUTOTUNE_AVAILABLE OFF CACHE INTERNAL
      "RocMlirAutotune.cpp compiled its real implementation, not its stub")
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
  )
  # Triton libraries are OBJECT libraries whose interface publishes
  # $<TARGET_OBJECTS>. Putting those objects into this STATIC library makes
  # CMake re-export them, so every WHOLE_ARCHIVE'd plugin that links
  # LibHipCompiler archives its own copy and the final link also gets the
  # loose objects (LNK2005 / multiple definition). Keep the objects off this
  # archive. Final binaries add them once (see hip_link_rocmlir_closure).
  # The non-object closure is also linked only from those binaries, and on
  # GNU ld it has to come after this archive: a RESCAN group placed earlier
  # is not searched again for symbols this archive introduces.
  set(_rocmlir_object_targets "")
  set(_rocmlir_static "")
  foreach(_lib IN LISTS HIPDNN_ROCMLIR_LINK_LIBS)
    if(NOT TARGET "${_lib}")
      continue()
    endif()
    get_target_property(_rocmlir_type "${_lib}" TYPE)
    if(_rocmlir_type STREQUAL "OBJECT_LIBRARY")
      list(APPEND _rocmlir_object_targets "${_lib}")
    else()
      list(APPEND _rocmlir_static "${_lib}")
    endif()
    target_include_directories(LibHipCompiler SYSTEM PRIVATE
      $<TARGET_PROPERTY:${_lib},INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(LibHipCompiler PRIVATE
      $<TARGET_PROPERTY:${_lib},INTERFACE_COMPILE_DEFINITIONS>)
  endforeach()
  set(HIPDNN_ROCMLIR_OBJECT_LINK_LIBS "${_rocmlir_object_targets}" CACHE INTERNAL
      "Triton OBJECT libraries added once, on final binaries only" FORCE)
  set(HIPDNN_ROCMLIR_STATIC_LINK_LIBS "${_rocmlir_static}" CACHE INTERNAL
      "Non-object rocMLIR/Triton libraries linked by final binaries" FORCE)
  # RocMlirKernelCompiler.cpp includes InitRocMLIRDialects.h, which includes
  # generated Triton/Proton Passes.h.inc headers. Those files are produced by
  # the object libraries' tablegen targets. Without this edge ninja compiles
  # this library before the .inc exists.
  if(_rocmlir_object_targets)
    add_dependencies(LibHipCompiler ${_rocmlir_object_targets})
  endif()
  # TritonToHsaco dlopens this library by name at kernel-compile time. The
  # link closure does not pull the custom target in, so depend on it directly.
  if(TARGET triton_amd_codegen_lib)
    add_dependencies(LibHipCompiler triton_amd_codegen_lib)
  endif()
  # Same name as the CMake option deliberately: one flag for the whole
  # feature, rather than a separate macro that can drift out of sync with it.
  target_compile_definitions(LibHipCompiler PUBLIC ENABLE_ROCMLIRTRITON=1)

  # InitRocMLIRDialects.h pulls in Triton headers, and Triton is C++20
  # (std::set::contains in triton/Tools/Sys/GetEnv.h). The rest of hip-ep stays
  # on C++17; this is the library that includes those headers.
  set_target_properties(LibHipCompiler PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES)

  # resolveRocMlirArch asks the live GPU which chip to build for; a code object
  # for the wrong chip fails at module load. Headers only, no hip::host link:
  # hip-compiler links this library but only compiles, and a load-time HIP
  # import stops it starting where no driver is installed.
  if(HIPDNN_EP_LINK_HIP_HOST)
    find_package(hip CONFIG REQUIRED)
    target_include_directories(LibHipCompiler SYSTEM PRIVATE
      $<TARGET_PROPERTY:hip::host,INTERFACE_INCLUDE_DIRECTORIES>)
    # Definitions as well as includes: the HIP headers #error out unless
    # __HIP_PLATFORM_AMD__ is set, and hip::host is what normally supplies it.
    target_compile_definitions(LibHipCompiler PRIVATE
      $<TARGET_PROPERTY:hip::host,INTERFACE_COMPILE_DEFINITIONS>)
    target_compile_definitions(LibHipCompiler PRIVATE HIPDNN_EP_LINK_HIP_HOST=1)
    target_link_libraries(LibHipCompiler PRIVATE ${CMAKE_DL_LIBS})
    # Real library name for the on-demand load, not a literal: Windows ships a
    # version-suffixed amdhip64_<major>.dll. Unknown means no device query.
    set(_hip_runtime_name "")
    if(TARGET hip::amdhip64)
      set(_hip_runtime_loc "")
      get_target_property(_hip_runtime_cfgs hip::amdhip64 IMPORTED_CONFIGURATIONS)
      foreach(_cfg IN LISTS _hip_runtime_cfgs)
        get_target_property(_hip_runtime_loc hip::amdhip64 IMPORTED_LOCATION_${_cfg})
        if(_hip_runtime_loc)
          break()
        endif()
      endforeach()
      # IMPORTED_CONFIGURATIONS is optional, and a package that sets only the
      # unqualified location would otherwise lose the device query entirely.
      if(NOT _hip_runtime_loc)
        get_target_property(_hip_runtime_loc hip::amdhip64 IMPORTED_LOCATION)
      endif()
      if(_hip_runtime_loc)
        get_filename_component(_hip_runtime_name "${_hip_runtime_loc}" NAME)
      endif()
    endif()
    if(_hip_runtime_name)
      target_compile_definitions(LibHipCompiler PRIVATE
        "HIPDNN_HIP_RUNTIME_LIB=\"${_hip_runtime_name}\"")
      # Matches RocMlirAutotune.cpp's guard: HIPDNN_EP_LINK_HIP_HOST for the
      # headers, HIPDNN_HIP_RUNTIME_LIB for the call-time lookup.
      set(HIPDNN_EP_ROCMLIR_AUTOTUNE_AVAILABLE ON CACHE INTERNAL
          "RocMlirAutotune.cpp compiled its real implementation, not its stub")
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

# Final binaries take the Triton objects and the non-object closure. Static
# plugins must not: a second copy in a WHOLE_ARCHIVE input duplicates the
# objects in hipgpu.dll (LNK2005 / multiple definition).
function(hip_link_rocmlir_closure target)
  if(NOT ENABLE_ROCMLIRTRITON)
    return()
  endif()
  # One copy of each Triton object, on this binary only. Not archived into
  # LibHipCompiler, so plugins that link that library do not absorb them.
  # $<TARGET_OBJECTS> does not bring in the object library's link interface.
  # Those edges name the static dialect archives the objects call
  # (MLIRNVVMDialect, MLIRROCDLDialect, MLIRGPUToROCDLTransforms, ...).
  # Without them GNU ld reports undefined references and stops.
  set(_rocmlir_object_ifaces "")
  foreach(_lib IN LISTS HIPDNN_ROCMLIR_OBJECT_LINK_LIBS)
    if(NOT TARGET "${_lib}")
      continue()
    endif()
    target_sources("${target}" PRIVATE "$<TARGET_OBJECTS:${_lib}>")
    get_target_property(_iface "${_lib}" INTERFACE_LINK_LIBRARIES)
    if(NOT _iface)
      continue()
    endif()
    foreach(_dep IN LISTS _iface)
      if(_dep MATCHES "^\\$<LINK_ONLY:(.+)>$")
        set(_dep "${CMAKE_MATCH_1}")
      endif()
      if(_dep MATCHES "^\\$<" OR NOT TARGET "${_dep}")
        continue()
      endif()
      get_target_property(_type "${_dep}" TYPE)
      if(_type STREQUAL "STATIC_LIBRARY")
        list(APPEND _rocmlir_object_ifaces "${_dep}")
      endif()
    endforeach()
  endforeach()
  if(_rocmlir_object_ifaces)
    list(REMOVE_DUPLICATES _rocmlir_object_ifaces)
    target_link_libraries("${target}" PRIVATE ${_rocmlir_object_ifaces})
  endif()
  if("${HIPDNN_ROCMLIR_STATIC_LINK_LIBS}" STREQUAL "")
    return()
  endif()
  # rocMLIR's static archives have circular dependencies. LibHipCompiler,
  # linked above, is what references them. MSVC rescans on its own. GNU ld
  # does not, so the archives go in a RESCAN group that is the last link
  # item. A group emitted with the link flags is scanned before
  # LibHipCompiler and never revisited. The group is only these archives:
  # folding the rest of the link into it makes a CMake dependency cycle
  # through the Triton object libraries.
  if(CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_rocmlir_group_deps "")
    # LibHipCompiler is in the group with the rock archives. GNU ld only
    # rescans inside the group, and CMake places the group before that
    # archive on the shared-library link line, so a group of the rock
    # archives alone is scanned before LibHipCompiler's references exist.
    foreach(_lib IN LISTS HIPDNN_ROCMLIR_STATIC_LINK_LIBS)
      if(TARGET "${_lib}")
        list(APPEND _rocmlir_group_deps "${_lib}")
      endif()
    endforeach()
    if(TARGET LibHipCompiler)
      list(APPEND _rocmlir_group_deps LibHipCompiler)
    endif()
    if(_rocmlir_group_deps)
      list(JOIN _rocmlir_group_deps "," _rocmlir_group)
      target_link_libraries("${target}" PRIVATE
        "$<LINK_GROUP:RESCAN,${_rocmlir_group}>")
    endif()
  else()
    target_link_libraries("${target}" PRIVATE ${HIPDNN_ROCMLIR_STATIC_LINK_LIBS})
  endif()
endfunction()

function(hip_rocmlir_apply_includes target)
  if(NOT ENABLE_ROCMLIRTRITON)
    return()
  endif()
  foreach(_lib IN LISTS HIPDNN_ROCMLIR_LINK_LIBS)
    if(NOT TARGET "${_lib}")
      continue()
    endif()
    target_include_directories("${target}" SYSTEM PRIVATE
      $<TARGET_PROPERTY:${_lib},INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions("${target}" PRIVATE
      $<TARGET_PROPERTY:${_lib},INTERFACE_COMPILE_DEFINITIONS>)
  endforeach()
endfunction()

# The EP target is created later, under morphizen. Attach the closure once the
# top-level directory has finished configuring.
function(hip_link_rocmlir_into_ep)
  if(TARGET "${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}")
    hip_link_rocmlir_closure("${morphizen_ONNXRUNTIME_MORPHIZEN_EP_TARGET}")
  endif()
endfunction()
if(ENABLE_ROCMLIRTRITON)
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL hip_link_rocmlir_into_ep)
endif()

# Set folder for IDE organization
set_target_properties(LibHipCompiler PROPERTIES FOLDER "mlir/Compiler")
