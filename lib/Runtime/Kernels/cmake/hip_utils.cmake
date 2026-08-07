##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.18)

# Prevent multiple inclusion
if(DEFINED _HIP_UTILS_INCLUDED)
    return()
endif()
set(_HIP_UTILS_INCLUDED TRUE)

# Path to the MSVC 14.51 <cmath> compatibility shim (see _hip_compile_sources).
# Resolved at include time so it doesn't depend on CMAKE_CURRENT_SOURCE_DIR at
# call time -- hip_add_library can be invoked from any subdir.
get_filename_component(_HIP_UTILS_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
set(_MSVC_HIP_CMATH_WORKAROUND_HEADER
    "${_HIP_UTILS_DIR}/../include/msvc_hip_cmath_workaround.h"
    CACHE INTERNAL "MSVC 14.51 <cmath> shim for HIP compilation")

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

function(_hip_root_has_toolchain root out_var)
    if(NOT root)
        set(${out_var} FALSE PARENT_SCOPE)
        return()
    endif()
    if(WIN32)
        if(EXISTS "${root}/bin/hipcc.exe" OR EXISTS "${root}/bin/hipcc.bat")
            set(${out_var} TRUE PARENT_SCOPE)
        else()
            set(${out_var} FALSE PARENT_SCOPE)
        endif()
    else()
        if(EXISTS "${root}/bin/hipcc" OR EXISTS "${root}/bin/amdclang++")
            set(${out_var} TRUE PARENT_SCOPE)
        else()
            set(${out_var} FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Resolve the ROCm/HIP root. Prefer an explicit, valid THEROCK_DIST over a
# stale HIP_PATH/ROCM_PATH env var (common on Windows dev boxes where
# C:/opt/rocm/7.1 is listed in the environment but not actually installed).
set(_hip_root_candidates "")
if(DEFINED THEROCK_DIST AND NOT "${THEROCK_DIST}" STREQUAL "")
    list(APPEND _hip_root_candidates "${THEROCK_DIST}")
endif()
if(DEFINED ENV{THEROCK_DIST} AND NOT "$ENV{THEROCK_DIST}" STREQUAL "")
    list(APPEND _hip_root_candidates "$ENV{THEROCK_DIST}")
endif()
if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
    list(APPEND _hip_root_candidates "$ENV{HIP_PATH}")
endif()
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    list(APPEND _hip_root_candidates "$ENV{ROCM_PATH}")
endif()
list(APPEND _hip_root_candidates "/opt/rocm")

set(HIP_PATH "")
foreach(_candidate IN LISTS _hip_root_candidates)
    _hip_root_has_toolchain("${_candidate}" _hip_root_ok)
    if(_hip_root_ok)
        set(HIP_PATH "${_candidate}")
        break()
    endif()
endforeach()

if(NOT HIP_PATH)
    message(FATAL_ERROR
        "Could not locate a ROCm/HIP SDK with hipcc.\n"
        "Set -DTHEROCK_DIST=/path/to/therock or point HIP_PATH at a valid install.")
endif()

# GPU architectures - must be set by developer
set(HIP_ARCHITECTURES "" CACHE STRING
    "Target GPU architectures (e.g., gfx1100;gfx1102). Required for HIP compilation.")

# Global compile options
set(HIP_COMPILE_OPTIONS "" CACHE STRING "Additional compile options for HIP sources")

# Add HIP paths to CMAKE_PREFIX_PATH for find_package
list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake")

#------------------------------------------------------------------------------
# Platform Setup
#------------------------------------------------------------------------------

# Set HIP_PLATFORM before find_package(hip) - required by hip-config.cmake
# The hip-config.cmake checks this variable to determine AMD vs NVIDIA platform
if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
endif()

# Try to find HIP package (provides hip::host target for host-side code)
set(_hip_utils_resolved_root "${HIP_PATH}")
find_package(hip QUIET)
# hip-config-amd.cmake prefers ENV{HIP_PATH} on Windows even when a valid
# THEROCK_DIST prefix was found; keep the validated root for hipcc compilation.
set(HIP_PATH "${_hip_utils_resolved_root}")

if(WIN32)
    # Find hipcc compiler (needed for .hip device code compilation)
    if(HIPCC_EXECUTABLE AND NOT EXISTS "${HIPCC_EXECUTABLE}")
        unset(HIPCC_EXECUTABLE CACHE)
    endif()
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc.exe hipcc hipcc.bat
        PATHS "${HIP_PATH}/bin"
        NO_DEFAULT_PATH
    )
    if(NOT HIPCC_EXECUTABLE OR NOT EXISTS "${HIPCC_EXECUTABLE}")
        message(FATAL_ERROR "Could not find hipcc in ${HIP_PATH}/bin")
    endif()

    # Set paths (used as fallback if hip::host not available)
    set(HIP_INCLUDE_DIR "${HIP_PATH}/include")
    set(HIP_LIBRARY_DIR "${HIP_PATH}/lib")
    set(HIP_RUNTIME_LIBRARY "${HIP_LIBRARY_DIR}/amdhip64.lib")

    # TheRock packs device bitcode under lib/llvm/amdgcn/bitcode; stock ROCm
    # uses lib/clang/<ver>/lib/amdgcn/bitcode. Teach hipcc where to look.
    set(HIP_DEVICE_LIB_PATH "")
    if(EXISTS "${HIP_PATH}/lib/llvm/amdgcn/bitcode")
        set(HIP_DEVICE_LIB_PATH "${HIP_PATH}/lib/llvm/amdgcn/bitcode")
    else()
        file(GLOB _hip_clang_lib_dirs "${HIP_PATH}/lib/clang/*/lib/amdgcn/bitcode")
        if(_hip_clang_lib_dirs)
            list(GET _hip_clang_lib_dirs 0 HIP_DEVICE_LIB_PATH)
        endif()
    endif()
    set(HIP_ROCM_COMPILE_FLAGS "--rocm-path=${HIP_PATH}")
    if(HIP_DEVICE_LIB_PATH)
        list(APPEND HIP_ROCM_COMPILE_FLAGS
            "--rocm-device-lib-path=${HIP_DEVICE_LIB_PATH}")
    endif()

    message(STATUS "[hip_utils] HIP_PATH: ${HIP_PATH}")
    if(HIP_DEVICE_LIB_PATH)
        message(STATUS "[hip_utils] device libs: ${HIP_DEVICE_LIB_PATH}")
    else()
        message(WARNING "[hip_utils] ROCm device bitcode not found under ${HIP_PATH}")
    endif()
    message(STATUS "[hip_utils] hipcc: ${HIPCC_EXECUTABLE}")
    message(STATUS "[hip_utils] HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")
    if(TARGET hip::host)
        message(STATUS "[hip_utils] hip::host target available - using imported target")
    else()
        message(STATUS "[hip_utils] hip::host not available - using manual HIP setup")
    endif()
else()
    # Linux HIP language setup.
    #
    # cmake 3.31's CMakeDetermineHIPCompiler resolves the ROCm root via
    # CMAKE_HIP_COMPILER_ROCM_ROOT, then a `clang++ -v` stderr-scrape, then
    # `hipconfig --rocmpath` — it ignores ENV{ROCM_PATH} / ENV{HIP_PATH}.
    # When ROCm lives at a non-default prefix (TheRock dist tarball, vcpkg,
    # ...) and isn't on PATH, the latter two routes fail. Pin the root
    # directly + point CMAKE_HIP_COMPILER at the in-tree clang++.
    if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${HIP_PATH}")
    endif()
    if(NOT CMAKE_HIP_COMPILER)
        # TheRock: <root>/llvm/bin; stock ROCm: <root>/bin. Prefer amdclang++.
        find_program(_HIP_CLANGXX
            NAMES amdclang++ clang++
            PATHS
                "${HIP_PATH}/bin"
                "${HIP_PATH}/llvm/bin"
            NO_DEFAULT_PATH
        )
        if(_HIP_CLANGXX)
            set(CMAKE_HIP_COMPILER "${_HIP_CLANGXX}")
            message(STATUS "[hip_utils] CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")
        else()
            message(FATAL_ERROR
                "Could not find amdclang++ or clang++ under "
                "${HIP_PATH}/bin or ${HIP_PATH}/llvm/bin. "
                "Set CMAKE_HIP_COMPILER explicitly or install a complete "
                "ROCm/TheRock distribution at HIP_PATH.")
        endif()
    endif()
    message(STATUS "[hip_utils] CMAKE_HIP_COMPILER_ROCM_ROOT: ${CMAKE_HIP_COMPILER_ROCM_ROOT}")

    enable_language(HIP)
    if(NOT hip_FOUND)
        find_package(hip REQUIRED)
    endif()

    set(HIP_INCLUDE_DIR "${hip_INCLUDE_DIRS}")

    message(STATUS "[hip_utils] Found HIP: ${hip_VERSION}")
    message(STATUS "[hip_utils] HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")
endif()

#------------------------------------------------------------------------------
# Internal: Get build-type-specific flags for a specific configuration
#------------------------------------------------------------------------------
function(_hip_get_build_flags_for_config CONFIG OUTPUT_VAR)
    if(CONFIG STREQUAL "Debug")
        set(${OUTPUT_VAR} -D_DEBUG -D_ITERATOR_DEBUG_LEVEL=2 -O0 -g PARENT_SCOPE)
    elseif(CONFIG STREQUAL "RelWithDebInfo")
        set(${OUTPUT_VAR} -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -O2 -g PARENT_SCOPE)
    else()
        # Release, MinSizeRel, or default
        set(${OUTPUT_VAR} -DNDEBUG -D_ITERATOR_DEBUG_LEVEL=0 -O2 PARENT_SCOPE)
    endif()
endfunction()

#------------------------------------------------------------------------------
# Internal: Get build-type-specific flags (for single-config generators)
#------------------------------------------------------------------------------
function(_hip_get_build_flags OUTPUT_VAR)
    set(build_type "${CMAKE_BUILD_TYPE}")
    if(NOT build_type)
        set(build_type "Debug")
    endif()
    _hip_get_build_flags_for_config(${build_type} flags)
    set(${OUTPUT_VAR} ${flags} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Internal: Check if using multi-config generator
#------------------------------------------------------------------------------
function(_hip_is_multi_config OUTPUT_VAR)
    get_property(is_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    set(${OUTPUT_VAR} ${is_multi} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Internal: Build architecture flags
#
# ARCH_LIST overrides the global HIP_ARCHITECTURES when non-empty so a single
# configure can build N per-arch shared libraries each with one fatbin slice.
#------------------------------------------------------------------------------
function(_hip_get_arch_flags ARCH_LIST OUTPUT_VAR)
    set(arch_flags "")
    if(NOT ARCH_LIST)
        set(ARCH_LIST ${HIP_ARCHITECTURES})
    endif()
    foreach(arch ${ARCH_LIST})
        list(APPEND arch_flags "--offload-arch=${arch}")
    endforeach()
    set(${OUTPUT_VAR} ${arch_flags} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Internal: Compile HIP sources to object files (Windows only)
# For multi-config generators, creates per-config object files.
#------------------------------------------------------------------------------
function(_hip_compile_sources TARGET_NAME HIP_SOURCES INCLUDE_DIRS COMPILE_OPTS ARCH_LIST OUTPUT_OBJS)
    _hip_get_arch_flags("${ARCH_LIST}" arch_flags)
    if(DEFINED HIP_ROCM_COMPILE_FLAGS)
        list(APPEND arch_flags ${HIP_ROCM_COMPILE_FLAGS})
    endif()

    # Build include flags
    # NOTE: -I and path are separate list items to handle paths with spaces
    # (e.g., "C:/Program Files/..."). Clang supports "-I" "<path>" as two args.
    set(include_flags "")
    list(APPEND include_flags "-I" "${HIP_INCLUDE_DIR}")
    foreach(dir ${INCLUDE_DIRS})
        list(APPEND include_flags "-I" "${dir}")
    endforeach()

    # Flip HIP_KERNEL_API to `__declspec(dllexport)` for the per-arch kernel
    # DLL build (see lib/Runtime/Kernels/include/hip_custom_kernels.h).
    # Harmless on static libs / executables -- nothing in this tree ships
    # the kernel symbols outside a SHARED `custom_kernels_<arch>` build.
    set(define_flags "-DHIP_CUSTOM_KERNELS_EXPORTS")

    # MSVC ABI compatibility flags
    set(abi_flags
        -fms-extensions
        -fms-compatibility
        -fexceptions
    )

    # MSVC <cmath> + clang-HIP `<cmath>`/`__clang_cuda_math_forward_declares.h`
    # incompatibility -- two-layer fix.
    #
    # MSVC 14.51's `<cmath>` adds `constexpr` overloads of `isless`,
    # `islessequal`, `isgreater`, `isgreaterequal`, `islessgreater`,
    # `isunordered`, `isfinite`, `isinf`, `isnan`, `isnormal` (see
    # microsoft/STL PR #4612).  In clang's HIP/CUDA mode every `constexpr`
    # function with no explicit annotation is implicitly `__host__ __device__`,
    # so these overloads collide with the `__device__`-only declarations in
    # clang-hip's bundled `__clang_cuda_math_forward_declares.h`:
    #
    #   error: __device__ function 'isgreater' cannot overload
    #          __host__ __device__ function 'isgreater'
    #
    # Layer 1 -- the canonical clang fix the diagnostic itself recommends:
    # `-fno-cuda-host-device-constexpr` stops clang from implicitly marking
    # unannotated `constexpr` functions as `__host__ __device__`, so MSVC's
    # `<cmath>` overloads become host-only and the conflict disappears.
    #
    # CAVEAT: this also makes STL `constexpr` member functions that previously
    # worked from device code -- notably `std::numeric_limits<T>::lowest()`,
    # `::infinity()`, `::quiet_NaN()` -- host-only.  Device-side call sites in
    # this tree have been migrated to the equivalent C macros (`INT32_MIN` /
    # `INT64_MIN` from <cstdint>, `INFINITY` / `NAN` from <cmath>), which are
    # not `constexpr` functions and so are unaffected.  See
    # `reduce_sum_kernel.hip` for the canonical pattern; do NOT call
    # `std::numeric_limits<T>::<fn>()` from `__device__` code without a
    # matching adjustment.
    #
    # NOTE: this is a CC1 (frontend) flag in TheRock's clang build, not a
    # driver flag, so it must be passed via `-Xclang`.  Driver-form
    # `-fno-cuda-host-device-constexpr` errors with "unknown argument".
    list(APPEND abi_flags -Xclang -fno-cuda-host-device-constexpr)
    #
    # Layer 2 -- belt-and-suspenders: force-include a header that pre-defines
    # the `_CLANG_BUILTIN1` / `_CLANG_BUILTIN2` macros as empty.  This
    # successfully neutralised older MSVC 14.51.x revisions that emitted the
    # overloads at namespace scope after the `#define`, but MSVC 14.51.36231
    # `<cmath>` re-defines the macros unconditionally inside cmath
    # (`#define _CLANG_BUILTIN2(NAME) ...` with no `#ifndef` guard), so this
    # layer alone is no longer sufficient.  Kept as a safety net for the
    # cases where Layer 1's flag is not respected by some hipcc/clang
    # combination.
    if(EXISTS "${_MSVC_HIP_CMATH_WORKAROUND_HEADER}")
        list(APPEND abi_flags -include "${_MSVC_HIP_CMATH_WORKAROUND_HEADER}")
    endif()

    # Warning suppression flags
    set(warning_flags
        -Wno-ignored-attributes
    )

    # hipcc on Windows resolves clang from ENV{HIP_PATH}. Override per compile so
    # a stale machine-wide HIP_PATH (e.g. C:/opt/rocm/7.1) cannot hijack builds
    # that target THEROCK_DIST.
    set(_hipcc_env
        "HIP_PATH=${HIP_PATH}"
        "ROCM_PATH=${HIP_PATH}"
        "HIP_PLATFORM=amd"
    )

    # Check if multi-config generator
    _hip_is_multi_config(is_multi)

    if(is_multi)
        # Multi-config generator: use generator expressions for per-config objects
        set(obj_dir "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_hip_objs/$<CONFIG>")

        # Create directories for each config at configure time
        foreach(config Debug Release RelWithDebInfo MinSizeRel)
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_hip_objs/${config}")
        endforeach()

        set(obj_files "")
        foreach(source ${HIP_SOURCES})
            get_filename_component(source_name ${source} NAME_WE)
            get_filename_component(source_abs ${source} ABSOLUTE)
            set(output_obj "${obj_dir}/${source_name}.obj")

            # Use generator expressions for config-specific flags
            add_custom_command(
                OUTPUT ${output_obj}
                COMMAND ${CMAKE_COMMAND} -E env
                    ${_hipcc_env}
                    ${HIPCC_EXECUTABLE}
                    -c "${source_abs}"
                    -o "${output_obj}"
                    ${arch_flags}
                    ${include_flags}
                    ${define_flags}
                    ${abi_flags}
                    ${warning_flags}
                    $<$<CONFIG:Debug>:-D_DEBUG>
                    $<$<CONFIG:Debug>:-D_ITERATOR_DEBUG_LEVEL=2>
                    $<$<CONFIG:Debug>:-O0>
                    $<$<CONFIG:Debug>:-g>
                    $<$<NOT:$<CONFIG:Debug>>:-DNDEBUG>
                    $<$<NOT:$<CONFIG:Debug>>:-D_ITERATOR_DEBUG_LEVEL=0>
                    $<$<NOT:$<CONFIG:Debug>>:-O2>
                    $<$<CONFIG:RelWithDebInfo>:-g>
                    ${HIP_COMPILE_OPTIONS}
                    ${COMPILE_OPTS}
                DEPENDS ${source_abs}
                COMMENT "Compiling HIP source: ${source_name}.hip ($<CONFIG>)"
                VERBATIM
                COMMAND_EXPAND_LISTS
            )
            list(APPEND obj_files ${output_obj})
        endforeach()
    else()
        # Single-config generator: use CMAKE_BUILD_TYPE
        set(obj_dir "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_hip_objs")
        file(MAKE_DIRECTORY ${obj_dir})

        _hip_get_build_flags(build_flags)

        set(obj_files "")
        foreach(source ${HIP_SOURCES})
            get_filename_component(source_name ${source} NAME_WE)
            get_filename_component(source_abs ${source} ABSOLUTE)
            set(output_obj "${obj_dir}/${source_name}.obj")

            add_custom_command(
                OUTPUT ${output_obj}
                COMMAND ${CMAKE_COMMAND} -E env
                    ${_hipcc_env}
                    ${HIPCC_EXECUTABLE}
                    -c "${source_abs}"
                    -o "${output_obj}"
                    ${arch_flags}
                    ${include_flags}
                    ${define_flags}
                    ${abi_flags}
                    ${warning_flags}
                    ${build_flags}
                    ${HIP_COMPILE_OPTIONS}
                    ${COMPILE_OPTS}
                DEPENDS ${source_abs}
                COMMENT "Compiling HIP source: ${source_name}.hip"
                VERBATIM
            )
            list(APPEND obj_files ${output_obj})
        endforeach()
    endif()

    set(${OUTPUT_OBJS} ${obj_files} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# hip_add_executable - Create executable from HIP sources
#------------------------------------------------------------------------------
# Usage:
#   hip_add_executable(my_app
#       kernel.hip main.hip
#       INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include
#       COMPILE_OPTIONS -Wall
#       LINK_LIBRARIES some_lib
#       DEPENDS some_target
#   )
#------------------------------------------------------------------------------
function(hip_add_executable TARGET_NAME)
    cmake_parse_arguments(ARG "" ""
        "INCLUDE_DIRECTORIES;COMPILE_OPTIONS;LINK_LIBRARIES;DEPENDS;OFFLOAD_ARCHS" ${ARGN})

    # Remaining arguments are source files
    set(sources ${ARG_UNPARSED_ARGUMENTS})

    if(NOT sources)
        message(FATAL_ERROR "hip_add_executable: No source files provided for ${TARGET_NAME}")
    endif()

    if(WIN32)
        # Build CRT flags based on CMAKE_MSVC_RUNTIME_LIBRARY
        # Default to MultiThreadedDLL (/MD) if not specified
        set(crt_flags "")
        if(CMAKE_MSVC_RUNTIME_LIBRARY)
            if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
                # /MD or /MDd - Dynamic CRT
                list(APPEND crt_flags -D_DLL -D_MT "-Xclang" "--dependent-lib=msvcrt")
            else()
                # /MT or /MTd - Static CRT
                list(APPEND crt_flags -D_MT "-Xclang" "--dependent-lib=libcmt")
            endif()
        else()
            # Default to dynamic CRT (/MD)
            list(APPEND crt_flags -D_DLL -D_MT "-Xclang" "--dependent-lib=msvcrt")
        endif()

        # Combine user options with CRT flags
        set(all_compile_opts ${ARG_COMPILE_OPTIONS} ${crt_flags})

        # Compile HIP sources with hipcc
        _hip_compile_sources(${TARGET_NAME} "${sources}"
            "${ARG_INCLUDE_DIRECTORIES}" "${all_compile_opts}"
            "${ARG_OFFLOAD_ARCHS}" hip_objs)

        # Create custom target for HIP compilation
        add_custom_target(${TARGET_NAME}_hip_compile DEPENDS ${hip_objs})

        # Create executable (empty source, objects added)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.cpp" "// Auto-generated\n")
        add_executable(${TARGET_NAME} "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.cpp")
        target_sources(${TARGET_NAME} PRIVATE ${hip_objs})
        add_dependencies(${TARGET_NAME} ${TARGET_NAME}_hip_compile)
    else()
        # Linux: Use CMake HIP language support
        add_executable(${TARGET_NAME} ${sources})
        set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)

        # Set architectures (per-target OFFLOAD_ARCHS overrides the global list)
        if(ARG_OFFLOAD_ARCHS)
            set_target_properties(${TARGET_NAME} PROPERTIES
                HIP_ARCHITECTURES "${ARG_OFFLOAD_ARCHS}"
            )
        elseif(HIP_ARCHITECTURES)
            set_target_properties(${TARGET_NAME} PROPERTIES
                HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
            )
        endif()

        # Warning suppression flags
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-ignored-attributes>)
    endif()

    # Link HIP runtime
    target_link_libraries(${TARGET_NAME} PRIVATE hip::host)

    # Apply user-specified options
    if(ARG_INCLUDE_DIRECTORIES)
        target_include_directories(${TARGET_NAME} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
    endif()

    if(ARG_LINK_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()

    if(ARG_DEPENDS)
        add_dependencies(${TARGET_NAME} ${ARG_DEPENDS})
    endif()
endfunction()

#------------------------------------------------------------------------------
# hip_add_library - Create library from HIP sources
#------------------------------------------------------------------------------
# Usage:
#   hip_add_library(my_lib STATIC
#       kernel.hip utils.hip
#       INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include
#       COMPILE_OPTIONS -Wall
#       LINK_LIBRARIES some_lib
#       DEPENDS some_target
#   )
#------------------------------------------------------------------------------
function(hip_add_library TARGET_NAME)
    cmake_parse_arguments(ARG "STATIC;SHARED" ""
        "INCLUDE_DIRECTORIES;COMPILE_OPTIONS;LINK_LIBRARIES;DEPENDS;OFFLOAD_ARCHS" ${ARGN})

    # Determine library type
    if(ARG_SHARED)
        set(lib_type SHARED)
    else()
        set(lib_type STATIC)
    endif()

    # Remaining arguments are source files
    set(sources ${ARG_UNPARSED_ARGUMENTS})

    if(NOT sources)
        message(FATAL_ERROR "hip_add_library: No source files provided for ${TARGET_NAME}")
    endif()

    if(WIN32)
        set(crt_flags "")
        if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
            list(APPEND crt_flags -D_DLL -D_MT "-Xclang" "--dependent-lib=msvcrt")
        else()
            list(APPEND crt_flags -D_MT "-Xclang" "--dependent-lib=libcmt")
        endif()

        # Combine user options with CRT flags
        set(all_compile_opts ${ARG_COMPILE_OPTIONS} ${crt_flags})

        # Compile HIP sources with hipcc
        _hip_compile_sources(${TARGET_NAME} "${sources}"
            "${ARG_INCLUDE_DIRECTORIES}" "${all_compile_opts}"
            "${ARG_OFFLOAD_ARCHS}" hip_objs)

        # Create custom target for HIP compilation
        add_custom_target(${TARGET_NAME}_hip_compile DEPENDS ${hip_objs})

        # For shared libraries, we need a dummy C++ file to ensure proper CRT linkage
        if(ARG_SHARED)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.cpp"
                "// Auto-generated dummy source for CRT initialization\n")
            add_library(${TARGET_NAME} ${lib_type}
                "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.cpp" ${hip_objs})
        else()
            add_library(${TARGET_NAME} ${lib_type} ${hip_objs})
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
        add_dependencies(${TARGET_NAME} ${TARGET_NAME}_hip_compile)
    else()
        # Linux: Use CMake HIP language support
        add_library(${TARGET_NAME} ${lib_type} ${sources})
        set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)

        # Set architectures (per-target OFFLOAD_ARCHS overrides the global list)
        if(ARG_OFFLOAD_ARCHS)
            set_target_properties(${TARGET_NAME} PROPERTIES
                HIP_ARCHITECTURES "${ARG_OFFLOAD_ARCHS}"
            )
        elseif(HIP_ARCHITECTURES)
            set_target_properties(${TARGET_NAME} PROPERTIES
                HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
            )
        endif()

        # Warning suppression flags
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-Wno-ignored-attributes>)
    endif()

    # Propagate HIP settings to dependents
    target_link_libraries(${TARGET_NAME} PUBLIC hip::host)

    # Apply user-specified options
    if(ARG_INCLUDE_DIRECTORIES)
        target_include_directories(${TARGET_NAME} PUBLIC ${ARG_INCLUDE_DIRECTORIES})
    endif()

    if(ARG_LINK_LIBRARIES)
        target_link_libraries(${TARGET_NAME} PUBLIC ${ARG_LINK_LIBRARIES})
    endif()

    if(ARG_DEPENDS)
        add_dependencies(${TARGET_NAME} ${ARG_DEPENDS})
    endif()
endfunction()
