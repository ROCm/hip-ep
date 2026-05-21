##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-backend-gfx<ARCH>.{dll,so} -- per-gfx EP backend hosting heavy ROCm
# kernels (Composable Kernel today; future hipBLASLt-tuned kernels, etc.).
# Loaded at runtime by model.dll via LoadLibrary/dlopen + GetProcAddress/
# dlsym so the giant device_conv_operations.lib (2.6 GB) never enters the
# per-model.dll lld-link path -- only this DLL pays the link cost, once at
# build time.
#
# Single exported symbol: HIPBackendAPI (pointer to a HIPBackendVTable).
# On Windows it's gated by hip_backend.def + /DEF: at link time; on Linux
# the rest of the DLL gets hidden visibility and HIPBackendAPI carries
# __attribute__((visibility("default"))) in main.cpp -- same one-symbol
# ABI either way.
#
# Two build paths because the HIP toolchain story differs across the OSes:
#
# Windows: use the _hip_compile_sources helper from custom_kernels'
# hip_utils directly. hip_add_library SHARED would emit a dummy .cpp and
# link hip::host PUBLIC, which propagates `-x hip` and similar hipcc-only
# flags to the dummy compile step -- cl.exe rejects those. Building the
# SHARED-library shell ourselves lets us keep the hipcc compile of CK
# code AND have a clean MSVC compile of the SHARED-lib boilerplate.
#
# Linux: use CMake's HIP language support (enable_language(HIP) +
# LANGUAGE HIP source property). main.cpp stays a regular C++ compile.

cmake_minimum_required(VERSION 3.18)

include(${CMAKE_SOURCE_DIR}/3rd-party/custom_kernels/cmake/hip_utils.cmake)

if(NOT HIP_ARCHITECTURES)
    message(FATAL_ERROR
        "[backend] HIP_ARCHITECTURES is not set.\n"
        "Pass -DHIP_ARCHITECTURES=<arch> to cmake (e.g. gfx1151).")
endif()

list(GET HIP_ARCHITECTURES 0 BACKEND_ARCH)
set(BACKEND_TARGET "hip-backend-${BACKEND_ARCH}")

# composable_kernel's imported targets reference Threads::Threads and
# hip::device transitively, so both must be findable before find_package
# below.
find_package(Threads REQUIRED)
find_package(hip REQUIRED CONFIG)
find_package(composable_kernel REQUIRED COMPONENTS device_conv_operations utility)
message(STATUS "[backend] composable_kernel: ${composable_kernel_DIR}")

# main.cpp is plain C++ -- no HIP intrinsics, no __device__ code. Keep it
# off the HIP path on both platforms; compiling through hipcc would also
# emit a device-side object with the static `g_vtable` and its function-
# pointer initializers, then the device linker would (correctly) reject
# the host-only target functions as undefined-on-device.
#
# Only TUs that actually contain HIP code (HIP intrinsics or __global__
# kernels) belong in BACKEND_HIP_SOURCES.
set(BACKEND_HIP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
)

set(BACKEND_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
    # -------------------------------------------------------------------
    # Windows: hipcc compile of conv.cpp + MSVC compile of the shell + main.
    # -------------------------------------------------------------------
    #
    # CRT: device_conv_operations.lib is built upstream against the dynamic
    # CRT (/MD). We MUST match: top-level CMakeLists sets
    # CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded (/MT) for the model.dll's
    # static-CRT requirement, but the backend DLL is a separate artifact --
    # override to /MD both for hipcc-compiled objects (here, via
    # -D_DLL/-D_MT/--dependent-lib=msvcrt) AND for the SHARED-lib target
    # itself (set_target_properties below).
    set(BACKEND_HIPCC_OPTS
        -std=c++17
        "-DHIPDNN_EP_BACKEND_ARCH=${BACKEND_ARCH}"
        -D_DLL -D_MT "-Xclang" "--dependent-lib=msvcrt"
    )

    # Compile HIP sources via hipcc -> .obj (uses _hip_compile_sources from
    # hip_utils.cmake). Output lands in
    # ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_hip_objs/.
    _hip_compile_sources(
        ${BACKEND_TARGET}
        "${BACKEND_HIP_SOURCES}"
        "${BACKEND_INCLUDES}"
        "${BACKEND_HIPCC_OPTS}"
        BACKEND_HIP_OBJS
    )
    add_custom_target(${BACKEND_TARGET}_hip_compile DEPENDS ${BACKEND_HIP_OBJS})

    # Minimal real shell .cpp -- gives MSVC something to compile so the
    # SHARED library's CRT init / DllMain / __DllMainCRTStartup chain is
    # wired up correctly. Plain C++; no hipcc-only flags.
    set(BACKEND_SHELL "${CMAKE_CURRENT_BINARY_DIR}/${BACKEND_TARGET}_shell.cpp")
    file(WRITE ${BACKEND_SHELL}
"// Auto-generated. Provides a translation unit for the SHARED hip-backend\n"
"// DLL so MSVC supplies CRT init. The real backend code lives in the\n"
"// hipcc-compiled .obj files attached via target_sources.\n"
"namespace { volatile int hip_backend_shell_anchor = 0; }\n"
)

    # Module-def file gates exports -- only symbols listed in hip_backend.def
    # leave the DLL. Adding it as a source makes CMake pass it to link.exe
    # via the LINK_FLAGS /DEF: mechanism (handled automatically for .def files).
    set(BACKEND_DEF "${CMAKE_CURRENT_SOURCE_DIR}/hip_backend.def")

    add_library(${BACKEND_TARGET} SHARED
        ${BACKEND_SHELL}
        ${BACKEND_DEF}
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    target_sources(${BACKEND_TARGET} PRIVATE ${BACKEND_HIP_OBJS})
    add_dependencies(${BACKEND_TARGET} ${BACKEND_TARGET}_hip_compile)

    set_target_properties(${BACKEND_TARGET} PROPERTIES
        LINKER_LANGUAGE CXX
        # Override the project-wide /MT to match CK's upstream /MD build.
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
else()
    # -------------------------------------------------------------------
    # Linux: CMake HIP language support compiles conv.cpp via amdclang++
    # with --offload-arch=<arch>. main.cpp compiled by the regular C++
    # compiler. Exports controlled by visibility -- only HIPBackendAPI
    # has __attribute__((visibility("default"))) (in main.cpp); the rest
    # of the DLL gets hidden visibility via the target properties below.
    # No .def-file equivalent needed.
    # -------------------------------------------------------------------
    add_library(${BACKEND_TARGET} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${BACKEND_HIP_SOURCES}
    )
    set_source_files_properties(${BACKEND_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    set_target_properties(${BACKEND_TARGET} PROPERTIES
        HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_options(${BACKEND_TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-ignored-attributes>
    )
endif()

target_include_directories(${BACKEND_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
# main.cpp's EP_STR(HIPDNN_EP_BACKEND_ARCH) needs the bare-identifier macro.
target_compile_definitions(${BACKEND_TARGET} PRIVATE
    HIPDNN_EP_BACKEND_ARCH=${BACKEND_ARCH})

# Link raw .lib / .so paths instead of the hip::host / composable_kernel::*
# imported targets -- those carry INTERFACE_COMPILE_OPTIONS like `-x hip`
# that CMake propagates to the SHARED-lib shell.cpp / main.cpp compiles,
# where the regular C++ compiler rejects them. The imported targets ARE
# used at the HIP-compile stage via either _hip_compile_sources (Windows)
# or CMake's HIP language support (Linux); we only bypass them at the
# host-side link/compile stages.
get_target_property(_ck_conv_lib composable_kernel::device_conv_operations
                    IMPORTED_LOCATION_RELEASE)
get_target_property(_ck_util_lib composable_kernel::utility
                    IMPORTED_LOCATION_RELEASE)
if(NOT _ck_conv_lib)
    get_target_property(_ck_conv_lib composable_kernel::device_conv_operations
                        IMPORTED_LOCATION)
endif()
if(NOT _ck_util_lib)
    get_target_property(_ck_util_lib composable_kernel::utility
                        IMPORTED_LOCATION)
endif()
message(STATUS "[backend] CK conv lib: ${_ck_conv_lib}")
message(STATUS "[backend] CK util lib: ${_ck_util_lib}")

# HIP runtime library: WIN32 path of hip_utils.cmake sets HIP_RUNTIME_LIBRARY
# explicitly (amdhip64.lib). On Linux we use the hip::host imported target
# from find_package(hip) -- its INTERFACE compile options only fire for
# HIP-language sources via CMake's HIP support, so they don't leak into
# main.cpp's regular C++ compile.
if(WIN32)
    set(_hip_runtime "${HIP_RUNTIME_LIBRARY}")
else()
    set(_hip_runtime "hip::host")
endif()

target_link_libraries(${BACKEND_TARGET} PRIVATE
    ${_hip_runtime}
    ${_ck_conv_lib}
    ${_ck_util_lib}
    Threads::Threads
)

install(TARGETS ${BACKEND_TARGET}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
)

install(FILES hipdnn_ep_backend.h
    DESTINATION include/hip
)

message(STATUS "[backend] Will produce ${BACKEND_TARGET} for arch ${BACKEND_ARCH}")
