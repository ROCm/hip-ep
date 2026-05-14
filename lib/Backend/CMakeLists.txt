##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# hip-backend-gfx<ARCH>.dll -- per-gfx EP backend hosting heavy ROCm kernels
# (Composable Kernel today; future hipBLASLt-tuned kernels, etc.). Loaded
# at runtime by model.dll via LoadLibrary + GetProcAddress so the giant
# device_conv_operations.lib (2.6 GB) never enters the per-model.dll
# lld-link path -- only this DLL pays the link cost, once at build time
# via MSVC link.exe.
#
# We use the _hip_compile_sources helper from custom_kernels' hip_utils
# directly (instead of hip_add_library SHARED) because hip_add_library's
# SHARED variant emits a dummy .cpp and links hip::host PUBLIC, which
# propagates `-x hip` and similar hipcc-only flags to the dummy compile
# step. cl.exe rejects those. Building the SHARED library shell ourselves
# lets us keep the hipcc compile of CK code AND have a clean MSVC compile
# of the SHARED-lib boilerplate.
#
# Exports are gated by hip_backend.def -- only the public C ABI declared
# in hipdnn_ep_backend.h crosses the DLL boundary.

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

# main.cpp is plain C++ -- no HIP intrinsics, no __device__ code. Compiled
# with cl.exe via the SHARED target's regular source list (see add_library
# below). Compiling it through hipcc would also emit a device-side object
# with the static `g_vtable` and its function-pointer initializers, then
# the device linker would (correctly) reject the host-only target functions
# as undefined-on-device. Keeping main.cpp out of the HIP path sidesteps
# that entirely.
#
# Only TUs that actually contain HIP code (HIP intrinsics or __global__
# kernels) belong in this list.
set(BACKEND_HIP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
)

set(BACKEND_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Pass C++17 (CK requires it) and the bare-identifier arch macro that
# main.cpp stringifies via STR(x).
#
# CRT: device_conv_operations.lib is built upstream against the dynamic
# CRT (/MD). We MUST match: top-level CMakeLists sets
# CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded (/MT) for the model.dll's static-CRT
# requirement, but the backend DLL is a separate artifact -- override to /MD
# both for hipcc-compiled objects (here, via -D_DLL/-D_MT/--dependent-lib=msvcrt)
# AND for the SHARED-lib target itself (set_target_properties below).
set(BACKEND_HIPCC_OPTS
    -std=c++17
    "-DHIPDNN_EP_BACKEND_ARCH=${BACKEND_ARCH}"
    -D_DLL -D_MT "-Xclang" "--dependent-lib=msvcrt"
)

# Compile HIP sources via hipcc -> .obj (uses _hip_compile_sources from
# hip_utils.cmake). Output is in ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_hip_objs/.
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
target_include_directories(${BACKEND_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
# main.cpp's EP_STR(HIPDNN_EP_BACKEND_ARCH) needs the bare-identifier macro.
target_compile_definitions(${BACKEND_TARGET} PRIVATE
    HIPDNN_EP_BACKEND_ARCH=${BACKEND_ARCH})
set_target_properties(${BACKEND_TARGET} PROPERTIES
    LINKER_LANGUAGE CXX
    # Override the project-wide /MT to match CK's upstream /MD build.
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
)

# Link raw .lib paths instead of the hip::host / composable_kernel::*
# imported targets -- those carry INTERFACE_COMPILE_OPTIONS like `-x hip`
# that CMake propagates to the SHARED-lib shell.cpp compile, where
# cl.exe rejects them. The imported targets ARE used at hipcc compile
# time via _hip_compile_sources's manual include/lib handling; we only
# bypass them at the shell-link stage.
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

target_link_libraries(${BACKEND_TARGET} PRIVATE
    ${HIP_RUNTIME_LIBRARY}
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

message(STATUS "[backend] Will produce ${BACKEND_TARGET}.dll for arch ${BACKEND_ARCH}")
