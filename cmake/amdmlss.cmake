##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##
# AMDMLSS integration for opt-in MLSS-backed conv (runtime dlopen of amdmlss.dll).
# Included early from cmake/deps.cmake so HIPDNN_EP_AMDMLSS_DLL is available when
# the MorphiZen EP target is created.

set(AMDMLSS_ROOT "" CACHE PATH "Path to AMDMLSS source tree (enables MLSS conv headers)")
if(NOT AMDMLSS_ROOT)
    set(_hip_ep_amdmlss_candidate "${CMAKE_SOURCE_DIR}/../../AMDMLSS")
    if(EXISTS "${_hip_ep_amdmlss_candidate}/modules/c_api/include/amdmlss/amdmlss_api.h")
        set(AMDMLSS_ROOT "${_hip_ep_amdmlss_candidate}" CACHE PATH
            "Path to AMDMLSS source tree (auto-detected)" FORCE)
    endif()
endif()

set(_hip_ep_amdmlss_third_party "${CMAKE_SOURCE_DIR}/lib/Runtime/third_party/amdmlss")
set(AMDMLSS_BUILD_ROOT "" CACHE PATH "Path to AMDMLSS build tree (generated headers + amdmlss.dll)")
if(NOT AMDMLSS_BUILD_ROOT AND AMDMLSS_ROOT)
    foreach(_amdmlss_build_candidate
            "${AMDMLSS_ROOT}/build/vs2022-release-samples"
            "${AMDMLSS_ROOT}/build/vs2022-release"
            "${AMDMLSS_ROOT}/build")
        if(EXISTS "${_amdmlss_build_candidate}/modules/c_api/include/amdmlss/amdmlss_export.h")
            set(AMDMLSS_BUILD_ROOT "${_amdmlss_build_candidate}" CACHE PATH
                "Path to AMDMLSS build tree (auto-detected)" FORCE)
            break()
        endif()
    endforeach()
endif()

set(HIPDNN_EP_AMDMLSS_DLL "" CACHE FILEPATH "amdmlss shared library staged beside EP binaries")
if(NOT HIPDNN_EP_AMDMLSS_DLL)
    if(AMDMLSS_BUILD_ROOT)
        foreach(_amdmlss_dll_candidate
                "${AMDMLSS_BUILD_ROOT}/modules/c_api/Release/amdmlss.dll"
                "${AMDMLSS_BUILD_ROOT}/samples/Release/amdmlss.dll"
                "${AMDMLSS_BUILD_ROOT}/lib/amdmlss.dll"
                "${AMDMLSS_BUILD_ROOT}/libamdmlss.so"
                "${AMDMLSS_BUILD_ROOT}/lib/amdmlss.so")
            if(EXISTS "${_amdmlss_dll_candidate}")
                set(HIPDNN_EP_AMDMLSS_DLL "${_amdmlss_dll_candidate}" CACHE FILEPATH
                    "amdmlss shared library (auto-detected from AMDMLSS build)" FORCE)
                break()
            endif()
        endforeach()
    endif()
    if(NOT HIPDNN_EP_AMDMLSS_DLL)
        foreach(_amdmlss_dll_candidate
                "${_hip_ep_amdmlss_third_party}/bin/amdmlss.dll"
                "${_hip_ep_amdmlss_third_party}/bin/libamdmlss.so")
            if(EXISTS "${_amdmlss_dll_candidate}")
                set(HIPDNN_EP_AMDMLSS_DLL "${_amdmlss_dll_candidate}" CACHE FILEPATH
                    "amdmlss shared library (vendored under lib/Runtime/third_party)" FORCE)
                break()
            endif()
        endforeach()
    endif()
endif()

set(HIPDNN_EP_AMDMLSS_COMPILE_FLAGS "")
if(AMDMLSS_ROOT AND EXISTS "${AMDMLSS_ROOT}/modules/c_api/include/amdmlss/amdmlss_api.h")
    list(APPEND HIPDNN_EP_AMDMLSS_COMPILE_FLAGS
        "-I${AMDMLSS_ROOT}/modules/common/include"
        "-I${AMDMLSS_ROOT}/modules/c_api/include"
        "-DHIPDNN_EP_AMDMLSS_HEADERS=1")
    if(AMDMLSS_BUILD_ROOT AND
       EXISTS "${AMDMLSS_BUILD_ROOT}/modules/c_api/include/amdmlss/amdmlss_export.h")
        list(APPEND HIPDNN_EP_AMDMLSS_COMPILE_FLAGS
            "-I${AMDMLSS_BUILD_ROOT}/modules/c_api/include")
    elseif(EXISTS "${_hip_ep_amdmlss_third_party}/include/amdmlss/amdmlss_export.h")
        list(APPEND HIPDNN_EP_AMDMLSS_COMPILE_FLAGS
            "-I${_hip_ep_amdmlss_third_party}/include")
    else()
        list(APPEND HIPDNN_EP_AMDMLSS_COMPILE_FLAGS
            "-I${CMAKE_SOURCE_DIR}/lib/Runtime/real/amdmlss_shim/include")
    endif()
    message(STATUS "AMDMLSS conv support: headers from ${AMDMLSS_ROOT}")
    if(HIPDNN_EP_AMDMLSS_DLL)
        message(STATUS "AMDMLSS conv support: runtime DLL ${HIPDNN_EP_AMDMLSS_DLL}")
    else()
        message(STATUS "AMDMLSS conv support: runtime DLL not found (build AMDMLSS or copy to third_party/amdmlss/bin)")
    endif()
else()
    message(STATUS "AMDMLSS conv support: disabled (set AMDMLSS_ROOT to enable)")
endif()
