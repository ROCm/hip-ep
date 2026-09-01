##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

function(hipdnn_ep_normalize_hip_architectures)
    if(NOT HIP_ARCHITECTURES OR NOT HIP_ARCHITECTURES MATCHES ",")
        return()
    endif()
    string(REPLACE "," ";" _hip_arch_raw "${HIP_ARCHITECTURES}")
    set(_hip_arch_normalized "")
    foreach(_a IN LISTS _hip_arch_raw)
        string(STRIP "${_a}" _a)
        if(_a)
            list(APPEND _hip_arch_normalized "${_a}")
        endif()
    endforeach()
    set(HIP_ARCHITECTURES "${_hip_arch_normalized}" CACHE STRING
        "Target GPU architectures (semicolon- or comma-separated, e.g. gfx1100;gfx1151)."
        FORCE)
    message(STATUS "[hip] Normalized HIP_ARCHITECTURES: ${HIP_ARCHITECTURES}")
endfunction()
