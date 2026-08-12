##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(_HIP_FAMILY_gfx115X_members gfx1150 gfx1151 gfx1152 gfx1153)
set(HIP_GPU_FAMILY_gfx115X-all ${_HIP_FAMILY_gfx115X_members})
unset(_HIP_FAMILY_gfx115X_members)

# True when NAME is a registered family (not a concrete gfxNNNN arch).
function(hip_is_gpu_family NAME OUT_VAR)
    if(DEFINED HIP_GPU_FAMILY_${NAME})
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Member arch list for a family; empty when NAME is not a family.
function(hip_expand_gpu_family NAME OUT_VAR)
    if(DEFINED HIP_GPU_FAMILY_${NAME})
        set(${OUT_VAR} ${HIP_GPU_FAMILY_${NAME}} PARENT_SCOPE)
    else()
        set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

# First concrete gfx arch for TheRock tarball selection / defaults.
function(hip_first_concrete_gpu_arch ENTRY OUT_VAR)
    hip_is_gpu_family("${ENTRY}" _is_family)
    if(_is_family)
        hip_expand_gpu_family("${ENTRY}" _members)
        if(_members)
            list(GET _members 0 _first)
            set(${OUT_VAR} "${_first}" PARENT_SCOPE)
            return()
        endif()
    endif()
    set(${OUT_VAR} "${ENTRY}" PARENT_SCOPE)
endfunction()
