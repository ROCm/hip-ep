##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##
# Stage amdmlss.dll/libamdmlss.so beside a runtime binary. Must be invoked as a
# macro from the same CMakeLists.txt that defines ${target} (CMake requires the
# target and add_custom_command to live in the same directory).
macro(hip_ep_stage_amdmlss_dll target)
    if(HIPDNN_EP_AMDMLSS_DLL AND TARGET ${target})
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${HIPDNN_EP_AMDMLSS_DLL}"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Staging amdmlss runtime DLL beside ${target}"
            VERBATIM)
    endif()
endmacro()
