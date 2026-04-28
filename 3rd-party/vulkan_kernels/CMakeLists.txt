##
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
##

# vulkan_kernels — Vulkan compute-shader kernels for ops without ROCm
#
# Build flow:
#   .comp files  → glslc → .spv (SPIR-V bytecode)
#   .spv files   → xxd-style embedding → .spv.h (C byte arrays)
#   .cpp + .spv.h → MSVC/GCC → .obj → static lib (vulkan_custom_kernels.lib)
#
# Usage:
#   cmake -DBUILD_VULKAN_RUNTIME=ON -DVULKAN_SDK=C:/VulkanSDK/x.y.z ...

cmake_minimum_required(VERSION 3.20)

# --- Find Vulkan SDK (glslc + headers + loader) ---
find_package(Vulkan REQUIRED COMPONENTS glslc)
message(STATUS "[vulkan_kernels] Vulkan SDK: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "[vulkan_kernels] glslc: ${Vulkan_GLSLC_EXECUTABLE}")

# --- Shader compilation: .comp → .spv ---
set(SHADER_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SPV_DIR     "${CMAKE_CURRENT_BINARY_DIR}/spv")
set(SPV_HDR_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv_headers")
file(MAKE_DIRECTORY ${SPV_DIR} ${SPV_HDR_DIR})

set(SHADER_SOURCES
    matmul_nbits.comp
    matmul_nbits_vec.comp
)

set(SPV_OUTPUTS "")
set(SPV_HEADERS "")

foreach(shader IN LISTS SHADER_SOURCES)
    get_filename_component(name_we ${shader} NAME_WE)
    set(spv_file "${SPV_DIR}/${name_we}.spv")
    set(hdr_file "${SPV_HDR_DIR}/${name_we}_spv.h")

    # Compile GLSL → SPIR-V
    add_custom_command(
        OUTPUT  ${spv_file}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE}
                -fshader-stage=compute
                --target-env=vulkan1.1
                -O
                -o ${spv_file}
                ${SHADER_DIR}/${shader}
        DEPENDS ${SHADER_DIR}/${shader}
        COMMENT "[vulkan_kernels] Compiling ${shader} → ${name_we}.spv"
        VERBATIM
    )

    # Embed SPIR-V as C header (xxd-style)
    add_custom_command(
        OUTPUT  ${hdr_file}
        COMMAND ${CMAKE_COMMAND}
                -DSPV_FILE=${spv_file}
                -DHDR_FILE=${hdr_file}
                -DVAR_NAME=${name_we}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
        DEPENDS ${spv_file}
        COMMENT "[vulkan_kernels] Embedding ${name_we}.spv → ${name_we}_spv.h"
        VERBATIM
    )

    list(APPEND SPV_OUTPUTS ${spv_file})
    list(APPEND SPV_HEADERS ${hdr_file})
endforeach()

# Custom target so library depends on shader compilation
add_custom_target(vulkan_shaders DEPENDS ${SPV_HEADERS})

# --- Build static library ---
add_library(vulkan_custom_kernels STATIC
    src/vulkan_device.cpp
    src/vulkan_matmul_nbits.cpp
)

add_dependencies(vulkan_custom_kernels vulkan_shaders)

target_include_directories(vulkan_custom_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SPV_HDR_DIR}
)

target_link_libraries(vulkan_custom_kernels PUBLIC Vulkan::Vulkan)

# Export paths to parent scope
set(VULKAN_CUSTOM_KERNELS_LIB_PATH "$<TARGET_FILE:vulkan_custom_kernels>" PARENT_SCOPE)
set(VULKAN_CUSTOM_KERNELS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

# --- Unit test executable ---
if(BUILD_VULKAN_TESTS)
    add_executable(test_matmul_nbits_vulkan test/test_matmul_nbits_vulkan.cpp)
    target_link_libraries(test_matmul_nbits_vulkan PRIVATE vulkan_custom_kernels Vulkan::Vulkan)
    add_test(NAME MatMulNBitsVulkan COMMAND test_matmul_nbits_vulkan)
endif()

# --- Install ---
install(TARGETS vulkan_custom_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

message(STATUS "[vulkan_kernels] Will install vulkan_custom_kernels to \${CMAKE_INSTALL_PREFIX}/lib")
