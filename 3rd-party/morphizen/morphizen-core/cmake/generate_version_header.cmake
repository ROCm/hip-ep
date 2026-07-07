##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Function to generate version_info_config.h for both Windows and Linux
function(generate_version_info_header)
    # Try to get version from release-kit (same as binMetaData.cmake)
    message(STATUS "Generating version info header...")

    # Try to fetch the release version via release-kit
    execute_process(
        COMMAND release-kit version --strategy RAI-DEV --format semantic --delimiter .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE RAI_DOTTED_VERSION
        ERROR_VARIABLE error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(${result} EQUAL 0)
        string(STRIP "${RAI_DOTTED_VERSION}" RAI_DOTTED_VERSION)
        message(STATUS "Retrieved RAI version: ${RAI_DOTTED_VERSION}")
    else()
        set(RAI_DOTTED_VERSION "9999.0.0")
        message(WARNING "Fetching RAI version failed, using default: ${RAI_DOTTED_VERSION}")
    endif()

    # Define build number
    if(DEFINED CI)
        if(DEFINED ENV{BUILD_NUMBER})
            set(RAI_VERSION_BUILD $ENV{BUILD_NUMBER})
        elseif(DEFINED ENV{GITHUB_RUN_ID})
            set(RAI_VERSION_BUILD $ENV{GITHUB_RUN_ID})
        else()
            message(WARNING "CI build but no BUILD_NUMBER or GITHUB_RUN_ID found, using 0")
            set(RAI_VERSION_BUILD 0)
        endif()
    else()
        set(RAI_VERSION_BUILD 0)
    endif()

    message(STATUS "Build number: ${RAI_VERSION_BUILD}")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

    # Configure the header file
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_info_config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version_info_config.h"
        @ONLY
    )

    # Make the generated header available to parent scope
    set(VERSION_INFO_HEADER "${CMAKE_CURRENT_BINARY_DIR}/version_info_config.h" PARENT_SCOPE)

    message(STATUS "Generated version header: ${CMAKE_CURRENT_BINARY_DIR}/version_info_config.h")
endfunction()
