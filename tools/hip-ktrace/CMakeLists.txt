##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

cmake_minimum_required(VERSION 3.24)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
    KTRACE_VERSION_STRING LIMIT_COUNT 1)
if(NOT KTRACE_VERSION_STRING)
    message(FATAL_ERROR "VERSION is missing or empty")
endif()
# project() only accepts numeric components, so drop any -rcN style suffix.
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" KTRACE_PROJECT_VERSION
    "${KTRACE_VERSION_STRING}")
if(NOT KTRACE_PROJECT_VERSION)
    message(FATAL_ERROR "VERSION does not start with a numeric version: "
        "${KTRACE_VERSION_STRING}")
endif()

project(ktrace VERSION ${KTRACE_PROJECT_VERSION} LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "ktrace currently supports Windows only")
endif()
if(NOT MSVC)
    message(FATAL_ERROR "ktrace requires the MSVC toolchain")
endif()

option(KTRACE_BUILD_CAPTURE "Build the injectable tracer, launcher, and version probe" ON)
option(KTRACE_BUILD_RDP "Build the combined ktrace and RGP launcher" ON)
option(KTRACE_BUILD_POSTPROCESS "Build the native post-processing tool" ON)
option(KTRACE_BUILD_WORKLOAD "Build the synthetic HIP workload" ON)

set(KTRACE_HIP_ROOT "$ENV{HIP}" CACHE PATH "HIP/ROCm installation used to build workloads")
set(KTRACE_HIP_HEADERS "$ENV{HIP_HEADERS}" CACHE PATH
    "HIP include directory matching the target process runtime")
set(KTRACE_GPU_ARCH "$ENV{GPU_ARCH}" CACHE STRING
    "Comma- or semicolon-separated GPU architectures passed to hipcc")

# Keep runnable artifacts separate from generated build-system files and from
# out/, which the capture scripts intentionally delete before each run.
set(KTRACE_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${KTRACE_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${KTRACE_BIN_DIR}")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${KTRACE_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${KTRACE_BIN_DIR}")
endforeach()

function(ktrace_set_common_options target)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        WIN32
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        KTRACE_VERSION_STRING="${KTRACE_VERSION_STRING}"
    )
    target_compile_options(${target} PRIVATE /EHsc)
endfunction()

if(KTRACE_BUILD_CAPTURE)
    if(NOT KTRACE_HIP_HEADERS)
        message(FATAL_ERROR
            "KTRACE_HIP_HEADERS is required when KTRACE_BUILD_CAPTURE=ON. "
            "It must match the runtime loaded by the traced process.")
    endif()
    if(NOT EXISTS "${KTRACE_HIP_HEADERS}/hip/amd_detail/hip_prof_str.h")
        message(FATAL_ERROR
            "KTRACE_HIP_HEADERS does not contain hip/amd_detail/hip_prof_str.h: "
            "${KTRACE_HIP_HEADERS}")
    endif()

    add_library(roctx64 SHARED src/roctx/roctx.cpp)
    ktrace_set_common_options(roctx64)
    target_compile_definitions(roctx64 PRIVATE ROCTX_EXPORTS)
    target_include_directories(roctx64 PRIVATE src/external/roctx)
    set_target_properties(roctx64 PROPERTIES PREFIX "" OUTPUT_NAME "roctx64")

    add_library(ktrace_tracer SHARED src/tracer/ktrace_tracer.cpp)
    ktrace_set_common_options(ktrace_tracer)
    target_compile_definitions(ktrace_tracer PRIVATE
        _WINDOWS
        __HIP_PLATFORM_AMD__
        USE_PROF_API=1
    )
    target_include_directories(ktrace_tracer PRIVATE "${KTRACE_HIP_HEADERS}")
    target_link_libraries(ktrace_tracer PRIVATE psapi)
    set_target_properties(ktrace_tracer PROPERTIES PREFIX "" OUTPUT_NAME "ktrace_tracer")

    add_executable(version_probe src/version_probe.cpp)
    ktrace_set_common_options(version_probe)
    target_compile_definitions(version_probe PRIVATE
        _CONSOLE
        __HIP_PLATFORM_AMD__
        USE_PROF_API=1
    )
    target_include_directories(version_probe PRIVATE "${KTRACE_HIP_HEADERS}")

    if(KTRACE_BUILD_RDP)
        add_library(ktrace_ipc STATIC src/ipc/ktrace_ipc.cpp)
        ktrace_set_common_options(ktrace_ipc)
        target_include_directories(ktrace_ipc PUBLIC src/ipc)

        add_executable(ktrace src/tracer/ktrace_launch.cpp)
        ktrace_set_common_options(ktrace)
        target_compile_definitions(ktrace PRIVATE
            _CONSOLE
            UNICODE
            _UNICODE
        )
        target_compile_options(ktrace PRIVATE /W4 /utf-8)
        target_include_directories(ktrace PRIVATE src/rdp)
        target_link_libraries(ktrace PRIVATE ktrace_ipc version)
        target_link_libraries(ktrace_tracer PRIVATE ktrace_ipc)
        add_dependencies(ktrace ktrace_tracer)

        add_executable(ktrace_collector
            src/collector/binary_journal.cpp
            src/collector/ktrace_collector.cpp
            src/collector/etw_collector.cpp
            src/collector/spool_writer.cpp)
        ktrace_set_common_options(ktrace_collector)
        target_include_directories(ktrace_collector PRIVATE src/collector)
        target_link_libraries(ktrace_collector PRIVATE ktrace_ipc advapi32 tdh)
    endif()

    add_custom_target(ktrace_tools
        DEPENDS roctx64 ktrace_tracer version_probe
    )
    if(TARGET ktrace)
        add_dependencies(ktrace_tools ktrace ktrace_collector)
    endif()
elseif(KTRACE_BUILD_RDP)
    message(FATAL_ERROR
        "KTRACE_BUILD_RDP requires KTRACE_BUILD_CAPTURE=ON because "
            "ktrace injects ktrace_tracer.dll.")
endif()

if(KTRACE_BUILD_POSTPROCESS)
    add_executable(ktrace_postprocess
        src/postprocess/main.cpp
        src/postprocess/common.cpp
        src/postprocess/finalize.cpp
        src/postprocess/analyze.cpp
        src/postprocess/chrome.cpp
        src/postprocess/pftrace_writer.cpp
        src/postprocess/all.cpp
    )
    ktrace_set_common_options(ktrace_postprocess)
    target_compile_options(ktrace_postprocess PRIVATE /utf-8)
    target_link_libraries(ktrace_postprocess PRIVATE dbghelp)
    target_include_directories(ktrace_postprocess PRIVATE
        src/postprocess
        src/external/amd_comgr
        src/external/rapidjson/include
    )
endif()

if(KTRACE_BUILD_WORKLOAD)
    if(NOT KTRACE_GPU_ARCH)
        message(FATAL_ERROR
            "KTRACE_GPU_ARCH is required when KTRACE_BUILD_WORKLOAD=ON "
            "(for example gfx1100 or gfx1151).")
    endif()
    find_program(KTRACE_HIPCC
        NAMES hipcc.exe hipcc.bat hipcc
        HINTS "${KTRACE_HIP_ROOT}/bin"
        DOC "Path to hipcc"
        REQUIRED
    )
    string(REPLACE "," ";" KTRACE_GPU_ARCH_LIST "${KTRACE_GPU_ARCH}")
    set(KTRACE_HIP_ARCH_FLAGS)
    set(KTRACE_GPU_ARCH_DISPLAY)
    foreach(arch IN LISTS KTRACE_GPU_ARCH_LIST)
        string(STRIP "${arch}" arch)
        if(arch)
            list(APPEND KTRACE_HIP_ARCH_FLAGS "--offload-arch=${arch}")
            list(APPEND KTRACE_GPU_ARCH_DISPLAY "${arch}")
        endif()
    endforeach()
    if(NOT KTRACE_HIP_ARCH_FLAGS)
        message(FATAL_ERROR "KTRACE_GPU_ARCH did not contain any valid architectures")
    endif()
    list(JOIN KTRACE_GPU_ARCH_DISPLAY ", " KTRACE_GPU_ARCH_TEXT)
endif()

if(KTRACE_BUILD_WORKLOAD)
    set(workload_exe "${KTRACE_BIN_DIR}/workload.exe")
    set(roctx_workload_exe "${KTRACE_BIN_DIR}/roctx_workload.exe")
    add_custom_command(
        OUTPUT "${workload_exe}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${KTRACE_BIN_DIR}"
        COMMAND "${KTRACE_HIPCC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/workload.cpp"
            -O2
            -std=c++17
            ${KTRACE_HIP_ARCH_FLAGS}
            -D_CRT_SECURE_NO_WARNINGS
            -o "${workload_exe}"
        DEPENDS src/workload.cpp
        COMMENT "Building synthetic HIP workload for ${KTRACE_GPU_ARCH_TEXT}"
        VERBATIM
    )
    add_custom_target(workload ALL DEPENDS "${workload_exe}")
    add_custom_command(
        OUTPUT "${roctx_workload_exe}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${KTRACE_BIN_DIR}"
        COMMAND "${KTRACE_HIPCC}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/roctx_workload.cpp"
            -O2
            -std=c++17
            ${KTRACE_HIP_ARCH_FLAGS}
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src/external/roctx"
            "-L${CMAKE_CURRENT_BINARY_DIR}"
            -lroctx64
            -D_CRT_SECURE_NO_WARNINGS
            -DROCTX_IMPORT_DECORATOR=
            -o "${roctx_workload_exe}"
        DEPENDS src/roctx_workload.cpp roctx64
        COMMENT "Building ROCTx HIP workload for ${KTRACE_GPU_ARCH_TEXT}"
        VERBATIM
    )
    add_custom_target(roctx_workload ALL DEPENDS "${roctx_workload_exe}")
endif()

include(CTest)
if(BUILD_TESTING)
    if(TARGET ktrace_postprocess)
        add_test(NAME ktrace.postprocess.help COMMAND ktrace_postprocess --help)
    endif()
    if(TARGET ktrace)
        add_test(NAME ktrace.rdp.help COMMAND ktrace --help)
        add_test(NAME ktrace.rdp.version COMMAND ktrace --version)
        add_test(NAME ktrace.rdp.invalid-option
            COMMAND ktrace --definitely-invalid)
        set_tests_properties(ktrace.rdp.invalid-option PROPERTIES WILL_FAIL TRUE)
        add_executable(ktrace_ipc_ring_test tests/ipc_ring_test.cpp)
        ktrace_set_common_options(ktrace_ipc_ring_test)
        target_link_libraries(ktrace_ipc_ring_test PRIVATE ktrace_ipc)
        add_test(NAME ktrace.ipc.ring COMMAND ktrace_ipc_ring_test)
    endif()
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME ktrace.schema
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests -v
        )
        set_tests_properties(ktrace.schema PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        add_test(
            NAME ktrace.fixtures
            COMMAND "${Python3_EXECUTABLE}" tests/rocprofv3_schema.py --fixtures
        )
        set_tests_properties(ktrace.fixtures PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        if(TARGET ktrace_postprocess)
            add_test(
                NAME ktrace.native.parity
                COMMAND "${Python3_EXECUTABLE}" tests/test_native_parity.py -v
            )
            set_tests_properties(ktrace.native.parity PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                ENVIRONMENT "KTRACE_POSTPROCESS=$<TARGET_FILE:ktrace_postprocess>"
            )
        endif()
    else()
        message(WARNING "Python was not found; schema tests are unavailable")
    endif()
endif()

set(ktrace_install_targets)
foreach(target roctx64 ktrace_tracer ktrace ktrace_collector version_probe ktrace_postprocess)
    if(TARGET ${target})
        list(APPEND ktrace_install_targets ${target})
    endif()
endforeach()
if(ktrace_install_targets)
    install(TARGETS ${ktrace_install_targets}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
        ARCHIVE DESTINATION lib
    )
endif()
if(KTRACE_BUILD_WORKLOAD)
    install(FILES "${workload_exe}" "${roctx_workload_exe}" DESTINATION bin OPTIONAL)
endif()
install(PROGRAMS "${KTRACE_BIN_DIR}/llvm-cxxfilt.exe" DESTINATION bin OPTIONAL)
install(FILES
    "${KTRACE_BIN_DIR}/RdpCaptureApi.dll"
    "${KTRACE_BIN_DIR}/GPUPerfAPICounters-x64.dll"
    "${KTRACE_BIN_DIR}/amd_comgr_3.dll"
    "${KTRACE_BIN_DIR}/amdgpu_dis.dll"
    DESTINATION bin
    OPTIONAL)
install(DIRECTORY python/ DESTINATION python
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY tests/ DESTINATION tests
    FILES_MATCHING PATTERN "*.py" PATTERN "*.csv" PATTERN "*.json"
    PATTERN "__pycache__" EXCLUDE)
install(FILES
    README.md
    VERSION
    THIRD_PARTY_NOTICES.md
    run.bat
    DESTINATION .)
# README.md links to the profiling skill, so ship it or that link dangles.
install(DIRECTORY .claude/ DESTINATION .claude)
install(FILES src/external/rapidjson/LICENSE
    DESTINATION licenses RENAME RapidJSON-LICENSE.txt)
install(FILES src/external/llvm/LICENSE.txt
    DESTINATION licenses RENAME LLVM-LICENSE.txt)
install(FILES src/external/amd_comgr/LICENSE.txt
    DESTINATION licenses RENAME AMD-COMGR-LICENSE.txt)
install(FILES src/external/roctx/LICENSE.md
    DESTINATION licenses RENAME ROCTX-LICENSE.md)
install(FILES src/external/roctx/roctx.h
    DESTINATION include)

set(CPACK_GENERATOR ZIP)
# Name the archive after the VERSION file, not PROJECT_VERSION, so that the
# archive name matches what the VERSION file and `ktrace.exe --version` report.
set(CPACK_PACKAGE_VERSION "${KTRACE_VERSION_STRING}")
set(CPACK_PACKAGE_FILE_NAME
    "ktrace-${KTRACE_VERSION_STRING}-windows-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
include(CPack)
