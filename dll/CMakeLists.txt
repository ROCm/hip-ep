##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

message(STATUS "Building hip-compiler.dll")

# Create the DLL target
add_library(hip-compiler-dll SHARED
    dll_main.cpp
    hip-compiler.def
)

# Link against CInterface library (which pulls in all dependencies).
# WHOLE_ARCHIVE on HipCInterface: dll_main.cpp is empty outside Windows
# (the DllMain hook is Windows-only), so nothing in this DLL references
# the C-API symbols (hip_compile_with_fs, hip_get_version) directly.
# Without WHOLE_ARCHIVE, ld drops the entire archive and produces an empty
# .so — the EP then dlsym()s "hip_get_version" and aborts with
# "no such function: hip_get_version". On Windows hip-compiler.def already
# forces these exports, so this is Linux-specific (but harmless on Win).
target_link_libraries(hip-compiler-dll PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,HipCInterface>"
    HipSupport
)

# Include directories for C API headers
target_include_directories(hip-compiler-dll PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set output name (no prefix on Windows)
# Don't override RUNTIME_OUTPUT_DIRECTORY - let it use CMAKE_RUNTIME_OUTPUT_DIRECTORY variable
set_target_properties(hip-compiler-dll PROPERTIES
    OUTPUT_NAME "hip-compiler"
)

hipdnn_ep_apply_binary_compliance(hip-compiler-dll
    DESCRIPTION "AMD GPU Execution Provider JIT compiler")

# Localize statically-linked LLVM/MLIR/clang symbols so they are not exported
# into the global dynamic symbol table. If they are exported, ROCm's
# libamd_comgr.so binds its own (versioned @LLVM_22.0) llvm:: references to our
# ABI-incompatible upstream-LLVM copy at runtime and its in-process device-code
# compile segfaults. The version script keeps only the C API visible. Windows
# is naturally immune (PE doesn't export symbols by default).
if(NOT WIN32)
    target_link_options(hip-compiler-dll PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/hip-compiler.exports")
endif()

# Install DLL and import library
install(TARGETS hip-compiler-dll
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Install C API headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hip
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

message(STATUS "hip-compiler.dll configuration complete")
