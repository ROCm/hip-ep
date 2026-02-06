##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# `/Zc:__cplusplus`: This option ensures that the `__cplusplus`
# macro reflects the correct version of the C++ standard used by the
# compiler. By default, MSVC might not update this macro correctly
# to reflect the C++ standard version. This option forces the
# compiler to update the macro appropriately, which can be crucial
# for conditional compilation depending on the C++ standard version.

# `/Zi`: This option enables the generation of complete debugging
# information. It allows for the creation of a PDB (Program
# Database) file, which stores debugging and project state
# information. The PDB file is used by debuggers to provide
# source-level debugging, including setting breakpoints, stepping
# through code, and inspecting variables.

# `/Qspectre`: This option enables mitigations against the Spectre
# vulnerability, a hardware vulnerability that affects modern
# microprocessors that perform branch prediction. By enabling this
# option, the compiler will generate code that is protected against
# this class of vulnerabilities, at the potential cost of some
# performance overhead.

# `/ZH:SHA_256`: This option specifies the hash algorithm used for
# generating content hashes in the PDB file. Setting it to `SHA_256`
# uses the SHA-256 algorithm, which is more secure than the default
# MD5, providing better protection against hash collision attacks.

# `/guard:cf`: This option enables Control Flow Guard (CFG), a
# security feature that checks that the target of a call or jump is
# valid at runtime. This can help protect against attacks that
# attempt to hijack the control flow of the program. It adds a
# runtime check but can significantly increase the security of the
# application.

# `/sdl`: Stands for "Security Development Lifecycle". This option
# enables additional security checks and makes warnings more
# stringent. It's part of a broader approach to developing software
# that reduces vulnerabilities and security issues.

# Microsoft requested

# HISTORICAL NOTE: /Zi, /ZH:SHA_256, and /DEBUG were previously hardcoded here,
# which forced PDB generation for ALL build types including Release.
# This violated CMake best practices and caused significant performance issues:
#
# 1. Build Performance Impact:
#    - PDB generation is the primary bottleneck in MSVC linking (per Microsoft docs)
#    - Windows CI Release builds were 2.03x slower than Linux (6.85m vs 3.37m)
#    - /DEBUG disables /OPT:REF and /OPT:ICF optimizations by default
#
# 2. CMake Best Practices Violation:
#    - Release build type should NOT include debug symbols
#    - For Release + debug symbols, use RelWithDebInfo build type instead
#    - This is standard CMake convention across all platforms
#
# 3. Why These Were Removed:
#    - /Zi: Forces PDB generation (compile time overhead + linker merging overhead)
#    - /ZH:SHA_256: Only useful with /Zi, provides secure PDB hashing
#    - /DEBUG: Forces PDB linking + disables link-time optimizations
#
# If you need debug symbols in an optimized build, use:
#   cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ...
#
# References:
# - https://devblogs.microsoft.com/cppblog/the-visual-c-linker-best-practices-developer-iteration/
# - https://learn.microsoft.com/en-us/cpp/build/reference/debug-generate-debug-info
# - https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html

set(MORPHIZEN_COMPILER_OPTIONS
  /Zc:__cplusplus #
  # /Zi # REMOVED: Use RelWithDebInfo build type instead of hardcoding debug symbols
  /Qspectre # enable Spectre mitigations, required by MS
  # /ZH:SHA_256 # REMOVED: Only useful with /Zi flag
  /guard:cf # Control Flow Guard
  /sdl # Security Development Lifecycle
  /MP # build with multiple processes
  /W4
  /WX # treat warnings as errors (matches Linux -Werror)
  /EHsc
  # Enable additional warnings to match Linux strictness (-Wextra equivalent)
  /w14505 # unreferenced local function has been removed (matches -Wunused-function)
  /w14189 # local variable is initialized but not referenced (matches -Wunused-but-set-variable)
  /w14456 # declaration hides previous local declaration
  /w14457 # declaration hides function parameter
  /w14458 # declaration hides class member
  /w14459 # declaration hides global declaration
  /w14946 # reinterpret_cast used between related classes (helps catch aliasing issues)
  # TODO: fix the following warning
  /wd4251 #warning C4251: needs to have dll-interface to be used by clients of
  /wd4275 #warning C4275: non dll-interface class
  # Suppress C4996 deprecation warnings globally for glog v0.7.1 compatibility
  # glog v0.7.1 marks CustomPrefixCallback as [[deprecated]] in glog/logging.h header.
  # Since glog/logging.h is included throughout the entire MorphiZen codebase,
  # this deprecation warning appears in virtually every translation unit, even though
  # MorphiZen doesn't use CustomPrefixCallback directly. The deprecation is purely
  # a glog internal API change (CustomPrefixCallback -> PrefixFormatterCallback).
  # We suppress this warning globally because:
  # 1. MorphiZen code doesn't use the deprecated API
  # 2. The warning is triggered by header inclusion, not our code
  # 3. glog/logging.h is a fundamental dependency used everywhere
  # 4. Individual suppression per file would be impractical
  # This allows us to use glog v0.7.1 (which fixes critical CMake policy errors
  # requiring CMake >= 3.5) while maintaining our strict /W4 warning level.
  /wd4996
  /utf-8
  CACHE STRING "Compiler options for Morphizen"
)

# /WX is now enabled for all platforms in MORPHIZEN_COMPILER_OPTIONS above

set(MORPHIZEN_LINKER_OPTIONS
  # `/DEBUG`: This option instructs the linker to generate debug
  # information for the compiled binaries. This debug information is
  # crucial for debugging the application, as it maps the binary code
  # back to the source code, allowing developers to step through the
  # code, set breakpoints, and inspect variables during a debugging
  # session. The generated debug information is typically stored in a
  # PDB (Program Database) file.
  #
  # REMOVED: /DEBUG was hardcoded here, which forced PDB generation and disabled
  # link-time optimizations (/OPT:REF and /OPT:ICF) in Release builds.
  # Use RelWithDebInfo build type instead if you need debug symbols.

  # `/guard:cf`: This option enables Control Flow Guard (CFG) in the
  # linked binary. CFG is a security feature that helps protect
  # against attacks that attempt to hijack the control flow of the
  # program. It works by inserting runtime checks that validate the
  # target of indirect function calls, making it harder for an
  # attacker to execute arbitrary code through techniques like
  # return-oriented programming (ROP). Enabling CFG can significantly
  # enhance the security of the application by mitigating a class of
  # common exploits.

  # `/CETCOMPAT`: This option enables compatibility with Control-flow
  # Enforcement Technology (CET), a hardware-based security feature
  # designed to prevent certain types of attacks by enforcing
  # stricter control flow integrity. CET works by introducing new CPU
  # instructions that mark legitimate targets for indirect calls and
  # returns, effectively creating a shadow stack. This helps protect
  # against return-oriented programming (ROP) and call-oriented
  # programming (COP) attacks. Enabling `/CETCOMPAT` ensures that the
  # generated binary can take advantage of CET if it's supported by
  # the hardware, further enhancing the security of the application.
  # /DEBUG # REMOVED: Use RelWithDebInfo build type instead
  /DYNAMICBASE
  /ignore:4099 # ignore warning about PDB file not found (may still occur with dependencies)
  /ignore:4197 # ignore warning about /INCREMENTAL:NO
  CACHE STRING "Linker options for Morphizen"
)

if (WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  list(APPEND MORPHIZEN_LINKER_OPTIONS /CETCOMPAT)
endif()

# put all executables and dll files into a shared libary, make
# debugging easy.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
