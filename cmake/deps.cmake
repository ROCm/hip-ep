##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
include(FetchContent)

# Parse cmake/deps.txt into DEP_URL_<name> / DEP_HASH_<name>.
# Each non-comment line is `name;url;hash`. file(STRINGS) preserves the
# embedded ';' so each line is one list element we split with POP_FRONT.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/deps.txt" _HIPDNN_DEPS_LIST)
foreach(_dep IN LISTS _HIPDNN_DEPS_LIST)
  if(_dep MATCHES "^#" OR _dep STREQUAL "")
    continue()
  endif()
  list(POP_FRONT _dep _dep_name)
  list(POP_FRONT _dep _dep_url)
  set(DEP_URL_${_dep_name} "${_dep_url}")
  set(DEP_HASH_${_dep_name} "${_dep}")  # remaining column = hash (may be empty)
endforeach()

# The shared toolchain (LLVM/MLIR/LLD, flatbuffers, cpptrace, TheRock) is needed
# by both the EP and the standalone HIP tools, so it is resolved whenever either
# is on. The EP-only deps (morphizen, ONNX Runtime, protobuf) stay gated on
# BUILD_EP. Both options are declared in the top-level CMakeLists.txt before this
# file is included.
if(BUILD_EP OR BUILD_HIP_TOOLS)
  set(_HIPDNN_NEED_TOOLCHAIN ON)
else()
  set(_HIPDNN_NEED_TOOLCHAIN OFF)
endif()

# ===========================================================================
# Toolchain deps, resolved first so the EP-only section below and the top-level
# BUILD_HIP_TOOLS block reuse the same targets. find_package wins when a prefix
# is present (prebuilt locally, or CI's source-built install); the from-source
# fallback only fires in a fresh tree with nothing installed.
#
# TheRock, cpptrace and LLVM/MLIR/LLD are shared (gated _HIPDNN_NEED_TOOLCHAIN);
# the EP-only deps follow. flatbuffers is BUILD_HIP_TOOLS only and lives at the
# very end of this file.
# ===========================================================================

if(_HIPDNN_NEED_TOOLCHAIN)
  # TheRock ROCm SDK (real builds only -- mock has no GPU/HIP, so it neither
  # needs the SDK nor should pollute CMAKE_PREFIX_PATH with it). Explicit path
  # first, then auto-download fallback.
  #
  # THEROCK_DIST points to a TheRock ROCm SDK install and can be set via:
  #   1. CMake cache variable: -DTHEROCK_DIST=/path/to/therock
  #   2. Environment variable:  THEROCK_DIST=/path/to/therock
  #   3. Auto-downloaded from the pinned TheRock release when neither is set.
  # The resolved path is added to CMAKE_PREFIX_PATH so find_package(hip) finds it.
  if(NOT BUILD_MOCK_RUNTIME)
    set(THEROCK_DIST "" CACHE PATH "TheRock ROCm SDK distribution path")
    set(_therock_dist_resolved "${THEROCK_DIST}")
    if(_therock_dist_resolved)
      string(STRIP "${_therock_dist_resolved}" _therock_dist_resolved)
    elseif(DEFINED ENV{THEROCK_DIST})
      string(STRIP "$ENV{THEROCK_DIST}" _therock_dist_resolved)
    endif()
    if(_therock_dist_resolved AND NOT EXISTS "${_therock_dist_resolved}/bin")
      message(WARNING
        "THEROCK_DIST is set to '${_therock_dist_resolved}' but bin/ is missing; "
        "falling back to auto-download under the build tree.")
      set(_therock_dist_resolved "")
    endif()
    if(_therock_dist_resolved)
      set(THEROCK_DIST "${_therock_dist_resolved}"
          CACHE PATH "TheRock ROCm SDK distribution path" FORCE)
    else()
      set(_therock_root "${CMAKE_BINARY_DIR}/_therock")
      if(NOT EXISTS "${_therock_root}/bin")
        if(WIN32)
          set(_therock_windows_archs gfx1150 gfx1151 gfx1152 gfx1153)
          foreach(_arch IN LISTS HIP_ARCHITECTURES)
            if(NOT _arch IN_LIST _therock_windows_archs)
              message(FATAL_ERROR
                "HIP_ARCHITECTURES has '${_arch}', which the pinned TheRock "
                "bundle (${_therock_windows_archs}) does not cover. Provide "
                "your own SDK with -DTHEROCK_DIST=/path/to/therock.")
            endif()
          endforeach()
          set(_therock_base "${DEP_HASH_therock_windows}")
          set(_therock_url "${DEP_URL_therock_windows}/${_therock_base}.tar.gz")
        else()
          set(_therock_arch "${HIP_ARCHITECTURES}")
          if(_therock_arch MATCHES ";")
            list(GET _therock_arch 0 _therock_arch)
          endif()
          if(NOT _therock_arch)
            message(FATAL_ERROR
              "Cannot derive the TheRock tarball: set -DHIP_ARCHITECTURES=<gfxNNNN> "
              "(GPU arch), or provide -DTHEROCK_DIST=/path/to/therock.")
          endif()
          set(_therock_base "therock-dist-linux-${_therock_arch}-${DEP_HASH_therock_linux}")
          set(_therock_url "${DEP_URL_therock_linux}/${_therock_base}.tar.gz")
        endif()
        set(_therock_tgz "${CMAKE_BINARY_DIR}/${_therock_base}.tar.gz")
        if(NOT EXISTS "${_therock_tgz}")
          message(STATUS "THEROCK_DIST not provided; downloading ${_therock_url}")
          file(DOWNLOAD "${_therock_url}" "${_therock_tgz}" SHOW_PROGRESS STATUS _therock_dl)
          list(GET _therock_dl 0 _therock_dl_code)
          if(NOT _therock_dl_code EQUAL 0)
            file(REMOVE "${_therock_tgz}")
            message(FATAL_ERROR "TheRock download failed (${_therock_dl}). "
              "Set -DTHEROCK_DIST=/path/to/therock to use a local SDK instead.")
          endif()
        endif()
        file(MAKE_DIRECTORY "${_therock_root}")
        message(STATUS "Extracting TheRock SDK into ${_therock_root} ...")
        # tar (ships with Windows 10+/Linux) flattens the single top-level dir;
        # file(ARCHIVE_EXTRACT) has no --strip-components. Pass the archive as a
        # relative basename with WORKING_DIRECTORY so the path carries no
        # drive-letter colon: GNU tar would otherwise read "D:\..." as a remote
        # "host:path" (the reason --force-local was here), and Windows bsdtar
        # rejects --force-local outright. The basename form needs neither and
        # works for both. -C takes an absolute path unaffected by this parsing.
        execute_process(
          COMMAND tar -xzf "${_therock_base}.tar.gz"
                  -C "${_therock_root}" --strip-components=1
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
          RESULT_VARIABLE _therock_tar_rc)
        if(NOT _therock_tar_rc EQUAL 0)
          message(FATAL_ERROR "TheRock extraction failed (tar rc=${_therock_tar_rc}).")
        endif()
      endif()
      set(THEROCK_DIST "${_therock_root}"
          CACHE PATH "TheRock ROCm SDK distribution path" FORCE)
      message(STATUS "[onnx-hipdnn-ep] TheRock auto-downloaded into ${THEROCK_DIST}")
    endif()

    # Add to CMAKE_PREFIX_PATH so find_package(hip) resolves it -- single point
    # for both the explicitly-provided path and the auto-downloaded SDK.
    if(THEROCK_DIST)
      string(STRIP "${THEROCK_DIST}" THEROCK_DIST)
      message(STATUS "[onnx-hipdnn-ep] THEROCK_DIST: ${THEROCK_DIST}")
      list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}")
      list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")
    endif()
  endif()

  # cpptrace for crash backtraces (EP morphizen + standalone hip-compiler)
  set(_saved_bsl_cpptrace ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  if(WIN32)
    set(CPPTRACE_UNWIND_WITH_WINAPI ON CACHE BOOL "" FORCE)
    set(CPPTRACE_UNWIND_WITH_DBGHELP OFF CACHE BOOL "" FORCE)
  endif()
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY ${DEP_URL_cpptrace}
    GIT_TAG ${DEP_HASH_cpptrace}
  )
  FetchContent_MakeAvailable(cpptrace)
  set(BUILD_SHARED_LIBS ${_saved_bsl_cpptrace})

  # LLVM/MLIR/LLD resolution.
  #
  # Tier 1 (preferred): find_package against an installed prefix (a prebuilt
  #   LLVM+MLIR+LLD, or CI's source-built install). HIPDNN_LLVM_EMBEDDED stays
  #   unset, so the standard LLVM_*/MLIR_* consumer variables come from the
  #   package configs.
  # Tier 2 (fallback): build LLVM/MLIR/LLD from source as a FetchContent
  #   subdirectory. The build tree has no consumable package config during the
  #   same configure, so we set the consumer variables by hand and rely on the
  #   in-tree-defined helper functions (mlir_tablegen, llvm_map_components_to_libnames,
  #   add_mlir_dialect, ...). See llvm/docs/CMake.rst + the FOSDEM MLIR-dialect talk.
  #
  # The sticky HIPDNN_LLVM_EMBEDDED flag makes reconfigures of a from-source tree
  # safe: morphizen's deps.cmake caches MLIR_DIR/LLVM_DIR pointing at the build
  # tree, and a plain find_package(MLIR) on the next configure would then fail in
  # the build-tree LLVMConfig includes -- so once embedded we skip find_package
  # entirely and clear those stale cache entries.
  if(HIPDNN_LLVM_EMBEDDED)
    unset(MLIR_DIR CACHE)
    unset(LLVM_DIR CACHE)
  endif()

  if(NOT HIPDNN_LLVM_EMBEDDED)
    find_package(MLIR CONFIG QUIET)
  endif()

  if(MLIR_FOUND AND NOT HIPDNN_LLVM_EMBEDDED)
    find_package(LLVM REQUIRED CONFIG)
  else()
    message(STATUS "LLVM/MLIR not found; building from source (${DEP_HASH_llvm})")
    # clang is built in-tree so a from-source bootstrap is fully self-contained:
    # lib/Runtime gets a version-matched clang for runtime bitcode with no
    # external dependency. Kept identical to the CI LLVM build so the prefix that
    # CI caches (find_package path) and this fallback produce equivalent toolsets.
    set(LLVM_ENABLE_PROJECTS "clang;mlir;lld" CACHE STRING "" FORCE)
    set(LLVM_TARGETS_TO_BUILD "X86" CACHE STRING "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(LLVM_INSTALL_UTILS ON CACHE BOOL "" FORCE)  # FileCheck/not/count for LIT
    FetchContent_Declare(llvm-project
      GIT_REPOSITORY ${DEP_URL_llvm}
      GIT_TAG ${DEP_HASH_llvm}
      GIT_SHALLOW TRUE
      SOURCE_SUBDIR llvm
      EXCLUDE_FROM_ALL)
    # Build the in-tree LLVM/MLIR/clang with hidden ELF visibility (source
    # hardening, scoped to the LLVM sub-build only). Defence-in-depth on top of
    # the per-shared-library version scripts: with default visibility the
    # statically-linked llvm:: symbols are exported into the global dynamic
    # symbol table, where ROCm's libamd_comgr.so binds its own (versioned
    # @LLVM_22.0) llvm:: references to our ABI-incompatible upstream-LLVM copy
    # and segfaults during its in-process device-code compile. Hidden visibility
    # keeps them out of .dynsym entirely. The presets are restored immediately
    # after so our own targets (the EP entry points etc.) keep default
    # visibility and export normally.
    set(_hipdnn_saved_cxx_visibility "${CMAKE_CXX_VISIBILITY_PRESET}")
    set(_hipdnn_saved_c_visibility "${CMAKE_C_VISIBILITY_PRESET}")
    set(_hipdnn_saved_inlines_hidden "${CMAKE_VISIBILITY_INLINES_HIDDEN}")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
    FetchContent_MakeAvailable(llvm-project)
    # Restore. unset() when the original was empty -- CMake rejects an empty
    # string as a *_VISIBILITY_PRESET value ("unsupported value \"\"").
    if(_hipdnn_saved_cxx_visibility STREQUAL "")
      unset(CMAKE_CXX_VISIBILITY_PRESET)
    else()
      set(CMAKE_CXX_VISIBILITY_PRESET "${_hipdnn_saved_cxx_visibility}")
    endif()
    if(_hipdnn_saved_c_visibility STREQUAL "")
      unset(CMAKE_C_VISIBILITY_PRESET)
    else()
      set(CMAKE_C_VISIBILITY_PRESET "${_hipdnn_saved_c_visibility}")
    endif()
    if(_hipdnn_saved_inlines_hidden STREQUAL "")
      unset(CMAKE_VISIBILITY_INLINES_HIDDEN)
    else()
      set(CMAKE_VISIBILITY_INLINES_HIDDEN "${_hipdnn_saved_inlines_hidden}")
    endif()

    # Embedded (subdirectory) LLVM. Do NOT find_package the build tree: a
    # sub-build in the same configure has no consumable package config yet (its
    # LLVMConfig.cmake include()s exports that are not materialized during the
    # same run -- this is the canonical reason the "embedded" path skips
    # find_package; see llvm/docs/CMake.rst + FOSDEM MLIR-dialect talk). The
    # in-tree build already defined the MLIR/LLVM targets and the helper
    # functions (mlir_tablegen, llvm_map_components_to_libnames, add_mlir_dialect,
    # ...) globally, so downstream just needs the consumer variables set by hand.
    set(HIPDNN_LLVM_EMBEDDED ON CACHE BOOL "LLVM provided via FetchContent subdirectory" FORCE)
    set(LLVM_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include" CACHE PATH "" FORCE)
    set(MLIR_INCLUDE_DIRS
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include" CACHE PATH "" FORCE)
    set(LLVM_CMAKE_DIR "${llvm-project_SOURCE_DIR}/llvm/cmake/modules" CACHE PATH "" FORCE)
    set(MLIR_CMAKE_DIR "${llvm-project_SOURCE_DIR}/mlir/cmake/modules" CACHE PATH "" FORCE)
    # Tool dirs so downstream find_program (e.g. lib/Runtime's llvm-link/opt
    # lookup) resolves against the in-tree LLVM build. clang is always built
    # in-tree (clang;mlir;lld), so lib/Runtime's runtime-bitcode step uses the
    # version-matched in-tree clang target.
    set(LLVM_BINARY_DIR "${llvm-project_BINARY_DIR}" CACHE PATH "" FORCE)
    set(LLVM_TOOLS_BINARY_DIR "${llvm-project_BINARY_DIR}/bin" CACHE PATH "" FORCE)
    # LLD headers (lld/Common/Driver.h) live in a separate source tree; the
    # prebuilt path gets them from the merged LLVM include prefix, but the
    # subdirectory layout keeps them under lld/include + the generated build dir.
    include_directories(SYSTEM
      "${llvm-project_SOURCE_DIR}/llvm/include"
      "${llvm-project_BINARY_DIR}/include"
      "${llvm-project_SOURCE_DIR}/mlir/include"
      "${llvm-project_BINARY_DIR}/tools/mlir/include"
      "${llvm-project_SOURCE_DIR}/lld/include"
      "${llvm-project_BINARY_DIR}/tools/lld/include")
  endif()
endif()  # _HIPDNN_NEED_TOOLCHAIN (shared toolchain)

# ===========================================================================
# EP-only deps (BUILD_EP): morphizen build setup, ONNX Runtime, protobuf, then
# add_subdirectory(morphizen). protobuf/ORT are resolved before the
# subdirectory so morphizen reuses the same targets.
# ===========================================================================

if(BUILD_EP)
  function(morphizen_add_version_info)
    set(options)
    set(oneValueArgs COMPONENT DIR)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                          ${ARGN})
    # Query git directly instead of testing for a .git directory: a linked
    # worktree has .git as a file, and that test would reject it.
    execute_process(
      COMMAND "git" rev-parse HEAD
      WORKING_DIRECTORY "${ARG_DIR}"
      OUTPUT_VARIABLE TMP_GIT_COMMIT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND "git" describe --tags --abbrev=1 HEAD
      WORKING_DIRECTORY "${ARG_DIR}"
      OUTPUT_VARIABLE TMP_VERSION
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT TMP_GIT_COMMIT)
      set(TMP_GIT_COMMIT "N/A")
    endif()
    if(NOT TMP_VERSION)
      set(TMP_VERSION "N/A")
    endif()
    set(COMP_GIT_COMMIT ${TMP_GIT_COMMIT} PARENT_SCOPE)
    set(COMP_VERSION ${TMP_VERSION} PARENT_SCOPE)
    message(STATUS "FindPackage Version info: ${ARG_COMPONENT}=${TMP_GIT_COMMIT} ${TMP_VERSION}")
  endfunction()

  ## MorphiZen is vendored in-tree as a git subtree under morphizen.
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/morphizen/CMakeLists.txt")
    message(FATAL_ERROR "MorphiZen sources not found under morphizen (expected as an in-tree git subtree).")
  endif()
  message(STATUS "Using MorphiZen subtree: morphizen")

  # Force static linking for glog to avoid runtime library conflicts
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
  set(GLOG_BUILD_SHARED OFF CACHE BOOL "Build glog shared library" FORCE)

  # MorphiZen EP build settings, forced before add_subdirectory consumes them.
  set(morphizen_ENABLE_RYZENAI_BIN_METADATA OFF CACHE BOOL "Disable ryzenai_bin_metadata" FORCE)
  set(morphizen_OUTPUT_NAME "hipgpu" CACHE STRING "Set output name" FORCE)
  set(MORPHIZEN_EP_REGISTRATION_NAME "hipgpu" CACHE STRING "EP registration name for ORT" FORCE)
  set(MORPHIZEN_JSON_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/etc/morphizen_config.json")

  # ONNX Runtime resolution (find_package first, official release zip fallback).
  #
  # The EP links onnxruntime::onnxruntime (headers + import lib). We resolve it
  # here, BEFORE add_subdirectory(morphizen), so morphizen's
  # find_onnxruntime.cmake -- guarded by `if(NOT TARGET onnxruntime::onnxruntime)`
  # -- reuses our target instead of running its own find_package.
  #
  # Tier 1: find_package against CMAKE_PREFIX_PATH (CI installs a source-built
  #   ORT, possibly carrying unreleased patches, into its prefix -> hits here).
  # Tier 2: download the official microsoft/onnxruntime release zip (public host)
  #   and synthesize the IMPORTED target. ORT is NOT built as a subdirectory:
  #   its CMake vendors its own protobuf/abseil/flatbuffers/onnx at versions that
  #   collide with ours, so a source subbuild is not viable (download only).
  find_package(onnxruntime CONFIG QUIET)
  if(NOT onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    # deps.txt pins a single onnxruntime row (release base URL + version in the
    # hash column); derive the per-OS release archive name from the host.
    set(ORT_VERSION "${DEP_HASH_onnxruntime}")
    if(WIN32)
      set(_ort_archive "onnxruntime-win-x64-${ORT_VERSION}.zip")
    else()
      set(_ort_archive "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    endif()
    set(_ort_url "${DEP_URL_onnxruntime}/v${ORT_VERSION}/${_ort_archive}")

    message(STATUS "onnxruntime not found via find_package; fetching ${_ort_url}")
    FetchContent_Declare(onnxruntime URL "${_ort_url}")
    FetchContent_MakeAvailable(onnxruntime)

    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    if(WIN32)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
    else()
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
        IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
    endif()

    # check-ort-version.cmake reads this; the synthesized path must set it
    # explicitly or the version gate degrades to a WARNING.
    set(onnxruntime_VERSION "${ORT_VERSION}")
    message(STATUS "onnxruntime::onnxruntime synthesized from release zip (v${ORT_VERSION})")
  endif()

  # Version info baked into the EP. Written after ORT resolution because
  # onnxruntime_VERSION is only set there.
  morphizen_add_version_info(COMPONENT hip-ep DIR "${CMAKE_SOURCE_DIR}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
       "hip-ep;${COMP_GIT_COMMIT};${COMP_VERSION}\n"
       "onnxruntime;;${onnxruntime_VERSION}\n")
  set(MORPHIZEN_VERSION_INFO_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.txt")

  # protobuf (+ bundled abseil). Name "Protobuf" matches morphizen's
  # FetchContent_Declare so the first-populated wins and morphizen reuses it.
  # CMAKE_CXX_STANDARD=17 is required (abseil pins its installed options.h ABI
  # from a configure-time _MSVC_LANG probe; C++14 default -> CopyToEncodedBuffer
  # link error). See windows-build.yml "Build protobuf from source".
  find_package(Protobuf CONFIG QUIET)
  if(NOT Protobuf_FOUND AND NOT TARGET protobuf::libprotobuf)
    message(STATUS "protobuf not found; building from source (${DEP_HASH_protobuf})")
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(protobuf_INSTALL ON CACHE BOOL "" FORCE)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    # SYSTEM marks protobuf's (and bundled abseil's) include dirs as system
    # headers so their C4100/C4127/etc. warnings don't trip morphizen-core's
    # /W4 /WX. Mirrors morphizen's own protobuf FetchContent_Declare.
    FetchContent_Declare(Protobuf
      SYSTEM
      GIT_REPOSITORY ${DEP_URL_protobuf}
      GIT_TAG ${DEP_HASH_protobuf}
      GIT_SHALLOW TRUE
      GIT_SUBMODULES_RECURSE TRUE
      EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(Protobuf)
    # morphizen/unit-test/.../proto.cmake calls find_package(Protobuf CONFIG
    # REQUIRED). FetchContent exposes protobuf:: targets but does not always
    # populate Protobuf_DIR for a nested CONFIG-mode find.
    if(TARGET protobuf::libprotobuf)
      if(EXISTS "${Protobuf_BINARY_DIR}/protobuf-config.cmake")
        set(Protobuf_DIR "${Protobuf_BINARY_DIR}" CACHE PATH
            "Protobuf package config (FetchContent)" FORCE)
      elseif(EXISTS "${Protobuf_BINARY_DIR}/cmake/protobuf-config.cmake")
        set(Protobuf_DIR "${Protobuf_BINARY_DIR}/cmake" CACHE PATH
            "Protobuf package config (FetchContent)" FORCE)
      endif()
    endif()
  endif()

  # Add morphizen subdirectory.
  #
  # GCC -Wconversion on protobuf >=22 *.pb.h accessors is suppressed at the
  # root by marking the generated-header BINARY_DIR as SYSTEM in
  # morphizen-{core-static,pattern}; the SYSTEM propagation handles all
  # transitive consumers, so no parent-side -Werror override is needed.
  # onnx-ir-imp still needs a per-target `-Wno-error=conversion` because
  # its .pb.h surface comes from external `onnx_proto` whose
  # INTERFACE_INCLUDE_DIRECTORIES we don't control (upstream onnx fix is a
  # follow-up).
  #
  # Cross-wire BUILD_MOCK_RUNTIME (this project) <-> morphizen's HIP GPU
  # allocator option so the same build invocation does the right thing on
  # both sides:
  #
  #   * Real build (BUILD_MOCK_RUNTIME=OFF): we want morphizen's HIP allocator,
  #     which means find_package(hip) needs HIP_PLATFORM seeded *before*
  #     add_subdirectory(morphizen) below; otherwise TheRock's
  #     hip-config.cmake errors out with "Unexpected HIP_PLATFORM:".
  #     (lib/Runtime/Kernels/cmake/hip_utils.cmake seeds it too, but
  #     that subdir is added later in the top-level CMakeLists.txt.)
  #     Morphizen's option default is already ON, so we don't have to FORCE
  #     it -- but we don't actively turn it off either.
  #
  #   * Mock build (BUILD_MOCK_RUNTIME=ON, the project default): the toolchain
  #     by definition has no ROCm SDK, so find_package(hip REQUIRED) inside
  #     morphizen's ort-bridge would fail at configure time. Force morphizen's
  #     allocator OFF so the EP DLL still compiles in mock mode (it just won't
  #     register a HIP-backed allocator -- which is fine, mock can't run on
  #     GPU anyway). Morphizen's own configure-time WARNING for ORT_BRIDGE +
  #     ALLOCATOR=OFF is the expected, advertised-behavior signal here.
  if(BUILD_MOCK_RUNTIME)
    set(morphizen_ENABLE_HIP_GPU_ALLOCATOR OFF CACHE BOOL "disabled in mock builds (no ROCm SDK available)" FORCE)
  else()
    if(NOT DEFINED HIP_PLATFORM)
      set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
    endif()
  endif()

  # NOTE: add_subdirectory(morphizen) moved to main CMakeLists.txt to ensure
  # it's processed after lib/Runtime (which provides hipdnn-ep-dd target)
endif()  # BUILD_EP

# ===========================================================================
# BUILD_HIP_TOOLS-only deps.
# ===========================================================================

# flatbuffers (the schemas/ + lib/* targets consume flatbuffers::flatbuffers +
# flatc; no EP-side target links it -- the EP loads hip-compiler.dll, which
# carries flatbuffers, at runtime).
# Version-pinned: the schemas use string field defaults under --gen-object-api,
# which require flatc >= the pinned version. An older flatbuffers that happens
# to sit on CMAKE_PREFIX_PATH (e.g. TheRock bundles an older one) is rejected
# here so the from-source build provides a new-enough flatc.
if(BUILD_HIP_TOOLS)
string(REGEX REPLACE "^v" "" _fb_version "${DEP_HASH_flatbuffers}")
find_package(flatbuffers ${_fb_version} CONFIG QUIET)
if(NOT flatbuffers_FOUND AND NOT TARGET flatbuffers::flatbuffers)
  message(STATUS "flatbuffers >= ${_fb_version} not found; building from source")
  set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
  set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)
  FetchContent_Declare(flatbuffers
    GIT_REPOSITORY ${DEP_URL_flatbuffers}
    GIT_TAG ${DEP_HASH_flatbuffers}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(flatbuffers)
  # The source build defines the plain `flatbuffers`/`flatc` targets but NOT the
  # `flatbuffers::` namespaced aliases that the installed package config exports.
  # Create them so downstream guards (`if(NOT TARGET flatbuffers::flatbuffers)`)
  # and consumers see the from-source build instead of falling back to an older
  # flatbuffers on CMAKE_PREFIX_PATH (e.g. TheRock's).
  if(NOT TARGET flatbuffers::flatbuffers AND TARGET flatbuffers)
    add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
  endif()
  if(NOT TARGET flatbuffers::flatc AND TARGET flatc)
    add_executable(flatbuffers::flatc ALIAS flatc)
  endif()
endif()
endif()  # BUILD_HIP_TOOLS (flatbuffers)
