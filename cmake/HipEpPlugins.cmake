##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# Static compiler-plugin registration (IREE-style).
#
# MLIR-contributing plugins (passes, dialects/ops, lowering + bufferization
# interface models) are linked STATICALLY into the host at configure time --
# NOT dlopen'd. This is the model the MLIR ecosystem converges on (IREE's
# `-DIREE_COMPILER_PLUGINS=`) and it works identically on Windows and Linux:
# no `-rdynamic`, no per-plugin `.def`, no PE export-table cap, no symbol
# export at all. See docs/design/plugin-interface.md, "Linkage model".
#
# Mechanism:
#   * `HIPDNN_EP_COMPILER_PLUGINS` (cache) is the semicolon-separated list of
#     plugin ids to link into this build. Default empty => no plugins (the
#     production default; the dispatch is a no-op and nothing is linked).
#   * A plugin package calls `hipdnn_ep_compiler_plugin_register(PLUGIN_ID <id>
#     TARGET <static-lib-target>)` to make itself AVAILABLE. Being available is
#     not being selected -- only ids listed in `HIPDNN_EP_COMPILER_PLUGINS` are
#     actually linked.
#   * `hipdnn_ep_finalize_static_plugins()` (called once, after every plugin
#     package has registered) generates
#     `${CMAKE_BINARY_DIR}/include/hip/Compiler/StaticLinkedPlugins.inc` -- one
#     `HANDLE_PLUGIN_ID(<id>)` line per selected plugin -- and links the
#     selected plugin static libs into `LibHipCompiler`.
#   * `lib/Compiler/StaticPlugins.cpp` (in LibHipCompiler) includes that .inc
#     twice: once to `extern "C"`-declare each `hipEpRegisterPlugin_<id>`, once
#     to CALL each. The explicit call is what keeps the linker from GC-dropping
#     the plugin object (`--gc-sections` / `/OPT:REF`) -- never rely on a static
#     initializer for registration.
#
# A plugin's per-id entry point is a single extern "C" function:
#
#     extern "C" void hipEpRegisterPlugin_<id>(hip::compiler::HipEpPluginRegistry &R);
#
# (the static-linking analogue of the old dynamic `hipEpGetPluginInfo`).
# ============================================================================

set(HIPDNN_EP_COMPILER_PLUGINS "" CACHE STRING
    "Semicolon-separated list of compiler-plugin ids to statically link into \
the host (empty = none). Each id must be registered by a plugin package via \
hipdnn_ep_compiler_plugin_register().")

define_property(GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_IDS
    BRIEF_DOCS "Ids of all available (registered) compiler plugins"
    FULL_DOCS "Populated by hipdnn_ep_compiler_plugin_register(); the subset \
also listed in HIPDNN_EP_COMPILER_PLUGINS is what gets linked.")
define_property(GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_TARGETS
    BRIEF_DOCS "Static-lib targets for the registered compiler plugins (index-aligned with the ids)"
    FULL_DOCS "Populated by hipdnn_ep_compiler_plugin_register().")

set_property(GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_IDS "")
set_property(GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_TARGETS "")

# hipdnn_ep_compiler_plugin_register(PLUGIN_ID <id> TARGET <target>)
#
# Register a plugin as AVAILABLE. Call from the plugin package's CMakeLists
# after defining its static-lib target. Does not link anything by itself --
# selection is driven by HIPDNN_EP_COMPILER_PLUGINS at finalize.
function(hipdnn_ep_compiler_plugin_register)
  cmake_parse_arguments(ARG "" "PLUGIN_ID;TARGET" "" ${ARGN})
  if(NOT ARG_PLUGIN_ID OR NOT ARG_TARGET)
    message(FATAL_ERROR
      "hipdnn_ep_compiler_plugin_register requires PLUGIN_ID and TARGET")
  endif()
  set_property(GLOBAL APPEND PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_IDS "${ARG_PLUGIN_ID}")
  set_property(GLOBAL APPEND PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_TARGETS "${ARG_TARGET}")
endfunction()

# hipdnn_ep_finalize_static_plugins()
#
# Generate the StaticLinkedPlugins.inc registrar include and link the selected
# plugin static libs into LibHipCompiler. Call ONCE, after all plugin packages
# have registered (i.e. after their add_subdirectory calls).
function(hipdnn_ep_finalize_static_plugins)
  get_property(_ids GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_IDS)
  get_property(_tgts GLOBAL PROPERTY HIPDNN_EP_REGISTERED_PLUGIN_TARGETS)

  set(_handle_lines "")
  set(_selected_targets "")
  list(LENGTH _ids _n)
  if(_n GREATER 0)
    math(EXPR _last "${_n} - 1")
    foreach(_i RANGE ${_last})
      list(GET _ids ${_i} _id)
      list(GET _tgts ${_i} _tgt)
      if("${_id}" IN_LIST HIPDNN_EP_COMPILER_PLUGINS)
        string(APPEND _handle_lines "HANDLE_PLUGIN_ID(${_id})\n")
        list(APPEND _selected_targets "${_tgt}")
      endif()
    endforeach()
  endif()

  # file(GENERATE) writes at the end of the configure step (before any build),
  # and only rewrites when the content changes -- so StaticPlugins.cpp is
  # rebuilt exactly when the selected plugin set changes.
  file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/include/hip/Compiler/StaticLinkedPlugins.inc"
    CONTENT "${_handle_lines}")

  if(_selected_targets)
    if(NOT TARGET LibHipCompiler)
      message(FATAL_ERROR
        "hipdnn_ep_finalize_static_plugins: LibHipCompiler target not defined")
    endif()
    # PUBLIC so every host that links LibHipCompiler (tools + EP DLL) pulls in
    # the selected plugin objects; StaticPlugins.cpp's explicit call to each
    # hipEpRegisterPlugin_<id> keeps them from being GC-dropped.
    target_link_libraries(LibHipCompiler PUBLIC ${_selected_targets})
    add_dependencies(LibHipCompiler ${_selected_targets})
    message(STATUS "Static compiler plugins linked: ${_selected_targets}")
  else()
    message(STATUS "Static compiler plugins: none selected "
      "(HIPDNN_EP_COMPILER_PLUGINS='${HIPDNN_EP_COMPILER_PLUGINS}')")
  endif()
endfunction()
