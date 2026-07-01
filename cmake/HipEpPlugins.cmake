##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# Static compiler-plugin registration.
#
# A plugin that contributes MLIR (a pass, a dialect/op, an interface model) must
# land its registration in the host's process-global MLIR state -- one pass
# registry, one set of op TypeIDs. The only robust way to guarantee that is to
# make the plugin and host ONE binary: plugins are linked STATICALLY into the
# host at configure time, not loaded at runtime. This needs no symbol export and
# no dynamic loader, so it behaves identically on every platform. See
# docs/design/plugin-interface.md, "Linkage model".
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
#     to CALL each. The explicit call is essential: it is the reference that
#     keeps the linker's dead-code elimination (`--gc-sections` / `/OPT:REF`)
#     from discarding the plugin object -- registration must never depend on a
#     static initializer, which such a collected object would silently drop.
#
# A plugin's per-id entry point is a single extern "C" function:
#
#     extern "C" void hipEpRegisterPlugin_<id>(hip::compiler::HipEpPluginRegistry &R);
# ============================================================================

set(HIPDNN_EP_COMPILER_PLUGINS "" CACHE STRING
    "Semicolon-separated list of compiler-plugin ids to statically link into \
the host (empty = none). Each id must be registered by a plugin package via \
hipdnn_ep_compiler_plugin_register().")

set(HIPDNN_EP_COMPILER_PLUGIN_PATHS "" CACHE STRING
    "Semicolon-separated list of out-of-tree plugin source directories to add \
to this build. Each must contain a CMakeLists.txt that defines a plugin \
static-lib target and calls hipdnn_ep_compiler_plugin_register(). Use together \
with HIPDNN_EP_COMPILER_PLUGINS, which selects which registered ids get linked. \
This lets a downstream co-build its plugin from its own repo without vendoring \
it into this tree.")

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

# hipdnn_ep_add_external_plugins()
#
# add_subdirectory() every directory in HIPDNN_EP_COMPILER_PLUGIN_PATHS so its
# CMakeLists can define its plugin target and register it. Each external dir is
# built under ${CMAKE_BINARY_DIR}/external-plugins/<name> (an out-of-tree source
# dir needs an explicit binary dir). Call ONCE, before
# hipdnn_ep_finalize_static_plugins() so the registrations are visible to it.
function(hipdnn_ep_add_external_plugins)
  foreach(_dir ${HIPDNN_EP_COMPILER_PLUGIN_PATHS})
    get_filename_component(_dir "${_dir}" ABSOLUTE)
    if(NOT EXISTS "${_dir}/CMakeLists.txt")
      message(FATAL_ERROR
        "HIPDNN_EP_COMPILER_PLUGIN_PATHS entry has no CMakeLists.txt: ${_dir}")
    endif()
    get_filename_component(_name "${_dir}" NAME)
    message(STATUS "Adding out-of-tree compiler-plugin dir: ${_dir}")
    add_subdirectory("${_dir}" "${CMAKE_BINARY_DIR}/external-plugins/${_name}")
  endforeach()
endfunction()

# hipdnn_ep_finalize_static_plugins()
#
# Generate the StaticLinkedPlugins.inc registrar include and link the selected
# plugin static libs into LibHipCompiler. Call ONCE, after all plugin packages
# (in-tree and out-of-tree) have registered.
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
