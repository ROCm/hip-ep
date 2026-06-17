##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT BUILD_EP)
  message(WARNING "BUILD_PYTHON_WHEEL requires BUILD_EP=ON; skipping wheel target")
  return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Build out-of-source: never write into the source python/ tree. The package is
# staged under the build dir, the wheel lands in <build>/python/dist.
set(_pkg_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/pkg")
# Native artifacts are staged into the onnxruntime/capi namespace package so the
# wheel ships them at onnxruntime/capi/ (platlib), installing next to
# onnxruntime.dll cross-platform.
set(_libs_dir "${_stage}/onnxruntime/capi")
set(_dist_dir "${CMAKE_CURRENT_BINARY_DIR}/dist")

# GPU arch this wheel targets, for the rocm --extra-index-url hint in
# pyproject.toml. Take the first if HIP_ARCHITECTURES is a list; fall back to
# gfx1151 when unset (e.g. mock builds).
set(MORPHIZEN_HIP_ARCH "${HIP_ARCHITECTURES}")
if(MORPHIZEN_HIP_ARCH MATCHES ";")
  list(GET MORPHIZEN_HIP_ARCH 0 MORPHIZEN_HIP_ARCH)
endif()
if(NOT MORPHIZEN_HIP_ARCH)
  set(MORPHIZEN_HIP_ARCH "gfx1151")
endif()

# Inject the project version + target arch into the staged pyproject.toml.
configure_file(
  "${_pkg_src}/pyproject.toml.in"
  "${_stage}/pyproject.toml"
  @ONLY)

# The EP shared library target is created by morphizen via deps.cmake; its name
# is the (git-versioned) unique id, output name onnxruntime_morphizen_ep.
set(_ep_target "${morphizen_CORE_DYNAMIC_UNIQUE_ID}")
if(NOT TARGET ${_ep_target})
  message(FATAL_ERROR
    "EP target '${_ep_target}' not found; cannot build the Python wheel.")
endif()

# The JIT compiler plugin (hip-compiler.dll) is loaded at runtime by the EP via
# Plugin::get("hip-compiler"); it MUST ship in the wheel or model compilation
# falls back to whatever (possibly stale) hip-compiler is on PATH.
set(_dll_args "--dll" "$<TARGET_FILE:${_ep_target}>")
set(_wheel_deps ${_ep_target})
if(TARGET hip-compiler-dll)
  list(APPEND _dll_args "--dll" "$<TARGET_FILE:hip-compiler-dll>")
  list(APPEND _wheel_deps hip-compiler-dll)
else()
  message(WARNING
    "hip-compiler-dll target not found (BUILD_HIP_TOOLS=OFF?); the wheel will "
    "not be able to JIT-compile models. Build with BUILD_HIP_TOOLS=ON.")
endif()

if(WIN32)
  set(_with_crt "--with-crt")
else()
  set(_with_crt "")
endif()

# Bundle the per-arch custom-kernels runtime libraries. The EP dlopens
# custom_kernels_<arch>.{dll,so} from its own directory at JIT init, so each
# must ship in onnxruntime/capi next to the EP DLL. ROCm import libs
# (amdhip64/MIOpen/hipBLASLt) come from the pip rocm[devel] package at runtime.
foreach(_arch IN LISTS HIP_ARCHITECTURES)
  set(_ck_target custom_kernels_${_arch})
  if(TARGET ${_ck_target})
    list(APPEND _dll_args "--dll" "$<TARGET_FILE:${_ck_target}>")
    list(APPEND _wheel_deps ${_ck_target})
  else()
    message(WARNING
      "custom kernels target '${_ck_target}' not found; the wheel cannot "
      "JIT-load GPU kernels for ${_arch}.")
  endif()
endforeach()

add_custom_target(wheel ALL
  # 1. Stage the package sources into the build dir (keep source tree clean).
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${_pkg_src}/onnxruntime_morphizen_ep"
          "${_stage}/onnxruntime_morphizen_ep"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${_pkg_src}/setup.py"
          "${_stage}/"
  # 2. Stage native artifacts (EP lib + hip-compiler + per-arch custom-kernels
  #    runtime DLLs/SOs + CRT import libs) into the onnxruntime/capi namespace
  #    package dir so the wheel ships them at onnxruntime/capi/ (platlib).
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_libs_dir}"
  COMMAND ${Python3_EXECUTABLE} "${_pkg_src}/_pack_libs.py"
          ${_dll_args}
          --dest "${_libs_dir}"
          ${_with_crt}
  # 3. Build the wheel (PEP 517). Drop setuptools' build cache + egg-info first:
  #    they linger across runs and would re-pack stale (e.g. removed) artifacts.
  #    --no-deps: do not pull deps here.
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "${_stage}/build" "${_stage}/onnxruntime_morphizen_ep.egg-info"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_dist_dir}"
  COMMAND ${Python3_EXECUTABLE} -m pip wheel "${_stage}"
          --no-deps --wheel-dir "${_dist_dir}"
  DEPENDS ${_wheel_deps}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Building onnxruntime_morphizen_ep wheel -> ${_dist_dir}"
  VERBATIM)

message(STATUS "Python wheel target 'wheel' configured (out: ${_dist_dir})")
