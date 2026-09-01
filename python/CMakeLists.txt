##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

if(NOT WIN32)
  message(FATAL_ERROR
    "BUILD_PYTHON_WHEEL is Windows-only; re-configure without it.")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Build out-of-source: never write into the source python/ tree. The package is
# staged under the build dir, the wheel lands in <build>/python/dist.
set(_pkg_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/pkg")
set(_libs_dir "${_stage}/onnxruntime_ep_amdgpu")
set(_dist_dir "${CMAKE_CURRENT_BINARY_DIR}/dist")

set(MORPHIZEN_HIP_ARCH "${HIP_ARCHITECTURES}")
if(MORPHIZEN_HIP_ARCH MATCHES ";")
  list(GET MORPHIZEN_HIP_ARCH 0 MORPHIZEN_HIP_ARCH)
endif()
if(NOT MORPHIZEN_HIP_ARCH)
  set(MORPHIZEN_HIP_ARCH "gfx1151")
endif()

configure_file(
  "${_pkg_src}/pyproject.toml.in"
  "${_stage}/pyproject.toml"
  @ONLY)

# The EP shared library target is created by morphizen via deps.cmake; its name
# is the (git-versioned) unique id, output name hipgpu (morphizen_OUTPUT_NAME).
set(_ep_target "${morphizen_CORE_DYNAMIC_UNIQUE_ID}")
if(NOT TARGET ${_ep_target})
  message(FATAL_ERROR
    "EP target '${_ep_target}' not found; cannot build the Python wheel.")
endif()

# No separate JIT compiler library to bundle: the compiler is linked into the EP
# itself and resolved in-process through the static "hip-compiler" plugin
# registration.
set(_dll_args "--dll" "$<TARGET_FILE:${_ep_target}>")
set(_wheel_deps ${_ep_target})

if(NOT THEROCK_DIST)
  message(FATAL_ERROR
    "THEROCK_DIST is empty; the wheel cannot bundle the ROCm runtime.")
endif()

# Bundle family-common + per-arch-accel custom-kernels runtime libraries. The EP
# dlopens both from its own directory at JIT init, so each must ship in
# onnxruntime/capi next to the EP DLL.
set(_CK_FAMILY_GFX115X_MEMBERS gfx1150 gfx1151 gfx1152 gfx1153)
set(_CK_NEEDS_GFX115X FALSE)
foreach(_arch IN LISTS HIP_ARCHITECTURES)
  if(_arch IN_LIST _CK_FAMILY_GFX115X_MEMBERS)
    set(_CK_NEEDS_GFX115X TRUE)
  endif()
endforeach()

if(_CK_NEEDS_GFX115X AND TARGET custom_kernels_gfx115x)
  list(APPEND _dll_args "--dll" "$<TARGET_FILE:custom_kernels_gfx115x>")
  list(APPEND _wheel_deps custom_kernels_gfx115x)
endif()

foreach(_arch IN LISTS HIP_ARCHITECTURES)
  if(_arch IN_LIST _CK_FAMILY_GFX115X_MEMBERS)
    continue()
  endif()
  set(_ck_common_target custom_kernels_${_arch}_common)
  if(TARGET ${_ck_common_target})
    list(APPEND _dll_args "--dll" "$<TARGET_FILE:${_ck_common_target}>")
    list(APPEND _wheel_deps ${_ck_common_target})
  endif()
endforeach()

foreach(_arch IN LISTS HIP_ARCHITECTURES)
  set(_ck_target custom_kernels_${_arch})
  if(TARGET ${_ck_target})
    list(APPEND _dll_args "--dll" "$<TARGET_FILE:${_ck_target}>")
    list(APPEND _wheel_deps ${_ck_target})
  else()
    message(WARNING
      "custom kernels accel target '${_ck_target}' not found; the wheel cannot "
      "JIT-load GPU accel kernels for ${_arch}.")
  endif()
endforeach()

add_custom_target(wheel ALL
  # 1. Stage the package sources into the build dir (keep source tree clean).
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${_pkg_src}/onnxruntime_ep_hip"
          "${_stage}/onnxruntime_ep_hip"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${_pkg_src}/setup.py"
          "${_stage}/"
  # 2. Stage native artifacts (EP lib + per-arch custom-kernels runtime
  #    DLLs/SOs + CRT import libs) into the onnxruntime_ep_amdgpu namespace
  #    package dir so the wheel ships them at onnxruntime_ep_amdgpu/ (platlib).
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_libs_dir}"
  COMMAND ${Python3_EXECUTABLE} "${_pkg_src}/_pack_libs.py"
          ${_dll_args}
          --dest "${_libs_dir}"
          --rocm-dist "${THEROCK_DIST}"
          --rocm-arch "${MORPHIZEN_HIP_ARCH}"
          --with-crt
  # 3. Build the wheel (PEP 517). Drop setuptools' build cache + egg-info first:
  #    they linger across runs and would re-pack stale (e.g. removed) artifacts.
  #    --no-deps: do not pull deps here.
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          "${_stage}/build" "${_stage}/onnxruntime_ep_hip.egg-info"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_dist_dir}"
  COMMAND ${Python3_EXECUTABLE} -m pip wheel "${_stage}"
          --no-deps --wheel-dir "${_dist_dir}"
  DEPENDS ${_wheel_deps}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Building onnxruntime_ep_hip wheel -> ${_dist_dir}"
  VERBATIM)

message(STATUS "Python wheel target 'wheel' configured (out: ${_dist_dir})")
