# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.25)
project(HipEpQsaExperiment LANGUAGES C CXX)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/hip_utils.cmake)
if(NOT HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "The QSA experiment currently validates gfx1151 wave32 only")
endif()
hip_add_library(qsa_kernels_gfx1151 SHARED ../hip/qsa/qsa_port.hip
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/../include
  COMPILE_OPTIONS -std=c++17 -ffp-contract=off
  OFFLOAD_ARCHS gfx1151)
