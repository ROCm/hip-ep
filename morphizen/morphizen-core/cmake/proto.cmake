##
# ** Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(PROTO_FILES
  src/config.proto
  src/anchor_point.proto
  src/capability.proto
  src/pass_context.proto
  src/version.proto
  src/model_compatibility.proto
)
set(PROTO_SRCS "")
set(PROTO_HDRS "")
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${PROTO_FILE_NAME}.pb.cc ${PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/morphizen
    COMMAND protobuf::protoc
    ARGS
      --proto_path=${protobuf_SOURCE_DIR}/src
      --cpp_out=dllexport_decl=MORPHIZEN_DLL_SPEC:${CMAKE_CURRENT_BINARY_DIR}/morphizen
      -I ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})

if(MSVC)
  set(MORPHIZEN_DLL_SPEC "__declspec(dllexport)")
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES COMPILE_DEFINITIONS "MORPHIZEN_DLL_SPEC=${MORPHIZEN_DLL_SPEC}"
    COMPILE_FLAGS "/w")
else(MSVC)
  # Leave the dllexport decoration empty on non-MSVC toolchains. protobuf v34's
  # generated code emits `class <DLL_SPEC> [[gnu::warn_unused]] Name`; a GNU
  # __attribute__ placed before the C++11 [[...]] attribute is rejected by GCC
  # (only accepted by clang). On Linux the default symbol visibility already
  # exports these proto symbols, so an empty decoration is both correct and
  # portable across GCC and clang hosts.
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES
    COMPILE_DEFINITIONS
    "MORPHIZEN_DLL_SPEC="
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-conversion")
endif(MSVC)
