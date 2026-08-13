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
      --cpp_out=dllexport_decl=MORPHIZEN_PROTO_DLL_SPEC:${CMAKE_CURRENT_BINARY_DIR}/morphizen
      -I ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/morphizen/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})

if(MSVC)
  set(MORPHIZEN_PROTO_DLL_SPEC "__declspec(dllexport)")
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES COMPILE_DEFINITIONS "MORPHIZEN_PROTO_DLL_SPEC=${MORPHIZEN_PROTO_DLL_SPEC}"
    COMPILE_FLAGS "/w")
else(MSVC)
  # Empty decoration for the generated protobuf classes only (the hand-written
  # API keeps __attribute__((visibility("default"))) via morphizen/export.h).
  # protobuf v34 emits `class <decl> [[gnu::warn_unused]] Name`, and GCC rejects
  # a GNU __attribute__ placed before a C++11 [[...]] attribute. Default ELF
  # visibility plus the `*morphizen*` version script already export these
  # symbols, so dropping the decoration here costs nothing. Must stay in sync
  # with MORPHIZEN_PROTO_DLL_SPEC in morphizen/export.h, which supplies the
  # macro to every other translation unit that includes the generated *.pb.h.
  set_source_files_properties(
    ${PROTO_SRCS}
    PROPERTIES
    COMPILE_DEFINITIONS
    "MORPHIZEN_PROTO_DLL_SPEC="
    COMPILE_FLAGS
    "-Wno-unused-variable -Wno-conversion")
endif(MSVC)
