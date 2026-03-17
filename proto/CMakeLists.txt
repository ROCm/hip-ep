##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
set(PROTO_FILES
  rocm.proto
)
set(PROTO_SRCS "")
set(PROTO_HDRS "")
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf CONFIG REQUIRED)
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto)
endif()
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)
  message(STATUS "generating proto --- ${PROTO_FILE_NAME}.pb.cc ${PROTO_FILE_NAME}.pb.h")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto/${PROTO_FILE_NAME}.pb.cc
      ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto/${PROTO_FILE_NAME}.pb.h
    COMMAND protobuf::protoc
    ARGS
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/rocm_proto
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
    DEPENDS ${PROTO_FILE})
  list(APPEND PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto/${PROTO_FILE_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto/${PROTO_FILE_NAME}.pb.h)
endforeach(PROTO_FILE ${PROTO_FILES})


add_library (rocm_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(rocm_proto PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/rocm_proto
)
target_link_libraries(rocm_proto PUBLIC protobuf::libprotobuf)
set_target_properties(rocm_proto
	PROPERTIES
	FOLDER "rocm")

if(NOT MSVC)
  target_compile_options(rocm_proto PUBLIC "-fPIC")
endif(NOT MSVC)


# Find flatc executable from the prebuilt package
find_program(FLATC_EXECUTABLE flatc
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    REQUIRED
)
message(STATUS "Using flatc: ${FLATC_EXECUTABLE}")

# Generate C++ headers from .fbs schemas
set(FBS_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/compilation_options.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/model_metadata.fbs
)

set(GENERATED_HEADERS "")
foreach(schema ${FBS_SCHEMAS})
    get_filename_component(schema_name ${schema} NAME_WE)
    set(generated_header ${CMAKE_CURRENT_BINARY_DIR}/${schema_name}_generated.h)
    add_custom_command(
        OUTPUT ${generated_header}
        COMMAND ${FLATC_EXECUTABLE} --cpp --gen-object-api --scoped-enums -o ${CMAKE_CURRENT_BINARY_DIR} ${schema}
        DEPENDS ${schema}
        COMMENT "Generating FlatBuffers header: ${schema_name}_generated.h"
    )
    list(APPEND GENERATED_HEADERS ${generated_header})
endforeach()

# Target that drives header generation
add_custom_target(UdnaSchemas ALL DEPENDS ${GENERATED_HEADERS})

# Embed each .fbs schema as a C++ string constant so consumers can use
# flatbuffers::Parser for JSON<->binary conversion without a separate schema file.
foreach(schema ${FBS_SCHEMAS})
    get_filename_component(schema_name ${schema} NAME_WE)
    file(READ ${schema} schema_content)
    file(CONFIGURE
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${schema_name}_schema.h
        CONTENT
"// Auto-generated from ${schema_name}.fbs — do not edit manually.
#pragma once
namespace udna {
inline const char* k_${schema_name}_schema() {
  static const char kSchema[] = R\"fbs(${schema_content})fbs\";
  return kSchema;
}
} // namespace udna
"
        NEWLINE_STYLE LF
    )
endforeach()

# Interface library: consumers depend on this to get generated headers
# and the flatbuffers include path (for flatbuffers/flatbuffers.h and
# flatbuffers/idl.h).
add_library(UdnaSchemasLib INTERFACE)
add_dependencies(UdnaSchemasLib UdnaSchemas)
target_include_directories(UdnaSchemasLib INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(UdnaSchemasLib INTERFACE
    flatbuffers::flatbuffers
)
