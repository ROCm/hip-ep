##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# Use the flatc that matches the resolved flatbuffers (source-built or imported
# via find_package), so the generator version stays in lock-step with the
# library. Only fall back to a standalone flatc on disk when no target exists.
# (TheRock bundles an older flatc that rejects the string-default schemas, so
# its bin/ must NOT be preferred when a matching flatc target is available.)
if(TARGET flatbuffers::flatc)
    set(FLATC_EXECUTABLE flatbuffers::flatc)
elseif(TARGET flatc)
    set(FLATC_EXECUTABLE flatc)
else()
    find_program(FLATC_EXECUTABLE flatc HINTS "${THEROCK_DIST}/bin" REQUIRED)
endif()
message(STATUS "Using flatc: ${FLATC_EXECUTABLE}")

# Generate C++ headers from .fbs schemas.
#
# Schemas do not all live here: gqa_autotune.fbs sits next to the code that
# reads it (lib/Runtime/Kernels/hip/autotune/gqa/) so the whole GQA autotune
# story -- schema, lookup, cost model -- is one directory. Generation stays
# centralised because every schema needs the same flatc invocation and the same
# generated-header include path.
set(FBS_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/compilation_options.fbs
    ${CMAKE_CURRENT_SOURCE_DIR}/model_metadata.fbs
    ${CMAKE_SOURCE_DIR}/lib/Runtime/Kernels/hip/autotune/gqa/gqa_autotune.fbs
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
add_custom_target(HipSchemas ALL DEPENDS ${GENERATED_HEADERS})

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
namespace mlir {
namespace hip {
inline const char* k_${schema_name}_schema() {
  static const char kSchema[] = R\"fbs(${schema_content})fbs\";
  return kSchema;
}
} // namespace hip
} // namespace mlir
"
        NEWLINE_STYLE LF
    )
endforeach()

# Interface library: consumers depend on this to get generated headers
# and the flatbuffers include path (for flatbuffers/flatbuffers.h and
# flatbuffers/idl.h).
add_library(HipSchemasLib INTERFACE)
add_dependencies(HipSchemasLib HipSchemas)
target_include_directories(HipSchemasLib INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(HipSchemasLib INTERFACE
    flatbuffers::flatbuffers
)
