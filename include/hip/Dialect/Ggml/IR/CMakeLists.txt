##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

set(LLVM_TARGET_DEFINITIONS GgmlDialect.td)
mlir_tablegen(GgmlDialect.h.inc -gen-dialect-decls -dialect=ggml)
mlir_tablegen(GgmlDialect.cpp.inc -gen-dialect-defs -dialect=ggml)
add_public_tablegen_target(GgmlDialectIncGen)

set(LLVM_TARGET_DEFINITIONS GgmlEnums.td)
mlir_tablegen(GgmlEnums.h.inc -gen-enum-decls)
mlir_tablegen(GgmlEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(GgmlEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS GgmlShapeExpr.td)
mlir_tablegen(GgmlShapeExprInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(GgmlShapeExprInterface.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(GgmlShapeExprIncGen)

set(LLVM_TARGET_DEFINITIONS GgmlShapeRule.td)
mlir_tablegen(GgmlShapeRuleInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(GgmlShapeRuleInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(GgmlShapeRuleIncGen)

set(LLVM_TARGET_DEFINITIONS GgmlAttrs.td)
mlir_tablegen(GgmlAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ggml)
mlir_tablegen(GgmlAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ggml)
add_public_tablegen_target(GgmlAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS GgmlOps.td)
mlir_tablegen(GgmlOps.h.inc -gen-op-decls)
mlir_tablegen(GgmlOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(GgmlOpsIncGen)

# What GGML keeps beside each operation, and everything that walks it. The
# declaration lives in GgmlOps.td beside the arguments it describes.
set(_ggml_tablegen_includes -I "${PROJECT_SOURCE_DIR}/include")
foreach(_include_dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND _ggml_tablegen_includes -I "${_include_dir}")
endforeach()

function(ggml_source_tablegen output action)
  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/${output}")
  add_custom_command(
    OUTPUT "${_generated}"
    COMMAND ggml-tblgen "${action}"
      ${_ggml_tablegen_includes}
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlOps.td"
      -o "${_generated}"
    DEPENDS ggml-tblgen
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlOps.td"
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlSource.td"
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlAttrs.td"
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlDialect.td"
      "${CMAKE_CURRENT_SOURCE_DIR}/GgmlEnums.td"
    COMMENT "Generating ${output}"
    VERBATIM)
  set(GGML_SOURCE_OUTPUTS
    ${GGML_SOURCE_OUTPUTS} "${_generated}" PARENT_SCOPE)
endfunction()

ggml_source_tablegen(GgmlParameters.inc -gen-ggml-parameters)
ggml_source_tablegen(GgmlDecode.inc -gen-ggml-decode)
ggml_source_tablegen(GgmlAttributes.inc -gen-ggml-attributes)
ggml_source_tablegen(GgmlAliasing.inc -gen-ggml-aliasing)
ggml_source_tablegen(GgmlOperationNames.inc -gen-ggml-operation-names)
ggml_source_tablegen(GgmlOperandArguments.inc
  -gen-ggml-operand-arguments)
# The enumerations are declared in their own file, so they are read from it.
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/GgmlEnumNames.inc"
  COMMAND ggml-tblgen -gen-ggml-enum-names
    ${_ggml_tablegen_includes}
    "${CMAKE_CURRENT_SOURCE_DIR}/GgmlEnums.td"
    -o "${CMAKE_CURRENT_BINARY_DIR}/GgmlEnumNames.inc"
  DEPENDS ggml-tblgen "${CMAKE_CURRENT_SOURCE_DIR}/GgmlEnums.td"
  COMMENT "Generating GgmlEnumNames.inc"
  VERBATIM)
list(APPEND GGML_SOURCE_OUTPUTS
  "${CMAKE_CURRENT_BINARY_DIR}/GgmlEnumNames.inc")

add_custom_target(GgmlSourceIncGen
  DEPENDS ${GGML_SOURCE_OUTPUTS})
