##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/gemm.json" JSON_CONTENT)
string(REPLACE "\"" "\\\"" JSON_CONTENT_ESCAPED ${JSON_CONTENT})
string(REPLACE "\n" "\\n" JSON_CONTENT_ESCAPED ${JSON_CONTENT_ESCAPED})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gemm_json.inc"
  "static const unsigned char gemm_json[] = \"${JSON_CONTENT_ESCAPED}\";\n"
  "static const unsigned int gemm_json_len = sizeof(gemm_json) - 1;\n"
)
