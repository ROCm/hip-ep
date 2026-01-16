##
# ** Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../patterns/conv.json" JSON_CONTENT)
string(REPLACE "\"" "\\\"" JSON_CONTENT_ESCAPED ${JSON_CONTENT})
string(REPLACE "\n" "\\n" JSON_CONTENT_ESCAPED ${JSON_CONTENT_ESCAPED})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/conv_json.inc"
  "static const unsigned char conv_json[] = \"${JSON_CONTENT_ESCAPED}\";\n"
  "static const unsigned int conv_json_len = sizeof(conv_json) - 1;\n"
)
