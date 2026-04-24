# embed_spirv.cmake — Convert a .spv binary to a C header with byte array
#
# Input variables (set via -D on command line):
#   SPV_FILE  — path to the compiled .spv file
#   HDR_FILE  — path to the output .h file
#   VAR_NAME  — C variable name prefix (e.g. "matmul_nbits")
#
# Output: a header defining:
#   static const uint32_t <VAR_NAME>_spv[] = { 0x..., ... };
#   static const size_t   <VAR_NAME>_spv_size = sizeof(<VAR_NAME>_spv);

file(READ "${SPV_FILE}" spv_data HEX)
string(LENGTH "${spv_data}" spv_hex_len)

# Convert hex string to comma-separated 32-bit words (SPIR-V is uint32 aligned)
set(words "")
math(EXPR num_bytes "${spv_hex_len} / 2")
math(EXPR num_words "${num_bytes} / 4")

set(i 0)
while(i LESS spv_hex_len)
    # Read 8 hex chars (4 bytes = 1 uint32, little-endian in file)
    string(SUBSTRING "${spv_data}" ${i} 8 word_hex)
    # Reverse byte order: SPIR-V file is little-endian, hex string is big-endian bytes
    string(SUBSTRING "${word_hex}" 6 2 b0)
    string(SUBSTRING "${word_hex}" 4 2 b1)
    string(SUBSTRING "${word_hex}" 2 2 b2)
    string(SUBSTRING "${word_hex}" 0 2 b3)
    if(words)
        string(APPEND words ",\n    ")
    endif()
    string(APPEND words "0x${b0}${b1}${b2}${b3}")
    math(EXPR i "${i} + 8")
endwhile()

# Write the header
file(WRITE "${HDR_FILE}"
"/* Auto-generated from ${VAR_NAME}.spv — do not edit */\n"
"#ifndef ${VAR_NAME}_SPV_H\n"
"#define ${VAR_NAME}_SPV_H\n\n"
"#include <stdint.h>\n"
"#include <stddef.h>\n\n"
"static const uint32_t ${VAR_NAME}_spv[] = {\n"
"    ${words}\n"
"};\n\n"
"static const size_t ${VAR_NAME}_spv_size = sizeof(${VAR_NAME}_spv);\n\n"
"#endif /* ${VAR_NAME}_SPV_H */\n"
)
