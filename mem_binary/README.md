<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
## Embedded Resource (Mem_Binary)

The Mem_Binary feature is enabled by defining the CMake variable `MORPHIZEN_EMBEDDED_RESOURCE_PATH`.
If this variable is not defined, the feature is disabled.

## Overview

When enabled, all files specified in `MORPHIZEN_EMBEDDED_RESOURCE_PATH` are packed directly into the library at build time.

## Resource Definition Format

The `embedded_resource.txt` file is parsed using Python’s `ast.literal_eval`.
This allows the use of a JSON-like format that can include comments.

## Compression Support

Each embedded file can optionally be stored in compressed form.

- When compression is set to `True`, the file is stored compressed.
- The file is automatically decompressed when its content is requested using
  `get_mem_binary("file_name.xxx")` or `get_mem_binary_span("file_name.xxx")`.

## Existence Check

To check whether a file exists inside the embedded resources, use
`has_mem_binary("file_name.xxx")` or `get_mem_binary_span("file_name.xxx")->has_value()`.

## Examples

A sample test is provided in the `test` directory.
It includes a sample resource file and example usage of the Mem_Binary APIs.
