# ONNXRuntime external dependency BUILD file
# Simple headers-only approach to avoid toolchain complexities

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Header-only ONNXRuntime library for now
# This provides access to ONNXRuntime headers without building the full library
cc_library(
    name = "onnxruntime_headers",
    hdrs = glob([
        "include/onnxruntime/**/*.h",
        "include/onnxruntime/core/**/*.h",
    ], allow_empty = True),
    includes = ["include", "include/onnxruntime", "onnxruntime/core/providers/vitisai/include"],
    deps = [
        "@gsl",
    ],
    visibility = ["//visibility:public"],
    defines = [
        "ORT_API_MANUAL_INIT",
    ],
)

# Alias for easier referencing
cc_library(
    name = "onnxruntime",
    deps = [":onnxruntime_headers"],
    visibility = ["//visibility:public"],
)

# Optional: filegroup of all sources for future CMake builds
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# TODO: Full ONNXRuntime library build using rules_foreign_cc
# This requires resolving the toolchain dependency issues first
# cmake(
#     name = "onnxruntime_full",
#     cache_entries = {...},
#     lib_source = ":all_srcs",
#     ...
# )
