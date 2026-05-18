"""BUILD file for ONNX v1.18.0"""

load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "gen_proto",
    srcs = ["onnx/gen_proto.py"],
    data = [
        "onnx/onnx.in.proto",
        "onnx/onnx-data.in.proto",
        "onnx/onnx-operators.in.proto",
    ],
)

genrule(
    name = "generate_onnx_proto",
    outs = [
        "onnx/onnx_morphizen_onnx-ml.proto",
        "onnx/onnx_morphizen_onnx-ml.proto3",
        "onnx/onnx_pb.py",
        "onnx/onnx-ml.pb.h",
    ],
    cmd = "$(location :gen_proto) -p morphizen_onnx -o $(@D)/onnx onnx -m",
    tools = [":gen_proto"],
)

genrule(
    name = "generate_onnx_operators_proto",
    outs = [
        "onnx/onnx-operators_morphizen_onnx-ml.proto",
        "onnx/onnx-operators_morphizen_onnx-ml.proto3",
        "onnx/onnx_operators_pb.py",
        "onnx/onnx-operators-ml.pb.h",
    ],
    cmd = "$(location :gen_proto) -p morphizen_onnx -o $(@D)/onnx onnx-operators -m >/dev/null",
    tools = [":gen_proto"],
)

genrule(
    name = "generate_onnx_data_proto",
    outs = [
        "onnx/onnx-data_morphizen_onnx.proto",
        "onnx/onnx-data_morphizen_onnx.proto3",
        "onnx/onnx_data_pb.py",
        "onnx/onnx-data.pb.h",
    ],
    cmd = "$(location :gen_proto) -p morphizen_onnx -o $(@D)/onnx onnx-data -m >/dev/null",
    tools = [":gen_proto"],
)

proto_library(
    name = "onnx_proto",
    srcs = [
        "onnx/onnx-data_morphizen_onnx.proto",
        "onnx/onnx-operators_morphizen_onnx-ml.proto",
        "onnx/onnx_morphizen_onnx-ml.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "onnx_proto_lib",
    visibility = ["//visibility:public"],
    deps = [":onnx_proto"],
)

cc_library(
    name = "onnx_proto_headers",
    hdrs = [
        "onnx/onnx-data.pb.h",
        "onnx/onnx-ml.pb.h",
        "onnx/onnx-operators-ml.pb.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":onnx_proto_lib",
    ],
)

cc_library(
    name = "onnx",
    srcs = glob(
        [
            "onnx/*.cc",
            "onnx/common/*.cc",
            "onnx/defs/**/*.cc",
            "onnx/shape_inference/*.cc",
            "onnx/version_converter/*.cc",
        ],
        exclude = [
            "onnx/cpp2py_export.cc",
        ],
    ),
    hdrs = glob([
        "onnx/*.h",
        "onnx/version_converter/*.h",
        "onnx/common/*.h",
        "onnx/defs/**/*.h",
        "onnx/shape_inference/*.h",
        "onnx/version_converter/adapters/*.h",
    ]),
    copts = select({
        "@platforms//os:windows": [
            "/std:c++17",
            "/W0",  # Disable all warnings
            "/wd4244",  # Disable conversion warnings specifically
            "/wd4267",  # Disable size_t conversion warnings
            "/wd4305",  # Disable truncation warnings
            "/wd4996",  # Disable deprecation warnings
        ],
        "@platforms//os:macos": [
            "-UDEBUG",
            "--std=c++17",
            "-w",  # Disable all warnings
        ],
        "//conditions:default": [
            "--std=c++17",
            "-w",  # Disable all warnings (equivalent to /W0 on Windows)
        ],
    }),
    defines = [
        "ONNX_ML=1",
    ],
    # ONNX_NAMESPACE=morphizen_onnx is intentionally PRIVATE (local_defines) so it
    # does not bleed into every target that transitively depends on @onnx.  CMake
    # mirrors this: morphizen-core-static.cmake links onnx PRIVATE to prevent
    # namespace pollution downstream.  Each consumer that directly includes <onnx/…>
    # headers must declare its own local_defines = ["ONNX_NAMESPACE=morphizen_onnx"].
    local_defines = [
        "ONNX_NAMESPACE=morphizen_onnx",
    ],
    includes = [
        ".",
        "onnx",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":onnx_proto_headers",
        ":onnx_proto_lib",
    ],
)
