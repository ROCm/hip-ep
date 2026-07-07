"""Bazel module extension for ONNX v1.18.0"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _onnx_extension_impl(module_ctx):
    """Implementation for the ONNX extension"""

    # ONNX v1.18.0
    http_archive(
        name = "onnx",
        urls = [
            "https://github.com/onnx/onnx/archive/refs/tags/v1.18.0.tar.gz",
        ],
        sha256 = "b466af96fd8d9f485d1bb14f9bbdd2dfb8421bc5544583f014088fb941a1d21e",
        strip_prefix = "onnx-1.18.0",
        build_file = "//3rd-party:onnx.BUILD",
    )

onnx_extension = module_extension(
    implementation = _onnx_extension_impl,
)
