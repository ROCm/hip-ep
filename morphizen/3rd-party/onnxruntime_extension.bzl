# ONNXRuntime module extension for Bazel bzlmod

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _onnxruntime_extension_impl(module_ctx):
    """Module extension to add ONNXRuntime headers dependency."""

    # ONNXRuntime from specific commit: ae7c54113ed103ca279b638bdb07f49b51611f5e
    http_archive(
        name = "onnxruntime",
        build_file = Label("//3rd-party:onnxruntime.BUILD"),
        sha256 = "bc60ad9ce3e8fb6b44ef8c3e9191c25b3471d583d94bf5694fe2268288929baf",
        strip_prefix = "onnxruntime-ae7c54113ed103ca279b638bdb07f49b51611f5e",
        urls = [
            "https://github.com/microsoft/onnxruntime/archive/ae7c54113ed103ca279b638bdb07f49b51611f5e.tar.gz",
        ],
    )

onnxruntime_extension = module_extension(
    implementation = _onnxruntime_extension_impl,
)
