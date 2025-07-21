"""Repository rule to download a file and apply patches using hermetic Python."""

def _patched_http_file_impl(ctx):
    """Implementation of the patched_http_file repository rule."""
    python = ctx.attr._python_interpreter[platform_common.ToolchainInfo].interpreter_path
    ctx.execute([python, "--version"])
    ctx.execute([python, "-m", "pip", "--version"])
    # Download the original file
    ctx.download(
        url = ctx.attr.urls,
        output = ctx.attr.downloaded_file_path,
        sha256 = ctx.attr.sha256 if ctx.attr.sha256 else "",
    )

    # Apply patches if provided
    for patch_file in ctx.attr.patches:
        # Copy patch file to the repository
        patch_path = ctx.path(patch_file)
        if patch_path.exists:
            # Get the Python toolchain
            python_interpreter = ctx.attr.python_bin
            result = ctx.execute([
                str(python_interpreter),
                "-m",
                "patch",
                ctx.attr.downloaded_file_path,
                "-i",
                str(patch_path),
            ])
            if result.return_code != 0:
                fail("Failed to apply patch %s: %s" % (patch_file, result.stderr))

    # Create a BUILD file that exposes the file
    ctx.file("BUILD.bazel", """
filegroup(
    name = "file",
    srcs = ["{}"],
    visibility = ["//visibility:public"],
)
""".format(ctx.attr.downloaded_file_path))

patched_http_file = repository_rule(
    implementation = _patched_http_file_impl,
    attrs = {
        "urls": attr.string_list(
            mandatory = True,
            doc = "List of URLs to download the file from.",
        ),
        "downloaded_file_path": attr.string(
            mandatory = True,
            doc = "Path where the downloaded file should be saved.",
        ),
        "patches": attr.label_list(
            default = [],
            doc = "List of patch files to apply after downloading.",
        ),
        "sha256": attr.string(
            default = "",
            doc = "Expected SHA256 hash of the downloaded file.",
        ),
    },
    doc = "Downloads a file from HTTP and applies patches to it using hermetic Python.",
)
