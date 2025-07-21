# 3rd-party dependencies

package(default_visibility = ["//visibility:public"])

# Export BUILD files for external dependencies
exports_files([
    "onnxruntime.BUILD",
])

cc_library(
   name = "hash-library",
   hdrs = glob(["hash-library/*.h"]),
   srcs = glob(["hash-library/*.cpp"]),
   strip_include_prefix = ".",
   visibility = ["//visibility:public"],
)
