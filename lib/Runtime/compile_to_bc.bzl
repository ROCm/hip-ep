"""compile_to_bc: compile C++ sources to LLVM bitcode.

clang.exe and clang-cl.exe are the same binary (identical SHA256). Driver mode
is selected by argv[0]: clang.exe → GCC mode, clang-cl.exe → CL mode.
Only GCC mode honours -emit-llvm; CL mode silently produces COFF.

WHY all_files (not compiler_files):
  CcToolchainInfo exposes only all_files as a public field; compiler_files is
  private (_compiler_files). all_files includes linker and archiver binaries
  that a compile-only action does not need, but those files are already cached
  in the RBE CAS from other actions in the same build, so the overhead is only
  on the first build of a new toolchain version.

WHY sentinel files for include directories (the _cc_include_dirs attribute):
  The natural API for system include paths is cc_toolchain.built_in_include_
  directories. However, that field stores pre-computed path strings. Under
  Bzlmod, when bazel_cpp_toolchain is a non-root module, those strings are
  wrong: the canonical repo name prefix is doubled. For example:
    wrong:   external/bazel_cpp_toolchain+/toolchain/external/
             bazel_cpp_toolchain++local_tool_deps+msvc/include
    correct: external/bazel_cpp_toolchain++local_tool_deps+msvc/include
  This is a confirmed Bazel design flaw (issues #4605, #15553, #11429);
  the Bazel team closed them as "P2, not planned".

  The workaround: declare one representative (sentinel) file from each include
  directory as a rule attribute (_cc_include_dirs). Calling File.dirname on a
  sentinel at analysis time yields the execution-root-relative path of its
  directory — always correct, regardless of Bzlmod canonical naming, because
  File objects carry the real sandbox path.

  The sentinel files themselves are already members of compiler_files, so no
  extra files are transferred to the RBE content-addressable store.

WHY --sysroot for Linux (and macOS):
  On Linux the C runtime headers (stdio.h, stdlib.h, …) live in @sysroot_linux,
  not in the clang archive. Rather than adding sentinels for every sysroot
  subdirectory, we pass --sysroot to clang and let it locate system headers
  under <sysroot>/usr/include automatically.

  cc_toolchain.sysroot returns the sysroot path as a string. Unlike
  built_in_include_directories, this string is safe: in the Linux toolchain
  config rule it is derived from File.path of actual @sysroot_linux File
  objects (via get_common_root_dir()), so it is immune to the Bzlmod
  path-doubling bug.

  cc_toolchain.sysroot is None on Windows (no sysroot used), so the --sysroot
  branch is a no-op there.

System headers reach the sandbox via cc_toolchain.compiler_files, which the
toolchain config populates per platform:
  Windows: @msvc/include (C++ stdlib + CRT), @windows_sdk/Include/{ucrt,
           shared,um,winrt}, @clang_windows/lib/clang/<ver>/include (built-ins)
  Linux:   @clang_linux/{include/c++/v1, lib/clang/<ver>/include}, @sysroot_linux
  macOS:   @macos_sdk (macOS SDK), @clang_darwin/{include/c++/v1,
           lib/clang/<ver>/include}
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

# map_each callback for args.add_all: returns the directory of a File as an
# execution-root-relative string. Defined at module level (not inside _impl)
# because Starlark requires map_each functions to be module-level references,
# not closures, so that Bazel can serialise the action for remote execution.
def _dirname(f):
    return f.dirname

def _compile_to_bc_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # Collect include paths and headers from all CcInfo deps transitively.
    merged = cc_common.merge_compilation_contexts(
        compilation_contexts = [
            dep[CcInfo].compilation_context
            for dep in ctx.attr.deps
            if CcInfo in dep
        ],
    )

    outputs = []
    for src in ctx.files.srcs:
        bc_file = ctx.actions.declare_file(src.basename[:-len(".cpp")] + ".bc")

        args = ctx.actions.args()
        args.add("-c")
        args.add("-emit-llvm")
        args.add("-o", bc_file)
        args.add(src)

        # Toolchain system include directories, derived from sentinel files.
        # uniquify=True deduplicates in case future changes introduce overlapping
        # sentinels. map_each=_dirname uses the module-level function so Bazel's
        # path-mapping infrastructure can intercept and rewrite paths if needed.
        args.add_all(
            ctx.files._cc_include_dirs,
            map_each = _dirname,
            before_each = "-isystem",
            uniquify = True,
        )

        # Linux / macOS sysroot: pass --sysroot so clang finds C runtime headers
        # in <sysroot>/usr/include. On Windows cc_toolchain.sysroot is None
        # (no sysroot configured), so this branch is skipped automatically.
        if cc_toolchain.sysroot:
            args.add("--sysroot", cc_toolchain.sysroot)

        # System includes from deps (e.g. lib/Runtime, lib/Runtime/mock,
        # schemas, flatbuffers) — cc_library.includes flows here as -isystem.
        args.add_all(merged.system_includes, before_each = "-isystem")

        # Regular -I includes from deps.
        args.add_all(merged.includes, before_each = "-I")

        # Quote includes: src.dirname so #include "sibling.h" resolves,
        # replicating what cc_library does automatically for its own srcs.
        args.add("-iquote", src.dirname)
        args.add_all(merged.quote_includes, before_each = "-iquote")

        # User-supplied flags (defines, optimisation, std version, …).
        args.add_all(ctx.attr.copts)

        inputs = depset(
            direct = [src] + ctx.files.hdrs,
            transitive = [
                merged.headers,               # all dep headers transitively
                cc_toolchain.all_files,        # clang binary + system headers
                                              # (compiler_files is private on
                                              # CcToolchainInfo; all_files is
                                              # the only public equivalent)
            ],
        )

        ctx.actions.run(
            executable = ctx.executable._clang,
            arguments = [args],
            inputs = inputs,
            outputs = [bc_file],
            mnemonic = "CompileToBc",
            progress_message = "Compiling %s to LLVM bitcode" % src.short_path,
        )
        outputs.append(bc_file)

    return [DefaultInfo(files = depset(outputs))]

compile_to_bc = rule(
    implementation = _compile_to_bc_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cpp"]),
        "hdrs": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = [CcInfo]),
        "copts": attr.string_list(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
        # One representative (sentinel) file per toolchain include directory.
        # File.dirname on each gives the correct -isystem path; see module
        # docstring for why this is necessary instead of using
        # cc_toolchain.built_in_include_directories.
        # cfg="exec" ensures the select() inside cc_include_dirs resolves for
        # the execution platform (where clang runs), not the target platform.
        "_cc_include_dirs": attr.label(
            default = "@bazel_cpp_toolchain//thirdparty/clang:cc_include_dirs",
            allow_files = True,
            cfg = "exec",
        ),
        # clang.exe = GCC driver mode (same binary as clang-cl.exe; driver mode
        # is selected by argv[0]). The alias in bazel_cpp_toolchain selects the
        # right binary per execution platform via @platforms//os:*.
        "_clang": attr.label(
            default = "@bazel_cpp_toolchain//thirdparty/clang:clang",
            executable = True,
            cfg = "exec",
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
