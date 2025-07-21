def _collect_pb_impl(ctx):
    # Create a list to hold the output files
    outputs = []

    # For each source file, copy it to the output directory
    for src in ctx.files.srcs:
        if not src.path.endswith(ctx.attr.extensions):
           continue
        # Determine the output path based on the output_dir attribute
        output = ctx.actions.declare_file(ctx.attr.output_dir + "/" + src.basename)
        outputs.append(output)

        # Create an action to copy the file
        ctx.actions.run_shell(
            inputs = [src],
            outputs = [output],
            command = "cp %s %s" % (src.path, output.path),
            progress_message = "Copying %s to %s" % (src.path, output.path),
        )

    # Return the outputs so they can be used by other rules
    return [DefaultInfo(
        files = depset(outputs),
        runfiles = ctx.runfiles(files = outputs),
    )]


collect_pb = rule(
    implementation = _collect_pb_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Source files to copy",
        ),
        "output_dir": attr.string(
            mandatory = True,
            doc = "Directory under the build directory where files will be copied",
        ),
        "extensions": attr.string(default = ".pb.h"),
    },
)
