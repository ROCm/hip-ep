#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import patch
import shutil
import pathlib

# This script applies a patch to a file specified in the command line arguments.
# Firstly it copies input files to a destination directory,
# then it applies the patch using the `patch` module.
# Usage:
# python patch.py <patch_file> [<source_file> <destination_directory>] ...  -- <options for patch>
# where <patch_file> is the patch to apply, <source_file> is the file to patch,
# and <destination_directory> is where the patched file will be saved.
#
# Example:
# python patch.py my_patch.patch dir1/file1.txt dest/dir1/file1.txt dir2/file2.txt dest/dir2/file1.txt  -- -p1
#
# NOTE: create directorys in the destination path if they do not exist.
#


def main1():
    if len(sys.argv) < 2:
        print(
            "Usage: python patch.py <patch_file> [<source_file> <destination_directory>] ... -- <options for patch>"
        )
        sys.exit(1)
    patch_file = sys.argv[1]
    print(f"input argv: {sys.argv}")
    for i in range(2, len(sys.argv) - 1, 2):
        if sys.argv[i] == "--":
            break
        print(
            f"Processing source file: {sys.argv[i]} and destination directory: {sys.argv[i + 1]}"
        )
        source_file = pathlib.Path(sys.argv[i])
        destination_directory = pathlib.Path(sys.argv[i + 1])
        print(f"Copying {source_file} to {destination_directory}")

        parent_directory = (
            destination_directory.parent
            if destination_directory.is_file()
            else destination_directory
        )
        # Ensure the destination directory exists
        print(f"Ensuring directory exists: {parent_directory}")
        parent_directory.mkdir(parents=True, exist_ok=True)

        # Copy the source file to the destination directory
        # print(f"Copying {source_file} to {destination_directory}")

        shutil.copy(source_file, destination_directory)

    patch_args = sys.argv[i + 2 :]  # Remaining arguments after '--'
    sys.argv = [
        sys.argv[0],
        patch_file,
    ] + patch_args  # Prepare arguments for patch module
    print(f"Applying patch: {patch_file} with arguments: {patch_args}")
    patch.main()


if __name__ == "__main__":
    print(f"patch module is imported {patch.__file__}")
    main1()
