#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Create a placeholder weight file of an exact byte size.

A test naming an external byte range needs a file the conversion can memory
map, but not the weights themselves: the compiler records the path and offset
and never reads the bytes. So the file is only given a length, and its contents
are left undefined. Systems that support sparse files back it with no blocks at
all; the rest allocate without writing, which is still near-instant.

Creating the directory and skipping an already-correct file keeps the caller to
one command, with no shell utilities to be portable about.

Usage: make_external_data.py <path> <size-in-bytes>
"""

import os
import sys


def main(argv):
    if len(argv) != 3:
        sys.exit(f"usage: {argv[0]} <path> <size-in-bytes>")
    path, size = argv[1], int(argv[2])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) == size:
        return

    with open(path, "wb") as f:
        f.truncate(size)


if __name__ == "__main__":
    main(sys.argv)
