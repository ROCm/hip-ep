#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

import zipfile
import subprocess
import sys
import os
from pathlib import Path


def is_whilte_list_file(filename):
    file_type = [
        "py",
        "pyc",
    ]
    for t in file_type:
        if filename.endswith(f".{t}"):
            return True
    return False


def copy_to_zip(zip, src, dst):
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            if "__pycache__" in str(root):  # no cache needed
                continue
            if "tests" in str(root):
                continue
            for file in files:
                if is_whilte_list_file(file) == False:
                    continue
                new_src = os.path.join(root, file)
                new_dst = dst / os.path.relpath(os.path.join(root, file), src)
                zip.write(new_src, new_dst)
    else:
        zip.write(src, dst)


def is_black_list_module(module):
    if module.endswith(".dist-info"):
        return True
    blocked_pkg = {
        "build",
        "pip",
        "wheel",
        "packaging",
        "setuptools",
        "pkg_resources",
        "pydoc_data",
        "email",
        "test",
        "tkinter",
        "_distutils_hack",
        "ensurepip",
        "distutils",
        "unittest",
        "lib2to3",
        "pyproject_hooks",
        # the following are statically linked to python
        "importlib",
        "io.py",
        "os.py",
        "abc.py",
        "stat.py",
        "genericpath.py",
        "posixpath.py",
        "shutil.py",
        "fnmatch.py",
        "collections",
        "bz2.py",
        "lzma.py",
        "struct.py",
        "threading.py",
        "functools.py",
        "contextlib.py",
        "codecs.py",
        "encodings",
        "types.py",
        "operator.py",
        "re.py",
        "warnings.py",
        "zipfile.py",
        "typing.py",
        "_collections_abc.py",
        "heapq.py",
        "keyword.py",
        "reprlib.py",
        "enum.py",
        "sre_compile.py",
        "sre_parse.py",
        "sre_constants.py",
        "copyreg.py",
        "ntpath.py",
        "signal.py",
        "_weakrefset.py",
        "traceback.py",
        "zipimport.py",
    }
    if module in blocked_pkg:
        return True
    return False


def copy_site_pkg(zip, path):
    module_list = os.listdir(path)
    for module in module_list:
        if is_black_list_module(module):
            continue
        copy_to_zip(zip, os.path.join(path, module), Path("/") / module)


def write_to_zip(python_path, voe_path, onnx_tool_path, filename):
    compression = zipfile.ZIP_DEFLATED
    compresslevel = 9

    with zipfile.ZipFile(
        filename, "w", compression=compression, compresslevel=compresslevel
    ) as zip:
        file_or_folder = os.listdir(python_path)
        for f in file_or_folder:
            if f != "site-packages":
                if is_black_list_module(f) == False:
                    copy_to_zip(zip, os.path.join(python_path, f), Path("/") / f)
            else:
                copy_site_pkg(zip, os.path.join(python_path, f))
        copy_to_zip(zip, voe_path, Path("/") / "voe")
        copy_to_zip(zip, onnx_tool_path, Path("/") / "onnx_tool")


if __name__ == "__main__":
    filename = sys.argv[1]
    python_path = sys.argv[2]
    voe_path = sys.argv[3]
    onnx_tool_path = sys.argv[4]
    write_to_zip(python_path, voe_path, onnx_tool_path, filename)
