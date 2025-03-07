# -*- coding: utf-8 -*-
##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##

import glob
import os
import platform
from pathlib import Path
from datetime import datetime
import shutil
import tempfile
import setuptools.command.build_ext
from setuptools import Extension, find_packages, setup
from subprocess import check_output

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
is_static_python = (
    len(glob.glob(os.path.join(TOP_DIR, "voe", "voe_cpp2py_export*.*"))) == 0
)
HOME = Path.home()

with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

ext_modules = [Extension(name="voe.voe_cpp2py_export", sources=[])]
data_files = []


class build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        src = glob.glob(os.path.join(TOP_DIR, "voe", "voe_cpp2py_export*.*"))[0]
        filename = "voe_cpp2py_export" + src.split("voe_cpp2py_export")[-1]
        dst = os.path.join(os.path.realpath(self.build_lib), "voe", filename)
        self.copy_file(src, dst)


def copy_folder(source, destination):
    try:
        # If destination does not exist, it will be created
        destination_path = os.path.join(destination, os.path.basename(source))
        shutil.copytree(source, destination_path)
        print(f"Folder copied from {source} to {destination_path}")
    except Exception as e:
        print(f"Error: {e}")


def recursive_data_files(source_folder):
    """Recursively collect all files in the source folder."""
    data_files = []
    for directory, _, files in os.walk(source_folder):
        file_paths = [os.path.join(directory, file) for file in files]
        if file_paths:
            # Generate path as expected in 'data_files' spec,
            # which is (destination_directory, [files...])
            data_files.append((directory, file_paths))
    return data_files


def copy_so_files(build_dir, destination_dir):
    # Find all .so files under build_dir
    so_files = glob.glob(os.path.join(build_dir, "**", "*.so"), recursive=True)

    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy .so files to destination directory
    for so_file in so_files:
        shutil.copy(so_file, destination_dir)


def parse_shortrev() -> str:
    if "VAIP_WORKSPACE_PATH" in os.environ:
        vaip_repo_path = str(Path(os.environ["VAIP_WORKSPACE_PATH"]) / "vaip")

        return check_output(
            ["git", "-C", vaip_repo_path, "rev-parse", "--short", "HEAD"],
            encoding="UTF-8",
            universal_newlines=True,
        ).strip()
    else:
        return "hashnone"


def vaiml_custom_ops_data_files(package_path):
    data_files = []
    custom_ops_build_path = os.path.join(
        "..", "..", "vaip_pass_vaiml_custom_op", "vaiml-custom-ops"
    )
    for d in os.listdir(custom_ops_build_path):
        op_build_path = os.path.join(custom_ops_build_path, d)
        if os.path.isdir(op_build_path):
            op_resources_path = os.path.join(custom_ops_build_path, d, "resources")
            for directory, _, files in os.walk(op_resources_path):
                file_paths = [os.path.join(directory, file) for file in files]
                if directory == str(op_resources_path):
                    # add so/dll from build directory of the oparator to the op root directory
                    file_paths.extend(
                        glob.glob(
                            f"{op_build_path}/lib{d}.dll"
                            if platform.system() == "Windows"
                            else f"{op_build_path}/lib{d}.so"
                        )
                    )
                    data_files.append((os.path.join(package_path, d), file_paths))
                else:
                    # retain the internal folder structure in package dir
                    data_files.append(
                        (
                            os.path.join(
                                package_path,
                                d,
                                os.path.relpath(directory, op_resources_path),
                            ),
                            file_paths,
                        )
                    )
    return data_files


voe_version = "1.4.0"
if "NIGHTLY_BUILD" in os.environ:
    voe_version += f".dev{datetime.today().strftime('%Y%m%d%H%M%S')}"
    if "VAIP_WORKSPACE_PATH" in os.environ:
        voe_version += f"+g{parse_shortrev()}"

if platform.system() == "Windows":
    vitisai = glob.glob("../*/onnxruntime_vitisai_ep.dll")[0]
    capi = [vitisai]
    data_files.append(("lib/site-packages/onnxruntime/capi", capi))
    # data_files.append(("lib/site-packages/onnxruntime/providers/tvm", [*gemm_asr]))
    voe_package_path = os.path.join("lib", "site-packages", "voe")
    data_files_custom_ops = vaiml_custom_ops_data_files(
        os.path.join(voe_package_path, "vaiml-custom-ops")
    )
    data_files.extend(data_files_custom_ops)

else:
    if "CMAKE_INSTALL_PREFIX" in os.environ:
        # copy vaip .so files from build dir
        build_dir = os.path.abspath("../..")
        tmp_dir = os.path.join(
            tempfile.gettempdir(), next(tempfile._get_candidate_names())
        )
        copy_so_files(build_dir, tmp_dir)
        src = Path(os.environ.get("CMAKE_INSTALL_PREFIX"))
        # Specify the source folder path
        source_folder = os.path.join(src, "bin")

        # Get the current working directory as the destination
        destination_folder = os.getcwd()

        python_version = "python" + ".".join(platform.python_version().split(".")[:2])
        package_path = os.path.join("lib", python_version, "site-packages", "voe")
        destination_folder = os.path.join(destination_folder, package_path)

        # Call the function to copy the folder
        copy_folder(source_folder, destination_folder)
        source_folder = os.path.join(src, "lib")
        copy_folder(source_folder, destination_folder)
        source_folder = os.path.join(src, "lib64")
        copy_folder(source_folder, destination_folder)
        # copy vaip related .so to lib
        [
            shutil.copy(
                os.path.join(tmp_dir, f), os.path.join(destination_folder, "lib")
            )
            for f in os.listdir(tmp_dir)
        ]

        data_files = recursive_data_files(os.path.join(package_path, "bin"))
        data_files_lib = recursive_data_files(os.path.join(package_path, "lib"))
        data_files_lib64 = recursive_data_files(os.path.join(package_path, "lib64"))
        data_files_custom_ops = vaiml_custom_ops_data_files(
            os.path.join(package_path, "vaiml-custom-ops")
        )
        data_files.extend(data_files_lib)
        data_files.extend(data_files_lib64)
        data_files.extend(data_files_custom_ops)

package_data = {"voe.passes.microkernel": ["*.mlir", "*.json"]}


class build_ext(setuptools.command.build_ext.build_ext):
    def build_extensions(self):
        if is_static_python:
            return
        src = glob.glob(os.path.join(TOP_DIR, "voe", "voe_cpp2py_export*.*"))[0]
        filename = "voe_cpp2py_export" + src.split("voe_cpp2py_export")[-1]
        dst = os.path.join(os.path.realpath(self.build_lib), "voe", filename)
        self.copy_file(src, dst)


setup(
    name="voe",
    version=voe_version,
    description="some common util for vaip dev",
    long_description=readme,
    author="Wang Chunye",
    install_requires=["glog==0.3.1"]
    if not is_static_python
    else [],  # change vai-rt static_cpython's install_pkg as well
    extras_require={
        "tools": ["graphviz", "pandas", "xlsxwriter", "onnx==1.16.1", "tabulate"]
        if not is_static_python
        else [],
        "vaiq": ["vai-q-onnx", "olive-ai[cpu]"] if not is_static_python else [],
    },
    entry_points={"console_scripts": ["voe_vis_pass = voe.vis_pass:main"]},
    license=license,
    packages=find_packages(where=".", exclude=("tests", "docs")),
    package_data=package_data,
    data_files=data_files,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
