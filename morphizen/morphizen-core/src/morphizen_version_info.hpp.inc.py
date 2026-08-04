#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import os
import pathlib
import subprocess

PROJECTS = [
    "glog",
    "gsl",
    "json",
    "protobuf",
    "onnxruntime",
    "MorphiZen",
    "xcompiler",
    "xir",
    "target_factory",
    "vart",
    "dod",
]
VERSION_RC = r"""
// Microsoft Visual C++ generated resource script.
//
#pragma code_page(65001)

// #include "resource.h"

#define APSTUDIO_READONLY_SYMBOLS
/////////////////////////////////////////////////////////////////////////////
//
// Generated from the TEXTINCLUDE 2 resource.
//
#include "winres.h"

/////////////////////////////////////////////////////////////////////////////
#undef APSTUDIO_READONLY_SYMBOLS

/////////////////////////////////////////////////////////////////////////////
// English (United States) resources

#if !defined(AFX_RESOURCE_DLL) || defined(AFX_TARG_ENU)
LANGUAGE LANG_ENGLISH, SUBLANG_ENGLISH_US

#ifdef APSTUDIO_INVOKED
/////////////////////////////////////////////////////////////////////////////
//
// TEXTINCLUDE
//

1 TEXTINCLUDE
BEGIN
    "resource.h\0"
END

2 TEXTINCLUDE
BEGIN
    "#include ""winres.h""\r\n"
    "\0"
END

3 TEXTINCLUDE
BEGIN
    "\r\n"
    "\0"
END

#endif    // APSTUDIO_INVOKED


/////////////////////////////////////////////////////////////////////////////
//
// Version
//

VS_VERSION_INFO VERSIONINFO
 FILEVERSION {MORPHIZEN_FILE_MAJOR}, {MORPHIZEN_FILE_MINOR}, {MORPHIZEN_FILE_PATCH}, {BUILD_NUMBER}
 PRODUCTVERSION {MORPHIZEN_PRODUCT_MAJOR}, {MORPHIZEN_PRODUCT_MINOR}, {MORPHIZEN_PRODUCT_PATCH}, {BUILD_NUMBER}
#ifdef _DEBUG
 FILEFLAGS 0x1L
#else
 FILEFLAGS 0x0L
#endif
 FILEOS 0x40004L
 FILETYPE 0x2L
 FILESUBTYPE 0x0L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904b0"
        BEGIN
            VALUE "FileDescription", "{PRODUCT_DESCRIPTION} b{BUILD_NUMBER}-g{GIT_COMMIT}@{GIT_BRANCH}"
            VALUE "FileVersion", "{MORPHIZEN_FILE_MAJOR}.{MORPHIZEN_FILE_MINOR}.{MORPHIZEN_FILE_PATCH}.{BUILD_NUMBER}"
            VALUE "ProductVersion", "{MORPHIZEN_PRODUCT_MAJOR}.{MORPHIZEN_PRODUCT_MINOR}.{MORPHIZEN_PRODUCT_PATCH}.{BUILD_NUMBER}"
            VALUE "CompanyName", "AMD"
            VALUE "ProductName", "{PRODUCT_NAME}"
            VALUE "OriginalFilename", "{morphizen_OUTPUT_NAME}"
            VALUE "InternalName", "{morphizen_OUTPUT_NAME}"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END


#endif    // English (United States) resources
/////////////////////////////////////////////////////////////////////////////



#ifndef APSTUDIO_INVOKED
/////////////////////////////////////////////////////////////////////////////
//
// Generated from the TEXTINCLUDE 3 resource.
//


/////////////////////////////////////////////////////////////////////////////
#endif    // not APSTUDIO_INVOKED

"""


def get_dir_version_info(path):
    try:
        print(f"CWD={path}")
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except subprocess.CalledProcessError:
        print(f"Error while getting morphizen git hash {path}")
        return "N/A", "N/A"

    try:
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=path, text=True
        ).strip()
    except subprocess.CalledProcessError:
        print(f"Error while getting morphizen git branch {path}")
        return "N/A", "N/A"
    git_branch = git_hash[0:6]
    return git_branch, git_hash


def get_morphizen_version_info():
    file_path = os.path.abspath(__file__)
    cur_file_basedir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    return get_dir_version_info(cur_file_basedir)


def main2(workspace_directory):
    output_file = "morphizen_version_info.hpp.inc"
    with open(output_file, "w") as f:
        for project in PROJECTS:
            if not os.path.exists(workspace_directory / project):
                print(f"Project {project} not found in {workspace_directory}")
                continue
            project_directory = workspace_directory / project
            print(f"Processing {project_directory}")
            git_commit_id = (
                os.popen(f"cd {project_directory} && git rev-parse HEAD").read().strip()
            )
            project_name = project
            project_version_id = (
                os.popen(
                    f"cd {project_directory} &&git describe --tags --abbrev=1 HEAD"
                )
                .read()
                .strip()
            )
            output = f"""
                {{"{project_name}", "{git_commit_id}", "{project_version_id}"}},
            """
            f.write(output)


def get_version_info_for_rc():
    # Run a command and capture its output
    return {
        "GIT_COMMIT": "N/A",
        "GIT_BRANCH": "N/A",
        "BUILD_NUMBER": "0",
        "MORPHIZEN_FILE_MAJOR": 1,
        "MORPHIZEN_FILE_MINOR": 0,
        "MORPHIZEN_FILE_PATCH": 0,
        "MORPHIZEN_PRODUCT_MAJOR": 1,
        "MORPHIZEN_PRODUCT_MINOR": 0,
        "MORPHIZEN_PRODUCT_PATCH": 0,
        "PRODUCT_DESCRIPTION": "ONNXRuntime MorphiZen EP",
        "PRODUCT_NAME": "MorphiZen",
        "morphizen_OUTPUT_NAME": "onnxruntime_morphizen_ep",
        **os.environ,
    }


def write_version_rc():
    with open("version.rc", "w") as f:
        f.write(VERSION_RC.format(**get_version_info_for_rc()))


def main(release_file):
    output_file = "morphizen_version_info.hpp.inc"
    with open(output_file, "w") as f_out:
        with open(release_file, "r") as f_in:
            for line in f_in.readlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                project_name, git_commit_id, project_version_id = line.split(";")
                output = f"""
                {{"{project_name}", "{git_commit_id}", "{project_version_id}"}},
                """
                f_out.write(output)
    write_version_rc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: morphizen_version_info.hpp.inc.py <DIR>")
        sys.exit(1)
    workspace_directory = pathlib.Path(sys.argv[1])
    main(workspace_directory)
