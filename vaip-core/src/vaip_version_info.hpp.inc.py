import sys
import os
import re
from datetime import datetime
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
            VALUE "FileDescription", "{PRODUCT_DESCRIPTION} @{GIT_COMMIT}"
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
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        raise ValueError("Error while getting morphizen git hash")

    try:
        git_branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=path, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        git_branch = git_hash[0:6]
    return git_branch, git_hash


def get_morphizen_version_info():
    file_path = os.path.abspath(__file__)
    cur_file_basedir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    return get_dir_version_info(cur_file_basedir)


def main2(workspace_directory):
    output_file = "vaip_version_info.hpp.inc"
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
    project_directory = pathlib.Path(__file__).parent.parent.parent
    # Run a command and capture its output
    branch_name = os.environ.get("GIT_BRANCH", "N/A")
    pattern_ddmm = re.compile(r".*_(\d{2})(\d{2})_rc.*")
    pattern_ddmmyyyy = re.compile(r".*_(\d{2})(\d{2})(\d{4})_rc.*")
    if match := pattern_ddmmyyyy.match(branch_name):
        month, day, year = match.groups()
    elif match := pattern_ddmm.match(branch_name):
        month, day = match.groups()
        year = datetime.now().year  # Year is not present in MMDD format
    else:
        year = datetime.now().year  # Year is not present in MMDD format
        month = datetime.now().month
        day = datetime.now().day
    return {
        "GIT_COMMIT": "N/A",
        "BUILD_NUMBER": "0",
        "MORPHIZEN_FILE_MAJOR": year,
        "MORPHIZEN_FILE_MINOR": month,
        "MORPHIZEN_FILE_PATCH": day,
        "MORPHIZEN_PRODUCT_MAJOR": year,
        "MORPHIZEN_PRODUCT_MINOR": month,
        "MORPHIZEN_PRODUCT_PATCH": day,
        "PRODUCT_DESCRIPTION": "ONNXRuntime VitisAI EP",
        "PRODUCT_NAME": "MorphiZen",
        "morphizen_OUTPUT_NAME": "onnxruntime_vitisai_ep",
        **os.environ,
    }


def write_version_rc():
    with open("version.rc", "w") as f:
        f.write(VERSION_RC.format(**get_version_info_for_rc()))


def main(release_file):
    output_file = "vaip_version_info.hpp.inc"
    with open(output_file, "w") as f_out:
        project_commit_id = os.environ.get("PROJECT_GIT_COMMIT_ID", "N/A")
        project_version_id = os.environ.get("PROJECT_VERSION_ID", "v1.0")
        if project_commit_id != "N/A":
            project_out = f"""
                {{"vai-rt", "{project_commit_id}", "{project_version_id}"}},
            """
            f_out.write(project_out)
        ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR = os.environ.get(
            "ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR", "N/A"
        )
        if ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR != "N/A":
            ort_branch, ort_git_hash = get_dir_version_info(
                ORT_CORE_PROVIDERS_VITISAI_INCLUDE_DIR
            )
            project_out = f"""
                {{"onnxruntime", "{ort_git_hash}", "{ort_branch}"}},
            """
            f_out.write(project_out)
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
            morphizen_branch, morphizen_git_hash = get_morphizen_version_info()
            morphizen_output = f"""
                {{"morphizen", "{morphizen_git_hash}", "{morphizen_branch}"}},
            """
            f_out.write(morphizen_output)
    write_version_rc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: vaip_version_info.hpp.inc.py <DIR>")
        sys.exit(1)
    workspace_directory = pathlib.Path(sys.argv[1])
    main(workspace_directory)
