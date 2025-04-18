import sys
import os
import re
import datetime
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: vaip_version_info.hpp.inc.py <DIR>")
        sys.exit(1)
    workspace_directory = pathlib.Path(sys.argv[1])
    main(workspace_directory)
