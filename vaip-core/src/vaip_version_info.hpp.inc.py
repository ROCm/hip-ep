import sys
import os
import re
import datetime
import pathlib
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
def main2(workspace_directory):
    output_file ="vaip_version_info.hpp.inc"
    with open(output_file, "w") as f:
        for project in PROJECTS:
            if not os.path.exists(workspace_directory/ project):
                print(f"Project {project} not found in {workspace_directory}")
                continue
            project_directory = workspace_directory/ project;
            print(f"Processing {project_directory}")
            git_commit_id = os.popen(f"cd {project_directory} && git rev-parse HEAD").read().strip()
            project_name = project
            project_version_id = os.popen(f"cd {project_directory} &&git describe --tags --abbrev=1 HEAD").read().strip()
            output = f"""
                {{"{project_name}", "{git_commit_id}", "{project_version_id}"}},
            """
            f.write(output)

def main(release_file):
    output_file ="vaip_version_info.hpp.inc"
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: vaip_version_info.hpp.inc.py <DIR>")
        sys.exit(1)
    workspace_directory = pathlib.Path(sys.argv[1])
    main(workspace_directory)