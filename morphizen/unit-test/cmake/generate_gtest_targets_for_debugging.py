#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import subprocess
import sys
import pathlib
import re

CURRENT_DIR = pathlib.Path(
    __file__
).parent.resolve()  # Get the current directory of the script
# Define the test executable path
test_executable = sys.argv[1]  # Get the test executable from command line arguments

TEMPLATE = r"""
add_custom_target(morphizen-unit-test-{suite}.{case}.{line}
    COMMAND $<TARGET_FILE:${{TEST_EXE_NAME}}> --gtest_filter={suite}.{case}
    DEPENDS ${{TEST_EXE_NAME}}
    WORKING_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}
    )
set_target_properties(morphizen-unit-test-{suite}.{case}.{line} PROPERTIES
    FOLDER "morphizen/unit-tests/cases/{suite}/{case}"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${{TEST_EXE_NAME}}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter={suite}.{case}
    )
target_sources(morphizen-unit-test-{suite}.{case}.{line} PRIVATE
        # {file} # line {line} don't add c++ file, otherwise `compile` does works
        cmake/generate_gtest_targets_for_debugging.py
        cmake/generated_gtest_targets.cmake
)
source_group("CMake Files" FILES cmake/generated_gtest_targets.cmake)
source_group("Python Codes" FILES cmake/generate_gtest_targets_for_debugging.py)
#end
    """
# Run the test executable and capture its output
try:
    result = subprocess.run(
        [test_executable, "--gtest_list_test_cases"],
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error (optional)
        text=True,  # Decode output as a string
        check=True,  # Raise an exception if the command fails
    )
except subprocess.CalledProcessError as e:
    print(f"Error while running the test executable: {e.stderr}")
    exit(1)

# Parse the output directly
lines = result.stdout.splitlines()  # Split the output into lines
suite = None
test_cases = []
# Regex to match test cases with file paths and line numbers
test_case_pattern = re.compile(r"  (?P<case>.+) (?P<file>.+):(?P<line>\d+)$")

for line in lines:
    if line.endswith("."):  # Test suite
        suite = line.strip()
        suite = suite[:-1]  # Remove the trailing dot
    elif match := test_case_pattern.match(line):  # Test case
        groupdict = match.groupdict()
        file = pathlib.Path(groupdict["file"])
        # get relative path to CURRENT_DIR
        file = file.relative_to((CURRENT_DIR / "..").resolve())
        groupdict["file"] = str(file.as_posix())
        test_cases.append({"suite": suite, **groupdict})

# Write the flattened test list to a file
fname = CURRENT_DIR / "generated_gtest_targets.cmake"
with open(fname, "w") as f:
    for test_case in test_cases:
        # Write the formatted string to the file
        f.write(TEMPLATE.format(**{**test_case}))

print(f"write to {fname} successfully")
