import subprocess
import sys
import pathlib

CURRENT_DIR = pathlib.Path(
    __file__
).parent.resolve()  # Get the current directory of the script
# Define the test executable path
test_executable = sys.argv[1]  # Get the test executable from command line arguments

TEMPLATE = r"""
add_custom_target(morphizen-unit-test-{suite}-{case}
    COMMAND $<TARGET_FILE:${{TEST_EXE_NAME}}> --gtest_filter={suite}.{case}
    DEPENDS ${{TEST_EXE_NAME}}
    WORKING_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}
    )
set_target_properties(morphizen-unit-test-{suite}-{case} PROPERTIES
    FOLDER "morphizen/unit-tests/cases/{suite}/{case}"
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:${{TEST_EXE_NAME}}>"
    VS_DEBUGGER_WORKING_DIRECTORY "$(ProjectDir)"
    VS_DEBUGGER_COMMAND_ARGUMENTS  --gtest_filter={suite}.{case}
    )
#end
    """
# Run the test executable and capture its output
try:
    result = subprocess.run(
        [test_executable, "--gtest_list_tests"],
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

for line in lines:
    if line.endswith("."):  # Test suite
        suite = line.strip()
        suite = suite[:-1]  # Remove the trailing dot
    elif line.startswith("  "):  # Test case
        test_cases.append({"suite": suite, "case": line.strip()})

# Write the flattened test list to a file
fname = CURRENT_DIR / "generated_gtest_targets.cmake"
with open(fname, "w") as f:
    for test_case in test_cases:
        # Write the formatted string to the file
        f.write(TEMPLATE.format(**test_case))

print(f"write to {fname} successfully")
