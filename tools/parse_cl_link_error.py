## to developers:
#
# This script is used to parse the error message from the linker and
# extract the unresolved symbols.  The unresolved symbols are then
# added to the def file to resolve the symbols.  The script takes 2
# arguments:
#
#   1. The error log message from the linker
#   2. The def file to be used to resolve the symbols
#
# The script reads the def file and extracts the symbols and then reads the error log message and extracts the unresolved symbols.
# The script then writes the symbols to the def file.
# The script is used to automate the process of resolving the linker errors.
#
# here is an example how to use the script:
#
#   % cd $W/MorphiZen/
#   % git checkout vaip-core/onnxruntime_vionnxruntime_vitisai_ep.def # restore the def file to the original state
#   % p=$BUILD/MorphiZen/build; set -o pipefail; cmake --build $p -j56 --config Debug  | tee c:/temp/error.log && cmake --install $p --config Debug
#   % python parse_cl_link_error.py c:/temp/error.log $W/MorphiZen/vaip-core/onnxruntime_vionnxruntime_vitisai_ep.def
#
import re
import argparse

# Regular expression to extract the fields
pattern = r"^(.+)\s*:\s*error\s*(LNK\d+):\s*unresolved\s*external\s*symbol\s*\"(.*?)\"\s*\((.+?)\)(\s*referenced\s*in\s* function.*)?"


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse the error message from the linker"
    )
    parser.add_argument("file", help="The error log message from the linker")
    parser.add_argument(
        "def_file", help="The def file to be used to resolve the symbols"
    )
    args = parser.parse_args()
    print(f"error log file = {args.file}")
    print(f"def_file = {args.def_file}")
    headers = ""
    symbols = set()
    with open(args.def_file) as f:
        # skip the first 2 lines and save it to to a variable
        headers += next(f)
        headers += next(f)
        for line in f:
            match = re.match(r"^\s*(.+)", line)
            if match:
                symbols.add(match.group(1))
            else:
                print(f"line does not match {line}")
    print(f"{headers}")
    for s in symbols:
        print(f"   {s}")
    with open(args.file) as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                group = match.groups()
                print(f"debug grpah= {group}")
                file_name, error_code, cxx_symbol, unresolved_symbol, referenced = group
                print("unresolved_symbol:", unresolved_symbol)
                symbols.add(unresolved_symbol)
            else:
                print(f"No match found. {line}")
                pass
    symbols = sorted(symbols)
    with open(args.def_file, "w") as f:
        f.write(headers)
        for s in symbols:
            f.write(f"    {s}\n")


if __name__ == "__main__":
    main()
