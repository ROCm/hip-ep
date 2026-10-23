/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"

#include <iostream>

namespace {
void PrintHelp() {
  std::cout
      << "usage: ktrace_postprocess.exe <command> [options]\n\n"
      << "commands:\n"
      << "  finalize  Convert private spools to rocprofv3 CSV/JSON\n"
      << "  analyze   Generate the ktrace text performance report\n"
      << "  chrome    Export a Chrome/Perfetto trace\n"
      << "  export    Export JSON and/or native pftrace from one event stream\n"
      << "  all       Finalize, analyze, and export the full Perfetto trace\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" ||
      std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
    PrintHelp();
    return 0;
  }

  std::vector<std::string> args;
  for (int i = 2; i < argc; ++i)
    args.emplace_back(argv[i]);
  try {
    const std::string command = argv[1];
    if (command == "finalize")
      return ktrace::FinalizeCommand(args);
    if (command == "analyze")
      return ktrace::AnalyzeCommand(args);
    if (command == "chrome")
      return ktrace::ChromeCommand(args);
    if (command == "export")
      return ktrace::ChromeCommand(args);
    if (command == "all")
      return ktrace::AllCommand(args);
    std::cerr << "error: unknown command '" << command << "'\n\n";
    PrintHelp();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
