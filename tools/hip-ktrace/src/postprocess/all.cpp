/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "common.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

namespace ktrace {
namespace {
bool IsPrivateSpool(const fs::path &path) {
  const std::string name = path.filename().string();
  return !name.empty() && name.front() == '.' &&
         EndsWithInsensitive(name, "_ops.csv");
}

bool IsKernelTrace(const fs::path &path) {
  return EndsWithInsensitive(path.filename().string(), "_kernel_trace.csv");
}

int ProcessCapture(const fs::path &kernel_trace, bool fixed_names,
                   bool chrome_json, bool chrome_pftrace, bool flows) {
  const Capture capture = ResolveCapture(kernel_trace);
  const fs::path report = fixed_names
                              ? kernel_trace.parent_path() / "ktrace_report.txt"
                              : capture.AutoOutput("_ktrace_report.txt");
  const fs::path chrome =
      fixed_names ? kernel_trace.parent_path() / "ktrace_perfetto.json"
                  : capture.AutoOutput("_ktrace_chrome.json");
  const fs::path pftrace =
      fixed_names ? kernel_trace.parent_path() / "ktrace_perfetto.pftrace"
                  : capture.AutoOutput("_ktrace_perfetto.pftrace");

  std::cout << "[all] analyze + trace export in parallel\n"
            << "[all] report   -> " << report.string() << '\n';
  if (chrome_json)
    std::cout << "[all] JSON     -> " << chrome.string() << '\n';
  if (chrome_pftrace)
    std::cout << "[all] pftrace  -> " << pftrace.string() << '\n';

  auto analyze = std::async(std::launch::async, [&] {
    const auto start = std::chrono::steady_clock::now();
    const int result = AnalyzeCommand(
        {kernel_trace.string(), "-o", report.string(), "--quiet"});
    return std::make_pair(result, std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
  });
  auto perfetto = std::async(std::launch::async, [&] {
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::string> export_args{kernel_trace.string(), "--all",
                                         "--quiet"};
    if (flows)
      export_args.push_back("--flows");
    if (chrome_json) {
      export_args.push_back("--chrome-json");
      export_args.push_back(chrome.string());
    }
    if (chrome_pftrace) {
      export_args.push_back("--chrome-pftrace");
      export_args.push_back(pftrace.string());
    }
    const int result = ChromeCommand(export_args);
    return std::make_pair(result, std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
  });

  const auto [analyze_result, analyze_seconds] = analyze.get();
  const auto [perfetto_result, export_seconds] = perfetto.get();
  std::cout << "[all] analyze " << analyze_seconds << " s, export "
            << export_seconds << " s\n";
  if (analyze_result)
    return analyze_result;
  if (perfetto_result)
    return perfetto_result;
  std::cout << "[all] report and trace export complete\n";
  return 0;
}
} // namespace

int AllCommand(const std::vector<std::string> &args) {
  if (args.empty() || args[0] == "-h" || args[0] == "--help" ||
      args[0] == "help") {
    std::cout
        << "usage: ktrace_postprocess.exe all <capture-dir|capture>\n"
        << "       [--chrome-json] [--chrome-pftrace] [--flows]\n"
        << "  Finalize every private spool, create a text report, and export\n"
        << "  the selected full-range trace format for every process.\n"
        << "  With no format flag, Chrome JSON remains the default.\n";
    return args.empty() ? 2 : 0;
  }
  bool chrome_json = false;
  bool chrome_pftrace = false;
  bool flows = false;
  bool format_selected = false;
  std::vector<std::string> positional;
  for (const auto &arg : args) {
    if (arg == "--chrome-json") {
      chrome_json = true;
      format_selected = true;
    } else if (arg == "--chrome-pftrace") {
      chrome_pftrace = true;
      format_selected = true;
    } else if (arg == "--flows")
      flows = true;
    else if (!arg.empty() && arg.front() == '-')
      throw std::runtime_error("unknown all option: " + arg);
    else
      positional.push_back(arg);
  }
  if (!format_selected)
    chrome_json = true;
  if (positional.size() != 1)
    throw std::runtime_error(
        "all accepts one capture directory, prefix, or file");

  const fs::path source = fs::absolute(positional[0]);
  std::vector<fs::path> kernel_traces;

  if (fs::is_directory(source)) {
    bool has_spool = false;
    for (const auto &entry : fs::directory_iterator(source))
      if (entry.is_regular_file() && IsPrivateSpool(entry.path())) {
        has_spool = true;
        break;
      }

    if (has_spool) {
      std::cout << "[all] finalize -> " << source.string() << '\n';
      const int result = FinalizeCommand({source.string()});
      if (result)
        return result;
    }

    for (const auto &entry : fs::directory_iterator(source))
      if (entry.is_regular_file() && IsKernelTrace(entry.path()))
        kernel_traces.push_back(entry.path());
    std::sort(kernel_traces.begin(), kernel_traces.end());
  } else {
    const Capture capture = ResolveCapture(source);
    const fs::path kernel = capture.Domain("kernels");
    if (fs::exists(kernel))
      kernel_traces.push_back(kernel);
  }

  if (kernel_traces.empty())
    throw std::runtime_error(source.string() +
                             ": no private spool or kernel trace found");

  const bool fixed_names = kernel_traces.size() == 1;
  for (const auto &kernel : kernel_traces) {
    const int result =
        ProcessCapture(kernel, fixed_names, chrome_json, chrome_pftrace, flows);
    if (result)
      return result;
  }
  std::cout << "[all] complete: " << kernel_traces.size()
            << " process capture(s)\n";
  return 0;
}
} // namespace ktrace
