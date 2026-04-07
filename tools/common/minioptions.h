/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class MiniOptions {
public:
  struct Option {
    std::string long_name;
    std::string short_name;
    std::vector<std::string> values;
    std::string default_value;
    std::string description;
    bool is_flag = false;
  };

  // Add an option (updated API)
  void add_option(const std::string &short_name, const std::string &long_name,
                  const std::string &description,
                  const std::string &default_value, bool is_flag = false) {
    Option opt;
    opt.short_name = short_name;
    opt.long_name = long_name;
    opt.description = description;
    opt.default_value = default_value;
    opt.is_flag = is_flag;

    options_["--" + long_name] = opt;
    name_map_[long_name] = "--" + long_name;

    if (!short_name.empty()) {
      options_["-" + short_name] = opt;
    }
  }

  void parse(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--help" || arg == "-h") {
        print_help(argv[0]);
        std::exit(0);
      }

      if (arg.rfind("-", 0) == 0 && !options_.count(arg) &&
          arg.find('=') == std::string::npos) {
        throw std::runtime_error("Unknown option: " + arg);
      }

      auto eq = arg.find('=');
      if (eq != std::string::npos) {
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);

        if (!options_.count(key)) {
          throw std::runtime_error("Unknown option: " + key);
        }

        set_value(key, val);
        continue;
      }

      if (options_.count(arg)) {
        const auto &opt = options_.at(arg);

        if (opt.is_flag) {
          set_value(arg, "true");
        } else {
          if (i + 1 >= argc || argv[i + 1][0] == '-') {
            throw std::runtime_error("Missing value for option: " + arg);
          }

          std::string val = argv[++i];
          set_value(arg, val);
        }
      } else if (!arg.empty() && arg[0] != '-') {
        throw std::runtime_error("Unexpected argument: " + arg);
      }
    }
  }

  void print_help(const std::string &prog) const {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";

    for (const auto &[key, opt] : options_) {
      if (key != "--" + opt.long_name)
        continue;

      std::ostringstream names;
      if (!opt.short_name.empty()) {
        names << "-" << opt.short_name << ", ";
      }
      names << "--" << opt.long_name;

      std::cout << std::left << std::setw(28) << names.str();
      std::cout << opt.description;

      if (!opt.default_value.empty() && !opt.is_flag) {
        std::cout << " (default: " << opt.default_value << ")";
      }

      if (opt.is_flag) {
        std::cout << " (flag)";
      }

      std::cout << "\n";
    }
  }

  size_t count(const std::string &name) const {
    auto key = get_key(name);
    return options_.at(key).values.size();
  }

  bool exists(const std::string &name) const { return count(name) > 0; }

  template <typename T> T get(const std::string &name) const {
    auto key = get_key(name);
    const auto &opt = options_.at(key);

    std::string val =
        opt.values.empty() ? opt.default_value : opt.values.back();

    try {
      return convert<T>(val);
    } catch (...) {
      throw std::runtime_error("Type conversion failed for --" + name +
                               " value: " + val);
    }
  }

  template <typename T>
  std::vector<T> get_vector(const std::string &name) const {
    auto key = get_key(name);
    const auto &opt = options_.at(key);

    std::vector<T> result;

    for (auto &v : opt.values) {
      std::stringstream ss(v);
      std::string item;
      while (std::getline(ss, item, ',')) {
        try {
          result.push_back(convert<T>(item));
        } catch (...) {
          throw std::runtime_error("Type conversion failed for --" + name +
                                   " value: " + item);
        }
      }
    }

    return result;
  }

private:
  std::unordered_map<std::string, Option> options_;
  std::unordered_map<std::string, std::string> name_map_;

  std::string get_key(const std::string &name) const {
    if (!name_map_.count(name)) {
      throw std::runtime_error("Option not found: " + name);
    }
    return name_map_.at(name);
  }

  // Keep short (-m) and long (--model) map entries in sync for get("model").
  void set_value(const std::string &key, const std::string &val) {
    Option &o = options_.at(key);
    o.values.push_back(val);
    const std::string longK = "--" + o.long_name;
    const std::string shortK =
        o.short_name.empty() ? std::string() : "-" + o.short_name;
    if (key == longK && !shortK.empty())
      options_[shortK].values.push_back(val);
    else if (!shortK.empty() && key == shortK)
      options_[longK].values.push_back(val);
  }

  template <typename T> static T convert(const std::string &s) {
    std::istringstream iss(s);
    T v;
    if (!(iss >> v)) {
      throw std::runtime_error("Bad conversion");
    }
    return v;
  }
};

// Specializations

template <>
inline std::string MiniOptions::convert<std::string>(const std::string &s) {
  return s;
}

template <> inline bool MiniOptions::convert<bool>(const std::string &s) {
  if (s == "true" || s == "1")
    return true;
  if (s == "false" || s == "0")
    return false;
  throw std::runtime_error("Invalid bool value: " + s);
}

template <>
inline unsigned int MiniOptions::convert<unsigned int>(const std::string &s) {
  unsigned long v = std::stoul(s);
  return static_cast<unsigned int>(v);
}

/*  usage example

#include "mini_options.hpp"
#include <iostream>

int main(int argc, char** argv) {
    MiniOptions opts;

    // ===== Define options =====
    opts.add_option("m", "model", "Path to ONNX model", "");
    opts.add_option("e", "ep", "Execution provider (cpu/cuda/hip)", "cpu");
    opts.add_option("n", "num_runs", "Number of runs", "10");
    opts.add_option("t", "threshold", "Score threshold", "0.5");
    opts.add_option("v", "verbose", "Enable verbose logging", "false", true);
    opts.add_option("", "input", "Input files (repeatable or comma-separated)",
                     "");
    try {
        // ===== Parse CLI =====
        opts.parse(argc, argv);

        // ===== Manual validation =====
        if (!opts.exists("model")) {
            throw std::runtime_error("Option --model is required");
        }

        // ===== Read values =====
        std::string model = opts.get<std::string>("model");
        std::string ep = opts.get<std::string>("ep");
        int num_runs = opts.get<int>("num_runs");
        float threshold = opts.get<float>("threshold");
        bool verbose = opts.get<bool>("verbose");

        std::vector<std::string> inputs =
            opts.get_vector<std::string>("input");

        // ===== Simulated usage =====
        std::cout << "=== Configuration ===\n";
        std::cout << "Model:      " << model << "\n";
        std::cout << "EP:         " << ep << "\n";
        std::cout << "Runs:       " << num_runs << "\n";
        std::cout << "Threshold:  " << threshold << "\n";
        std::cout << "Verbose:    " << (verbose ? "true" : "false") << "\n";

        if (opts.exists("input")) {
            std::cout << "Inputs (" << opts.count("input") << "):\n";
            for (const auto& s : inputs) {
                std::cout << "  - " << s << "\n";
            }
        } else {
            std::cout << "No input files provided\n";
        }

        // ===== Example logic =====
        if (verbose) {
            std::cout << "[DEBUG] Starting inference loop...\n";
        }

        for (int i = 0; i < num_runs; ++i) {
            if (verbose) {
                std::cout << "[DEBUG] Run " << i << "\n";
            }
            // simulate inference
        }

        std::cout << "Done.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        opts.print_help(argv[0]);
        return 1;
    }

    return 0;
}
*/
