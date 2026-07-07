/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "morphizen/util.hpp"
#include "md5.h"
#include <cstdio>

#include <glog/logging.h>

#include "morphizen/graph.hpp"
#include "morphizen/pass_context.hpp"
#include "pass_context_imp.hpp"
#include <morphizen/morphizen_ort_api.h>

#include "morphizen/env_config.hpp"
#include "morphizen/temp_file_stream.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#ifdef ENABLE_PYTHON
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

DEF_ENV_PARAM(DEBUG_MORPHIZEN_UTIL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_MORPHIZEN_UTIL) >= n)

namespace morphizen {
MORPHIZEN_DLL_SPEC void dump_graph(const Graph &graph,
                                   const std::string &filename) {
  std::ofstream out(filename);
  auto text = morphizen_cxx::GraphConstRef(graph).to_string();
  out << text;
  out.close();
}

MORPHIZEN_DLL_SPEC std::unique_ptr<int> scale_to_fix_point(float scale) {
  auto fix_point = (int)(std::log2f(1 / scale));
  if (std::exp2f((float)fix_point) * scale == 1) {
    return std::make_unique<int>(fix_point);
  } else
    return std::make_unique<int>();
}

static std::vector<std::string> split_path(const char *env_name) {
  std::string path;
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  auto env_value = getenv(env_name);
  path = env_value != nullptr ? env_value : "";
#ifdef _WIN32
#pragma warning(pop)
#endif
  auto ret = std::vector<std::string>();
#ifdef _MSC_VER
  char sep = ';';
#else
  char sep = ':';
#endif

  std::string::size_type pos0 = 0u;
  for (auto pos = path.find(sep, pos0); pos != std::string::npos;
       pos = path.find(sep, pos0)) {
    ret.push_back(path.substr(pos0, pos - pos0));
    pos0 = pos + 1;
  }
  if (pos0 != std::string::npos) {
    ret.push_back(path.substr(pos0, path.size() - pos0));
  }
  return ret;
}

std::string find_file_in_path(const std::string &file, const char *env_name,
                              bool required) {
  auto path = split_path(env_name);
  namespace fs = std::filesystem;
  for (auto &p : path) {
    auto dir_path = fs::path(p);
    auto file_path = dir_path / fs::path(file);
    MY_LOG(1) << "for vai_config.json trying " << file_path;
    if (fs::exists(file_path)) {
      return file_path.u8string();
    }
  }
  std::ostringstream str;
  if (required) {
    str << "cannot find file " << file << " after searching following path\n";
    for (auto &p : path) {
      auto dir_path = fs::path(p);
      auto file_path = dir_path / fs::path(file);
      str << "\t" << file_path << "\n";
    }
    str << "please check enviroment variable " << env_name;
    LOG(FATAL) << str.str();
  }
  return std::string();
}

std::string slurp(const char *filename) {
  return slurp(std::filesystem::u8path(std::string(filename)));
}
MORPHIZEN_DLL_SPEC std::string slurp(const std::filesystem::path &path) {
  std::ifstream in;
  in.open(path, std::ifstream::in);
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}
std::string slurp_if_exists(const std::filesystem::path &path) {
  if (std::filesystem::exists(path))
    return slurp(path);
  else
    return std::string("");
}
#ifdef ENABLE_PYTHON
std::shared_ptr<void> init_interpreter() {
  static std::mutex mtx;
  static std::weak_ptr<void> py_interpreter_holder;
  std::shared_ptr<void> ret;
  std::lock_guard<std::mutex> lock(mtx);
  if (!Py_IsInitialized()) {
    static py::scoped_interpreter inter{};
    auto p = static_cast<void *>(&inter);
    ret = std::shared_ptr<void>(p, [](void * /*p*/) {});
    py_interpreter_holder = ret;
  }
  if (!ret) {
    ret = py_interpreter_holder.lock();
  }
  return ret;
}

MORPHIZEN_DLL_SPEC void eval_python_code(const std::string &code) {
  auto inter = init_interpreter();
  py::gil_scoped_acquire acquire;
  py::eval(code);
}
#endif

std::string dos2unix(const gsl::span<const char> input) {
  std::string ret;
  ret.reserve(input.size());
  for (auto c : input) {
    if (c == '\r') {
      continue;
    }
    ret.push_back(c);
  }
  return ret;
}
template <typename T> struct binary_io {
  using char_type = T;
  static std::vector<char_type>
  slurp_binary(const std::filesystem::path &filename) {
    std::ifstream is(filename, std::ios::binary);
    CHECK(is.good()) << "cannot open file " << filename;
    CHECK(is.seekg(0, std::ios_base::end).good());
    auto size = is.tellg();
    CHECK_NE(size, -1);
    CHECK(is.seekg(0, std::ios_base::beg).good());
    auto buffer = std::vector<char_type>((size_t)size / sizeof(char_type));
    CHECK(is.read(reinterpret_cast<char *>(buffer.data()), size).good());
    return buffer;
  }

  static bool dump_binary(const std::filesystem::path &filename,
                          gsl::span<const char_type> data) {
    std::ofstream out(filename, std::ios::binary);
    CHECK(out.write(reinterpret_cast<const char *>(data.data()),
                    data.size() * sizeof(char_type))
              .good());
    return true;
  }
};

std::vector<uint8_t> slurp_binary_u8(const std::filesystem::path &filename) {
  return binary_io<uint8_t>::slurp_binary(filename);
}
std::vector<int8_t> slurp_binary_i8(const std::filesystem::path &filename) {
  return binary_io<int8_t>::slurp_binary(filename);
}
std::vector<char> slurp_binary_c8(const std::filesystem::path &filename) {
  return binary_io<char>::slurp_binary(filename);
}

bool dump_binary(const std::filesystem::path &filename,
                 gsl::span<const uint8_t> data) {
  return binary_io<uint8_t>::dump_binary(filename, data);
}
bool dump_binary(const std::filesystem::path &filename,
                 gsl::span<const int8_t> data) {
  return binary_io<int8_t>::dump_binary(filename, data);
}
bool dump_binary(const std::filesystem::path &filename,
                 gsl::span<const char> data) {
  return binary_io<char>::dump_binary(filename, data);
}

// Wrapper stream that owns the TempFileStream
namespace {
class TempFileIstream : public std::istream {
public:
  TempFileIstream(std::unique_ptr<TempFileStream> temp)
      : std::istream(temp->get_read_stream().rdbuf()), temp_(std::move(temp)) {}

private:
  std::unique_ptr<TempFileStream> temp_;
};
} // namespace

std::unique_ptr<std::istream>
context_cache_files_to_tar_stream(PassContext &context) {
  auto &ctx_imp = dynamic_cast<PassContextImp &>(context);
  CHECK(ctx_imp.tar_file_ != nullptr) << "tar_file_ should exist";

  auto size = ctx_imp.tar_file_->current_size();
  auto buffer = std::make_shared<std::string>(size, '\0');
  CHECK(ctx_imp.tar_file_->dump_to(buffer->data(), size))
      << "failed to dump tar file";

  return std::make_unique<std::istringstream>(*buffer, std::ios::binary);
}

std::string get_md5_of_buffer(const char *buffer, size_t size) {
  auto MD5_computer = MD5();
  MD5_computer.add(buffer, size);
  return MD5_computer.getHash();
}
std::string get_md5_of_file(const std::filesystem::path &path) {
  if (!std::filesystem::exists(path))
    return "";
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return "";
  }
  char buffer[1024] = {0};
  auto sz = file.read(buffer, 1024).gcount();
  auto MD5_computer = MD5();
  while (sz != 0) {
    MD5_computer.add(buffer, sz);
    sz = file.read(buffer, 1024).gcount();
  }
  file.close();
  return MD5_computer.getHash();
}
} // namespace morphizen
