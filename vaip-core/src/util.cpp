/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
#include "morphizen/util.hpp"
#include "hash-library/md5.h"
#include <cstdio>

#include <glog/logging.h>

#include "morphizen/graph.hpp"
#include <vaip/vaip_ort_api.h>

#include "morphizen/env_config.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#ifdef ENABLE_PYTHON
#  include <pybind11/embed.h>
#  include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

DEF_ENV_PARAM(DEBUG_VAIP_UTIL, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_VAIP_UTIL) >= n)

namespace vaip_core {
VAIP_DLL_SPEC void dump_graph(const Graph& graph, const std::string& filename) {
  std::ofstream out(filename);
  auto text = graph_as_string(graph);
  out << text;
  out.close();
}

VAIP_DLL_SPEC std::unique_ptr<int> scale_to_fix_point(float scale) {
  auto fix_point = (int)(std::log2f(1 / scale));
  if (std::exp2f((float)fix_point) * scale == 1) {
    return std::make_unique<int>(fix_point);
  } else
    return std::make_unique<int>();
}

std::string convert_to_xir_op_type(const std::string& domain,
                                   const std::string& op_type) {
  if (domain == "com.xilinx") {
    if (op_type.size() >= 4u && op_type.substr(op_type.size() - 4u) == "_fix") {
      return op_type.substr(0u, op_type.size() - 4u) + "-fix";
    } else if (op_type == "transposed_conv2d") {
      return "transposed-conv2d";
    } else if (op_type == "leaky_relu") {
      return "leaky-relu";
    } else if (op_type == "depthwise_conv2d") {
      return "depthwise-conv2d";
    } else if (op_type == "depthwise_conv1d") {
      return "depthwise-conv1d";
    } else if (op_type == "hard_sigmoid") {
      return "hard-sigmoid";
    } else if (op_type == "hard_softmax") {
      return "hard-softmax";
    } else if (op_type == "pixel_shuffle") {
      return "pixel-shuffle";
    } else if (op_type == "quantize_linear") {
      return "quantize-linear";
    } else if (op_type == "dequantize_linear") {
      return "dequantize-linear";
    } else if (op_type == "quantize_linear_int8") {
      return "quantize-linear-int8";
    } else if (op_type == "quantize_linear_uint8") {
      return "quantize-linear-uint8";
    } else if (op_type == "dequantize_linear_int8") {
      return "dequantize-linear-int8";
    } else if (op_type == "dequantize_linear_uint8") {
      return "dequantize_linear_uint8";
    }
    return op_type;
  }
  return domain + ":" + op_type;
}

static std::vector<std::string> split_path(const char* env_name) {
  std::string path;
#ifdef _WIN32
#  pragma warning(push)
#  pragma warning(disable : 4996)
#endif
  auto env_value = getenv(env_name);
  path = env_value != nullptr ? env_value : "";
#ifdef _WIN32
#  pragma warning(pop)
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

std::string find_file_in_path(const std::string& file, const char* env_name,
                              bool required) {
  auto path = split_path(env_name);
  namespace fs = std::filesystem;
  for (auto& p : path) {
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
    for (auto& p : path) {
      auto dir_path = fs::path(p);
      auto file_path = dir_path / fs::path(file);
      str << "\t" << file_path << "\n";
    }
    str << "please check enviroment variable " << env_name;
    LOG(FATAL) << str.str();
  }
  return std::string();
}

std::string slurp(const char* filename) {
  return slurp(std::filesystem::u8path(std::string(filename)));
}
VAIP_DLL_SPEC std::string slurp(const std::filesystem::path& path) {
  std::ifstream in;
  in.open(path, std::ifstream::in);
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}
std::string slurp_if_exists(const std::filesystem::path& path) {
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
    auto p = static_cast<void*>(&inter);
    ret = std::shared_ptr<void>(p, [](void* p) {});
    py_interpreter_holder = ret;
  }
  if (!ret) {
    ret = py_interpreter_holder.lock();
  }
  return ret;
}

VAIP_DLL_SPEC void eval_python_code(const std::string& code) {
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
  slurp_binary(const std::filesystem::path& filename) {
    std::ifstream is(filename, std::ios::binary);
    CHECK(is.good()) << "cannot open file " << filename;
    CHECK(is.seekg(0, std::ios_base::end).good());
    auto size = is.tellg();
    CHECK_NE(size, -1);
    CHECK(is.seekg(0, std::ios_base::beg).good());
    auto buffer = std::vector<char_type>((size_t)size / sizeof(char_type));
    CHECK(is.read(reinterpret_cast<char*>(buffer.data()), size).good());
    return buffer;
  }

  static bool dump_binary(const std::filesystem::path& filename,
                          gsl::span<const char_type> data) {
    std::ofstream out(filename, std::ios::binary);
    CHECK(out.write(reinterpret_cast<const char*>(data.data()),
                    data.size() * sizeof(char_type))
              .good());
    return true;
  }
};

std::vector<uint8_t> slurp_binary_u8(const std::filesystem::path& filename) {
  return binary_io<uint8_t>::slurp_binary(filename);
}
std::vector<int8_t> slurp_binary_i8(const std::filesystem::path& filename) {
  return binary_io<int8_t>::slurp_binary(filename);
}
std::vector<char> slurp_binary_c8(const std::filesystem::path& filename) {
  return binary_io<char>::slurp_binary(filename);
}

bool dump_binary(const std::filesystem::path& filename,
                 gsl::span<const uint8_t> data) {
  return binary_io<uint8_t>::dump_binary(filename, data);
}
bool dump_binary(const std::filesystem::path& filename,
                 gsl::span<const int8_t> data) {
  return binary_io<int8_t>::dump_binary(filename, data);
}
bool dump_binary(const std::filesystem::path& filename,
                 gsl::span<const char> data) {
  return binary_io<char>::dump_binary(filename, data);
}

/**
 * Compresses the given memory data using the ZIP algorithm.
 *
 * @param data The memory data to be compressed.
 * @return A vector containing the compressed data.
 */
#include <zlib.h>
template <typename T> struct zlib {
  using char_type = T;

  /*
        Update a running crc with the bytes buf[0..len-1] and return
      the updated crc. The crc should be initialized to zero. Pre- and
      post-conditioning (one's complement) is performed within this
      function so it shouldn't be done by the caller. Usage example:

        unsigned long crc = 0L;

        while (read_buffer(buffer, length) != EOF) {
          crc = update_crc(crc, buffer, length);
        }
        if (crc != original_crc) error();
     */
  static unsigned long update_crc(unsigned long crc, const unsigned char* buf,
                                  int len) {
    /* Table of CRCs of all 8-bit messages. */
    static unsigned long crc_table[256];
    /* Flag: has the table been computed? Initially false. */
    static int crc_table_computed;
    /* Make the table for a fast CRC. */
    auto make_crc_table = [&] {
      unsigned long c;
      int n, k;
      for (n = 0; n < 256; n++) {
        c = (unsigned long)n;
        for (k = 0; k < 8; k++) {
          if (c & 1) {
            c = 0xedb88320L ^ (c >> 1);
          } else {
            c = c >> 1;
          }
        }
        crc_table[n] = c;
      }
      crc_table_computed = 1;
    };
    unsigned long c = crc ^ 0xffffffffL;
    int n;

    if (!crc_table_computed) {
      make_crc_table();
    }
    for (n = 0; n < len; n++) {
      c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffL;
  }

  /* Return the CRC of the bytes buf[0..len-1]. */
  static unsigned long calculate_crc(const unsigned char* buf, int len) {
    return update_crc(0L, buf, len);
  }
  static std::vector<char_type> compress(gsl::span<const char_type> data,
                                         int level) {
#if _WIN32
    FILE* tmp_file = nullptr;
    auto err = tmpfile_s(&tmp_file);
    CHECK_EQ(err, 0) << "tmpfile_s error";
    auto fd = _fileno(tmp_file);
#else
    FILE* tmp_file = tmpfile();
    CHECK(tmp_file != nullptr) << "cannot create tmp file";
    auto fd = fileno(tmp_file);
#endif
    auto gzfile = gzdopen(fd, "wb");
    CHECK(gzfile != nullptr) << "gzopen error";
    auto status = gzwrite(gzfile, data.data(), (unsigned)data.size());
    CHECK_EQ((size_t)status, data.size()) << "gzwrite error";
    status = gzflush(gzfile, Z_FINISH);
    CHECK_EQ(status, 0) << "gzflush error";
    status = fseek64(tmp_file, 0, SEEK_END);
    CHECK(status == 0) << "fseek error";
    auto size = ftell64(tmp_file);
    CHECK_NE(size, -1) << "ftell error";
    status = fseek64(tmp_file, 0, SEEK_SET);
    auto output_buffer = std::vector<char_type>((size_t)size);
    auto read_size =
        std::fread(output_buffer.data(), 1, output_buffer.size(), tmp_file);
    CHECK_EQ((size_t)read_size, output_buffer.size());
    fclose(tmp_file);
    return output_buffer;
  }

  static std::vector<char_type> uncompress(gsl::span<const char_type> data) {
#if _WIN32
    FILE* tmp_file = nullptr;
    auto err = tmpfile_s(&tmp_file);
    CHECK_EQ(err, 0) << "tmpfile_s error";
#else
    FILE* tmp_file = tmpfile();
    CHECK(tmp_file != nullptr) << "cannot create tmp file";
    int err = 0;
#endif
    auto write_size = std::fwrite(data.data(), 1, data.size(), tmp_file);
    CHECK_EQ((size_t)write_size, data.size());
    err = fflush(tmp_file);
    auto status = fseek64(tmp_file, 0, SEEK_SET);
    CHECK_EQ(status, 0) << "fseek error";
    CHECK_EQ(err, 0) << "fflush error";
#if _WIN32
    auto fd = _fileno(tmp_file);
#else
    auto fd = fileno(tmp_file);
#endif
    auto gzfile = gzdopen(fd, "rb");
    CHECK(gzfile != nullptr) << "gzopen error";
    auto output_buffer = std::vector<char_type>();
    auto tmpbuffer = std::vector<char_type>(8 * 1024u);
    do {
      auto read_size =
          (size_t)gzread(gzfile, tmpbuffer.data(),
                         (unsigned)(tmpbuffer.size() * sizeof(char_type)));
      int err_num = 0;
      auto err_msg = gzerror(gzfile, &err_num);
      CHECK_EQ(err_num, Z_OK) << "gzread error: " << err_msg;
      output_buffer.insert(output_buffer.end(), tmpbuffer.begin(),
                           tmpbuffer.begin() + read_size / sizeof(char_type));
    } while (!gzeof(gzfile));
    fclose(tmp_file);
    return output_buffer;
  }
};
std::vector<uint8_t> compress(gsl::span<const uint8_t> data, int level) {
  return zlib<uint8_t>::compress(data, level);
}
std::vector<int8_t> compress(gsl::span<const int8_t> data, int level) {
  return zlib<int8_t>::compress(data, level);
}
std::vector<char> compress(gsl::span<const char> data, int level) {
  return zlib<char>::compress(data, level);
}
std::vector<uint8_t> uncompress(gsl::span<const uint8_t> data) {
  return zlib<uint8_t>::uncompress(data);
}
std::vector<int8_t> uncompress(gsl::span<const int8_t> data) {
  return zlib<int8_t>::uncompress(data);
}
std::vector<char> uncompress(gsl::span<const char> data) {
  return zlib<char>::uncompress(data);
}
std::unique_ptr<IStreamReader> compress(const IStreamReader& src,
                                        int compress_level) {
  auto filter = [](const IStreamReader& src, IStreamWriter& dst,
                   int compress_level) { compress(src, dst, compress_level); };
  return stream_filter(src, filter, compress_level);
}
std::unique_ptr<IStreamReader> uncompress(const IStreamReader& src) {
  auto filter = [](const IStreamReader& src, IStreamWriter& dst) {
    uncompress(src, dst);
  };
  return stream_filter(src, filter);
}
void compress(const IStreamReader& src, IStreamWriter& dst,
              int compress_level) {
  CHECK(compress_level > -1 && compress_level < 10)
      << "Invalid compression level. Must be between 0 and 9";
  const int CHUNK = 1024;
  z_stream strm;
  memset(&strm, 0, sizeof(strm));

  int ret = deflateInit(&strm, compress_level);
  CHECK(ret == Z_OK) << "Failed to initialize deflate.";
  char out[CHUNK];
  int flush;
  size_t bytes_read;
  do {
    auto in = src.read(CHUNK);
    bytes_read = in.has_value() ? in->size() : 0;
    // comparison of unsigned expression in '< 0' is always false
    // [-Werror=type-limits]
    //
    // if (bytes_read < 0) {
    //   deflateEnd(&strm);
    //   LOG(FATAL) << "Failed to read from source stream.";
    //   return;
    // }
    flush = (bytes_read == 0) ? Z_FINISH : Z_NO_FLUSH;
    strm.avail_in = (uInt)bytes_read;
    strm.next_in =
        (bytes_read == 0) ? nullptr : reinterpret_cast<Bytef*>(in->data());
    do {
      strm.avail_out = CHUNK;
      strm.next_out = reinterpret_cast<Bytef*>(out);

      ret = deflate(&strm, flush);
      if (ret == Z_STREAM_ERROR) {
        deflateEnd(&strm);
        LOG(FATAL) << "Stream error during compression.";
        return;
      }
      size_t have = CHUNK - strm.avail_out;
      if (dst.write(out, have) != have) {
        deflateEnd(&strm);
        LOG(FATAL) << "Failed to write to destination stream.";
        return;
      }
    } while (strm.avail_out == 0);
  } while (flush != Z_FINISH);

  if (ret != Z_STREAM_END) {
    deflateEnd(&strm);
    LOG(FATAL) << "Compression ended prematurely.";
  }
  deflateEnd(&strm);
}

void uncompress(const IStreamReader& src, IStreamWriter& dst) {
  const int CHUNK = 1024;

  z_stream strm;
  memset(&strm, 0, sizeof(strm));

  int ret = inflateInit(&strm);
  CHECK(ret == Z_OK) << "Failed to initialize deflate.";
  char out[CHUNK];
  size_t bytes_read;
  do {
    auto in = src.read(CHUNK);
    if (!in.has_value()) {
      LOG(FATAL) << "Failed to read from source stream.";
      inflateEnd(&strm);
      return;
    }
    bytes_read = in->size();
    strm.avail_in = (uInt)bytes_read;
    if (strm.avail_in == 0) {
      break;
    }
    strm.next_in = reinterpret_cast<Bytef*>(in->data());
    do {
      strm.avail_out = CHUNK;
      strm.next_out = reinterpret_cast<Bytef*>(out);

      ret = inflate(&strm, Z_NO_FLUSH);
      if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
        inflateEnd(&strm);
        LOG(FATAL) << "Error during decompression: " << ret;
        return;
      }
      size_t have = CHUNK - strm.avail_out;
      if (dst.write(out, have) != have) {
        inflateEnd(&strm);
        LOG(FATAL) << "Failed to write to destination stream.";
        return;
      }
    } while (strm.avail_out == 0);
  } while (ret != Z_STREAM_END);
  inflateEnd(&strm);
  if (ret != Z_STREAM_END) {
    LOG(FATAL) << "Incomplete decompression.";
  }
}

std::unique_ptr<IStreamReader>
context_cache_files_to_tar_stream(PassContext& context) {
  auto temp_file = std::make_shared<TempFile>();
  auto writer = temp_file->build_writer();
  context.cache_files_to_tar_file(*writer);
  auto reader_and_size = temp_file->build_reader();
  return std::move(reader_and_size.first);
}
std::string get_md5_of_buffer(const char* buffer, size_t size) {
  auto MD5_computer = MD5();
  MD5_computer.add(buffer, size);
  return MD5_computer.getHash();
}
std::string get_md5_of_file(const std::filesystem::path& path) {
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
} // namespace vaip_core
