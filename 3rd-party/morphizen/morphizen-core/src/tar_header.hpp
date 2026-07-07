/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <array>
#include <iostream>
#include <memory>
#include <morphizen/export.h>
#include <optional>
#include <streambuf>
#include <vector>

// TAR header structure definitions (from POSIX ustar format)
#define TNMSZ 100
#define CHK_LEN 8
#define TPFSZ 155

#if defined(_WIN32)
#define PACKED(x) __declspec(align(x))
#else
#define PACKED(x) __attribute__((packed, aligned(x)))
#endif

typedef struct PACKED(1) {
  char name[TNMSZ];     // name of entry
  char mode[8];         // mode
  char uid[8];          // uid
  char gid[8];          // gid
  char size[12];        // size
  char mtime[12];       // modification time
  char chksum[CHK_LEN]; // checksum
  char typeflag;        // type of file
  char linkname[TNMSZ]; // linked to name
  char magic[6];        // ustar magic "ustar\0"
  char version[2];      // ustar version "00"
  char uname[32];       // user name
  char gname[32];       // group name
  char devmajor[8];     // major device number
  char devminor[8];     // minor device number
  char prefix[TPFSZ];   // prefix for long names
} HD_USTAR;

namespace morphizen {
template <typename T> static T round_up_to_block_size(T size) {
  T x511 = 511;
  return (size + x511) & ~x511; // round up to the nearest 512 bytes
}
class TarHeader {
public:
  TarHeader(const std::string &name, size_t size);
  ~TarHeader() = default;

public:
  void set_name(const std::string &name) { name_ = name; }
  void set_size(size_t size) { size_ = size; }
  void set_link_name(const std::string &real_path) { real_path_ = real_path; }
  const std::string &path() const { return name_; }
  const std::optional<std::string> real_path() const { return real_path_; }
  bool is_symlink() const {
    return real_path_.has_value() && !real_path_->empty();
  }
  size_t size() const { return size_; }
  std::streampos data_begin_pos() const { return data_begin_pos_; }
  std::streampos data_end_pos() const { return data_end_pos_; }
  std::streampos block_begin_pos() const { return block_begin_pos_; }
  std::streampos block_end_pos() const { return block_end_pos_; }
  bool write_header(std::ostream &os);
  static std::optional<TarHeader> read_header(std::istream &is);

  // for logging;
  std::string to_string() const;

private:
  static constexpr size_t BLOCKSIZE = 512;
  void allocate_block();
  bool is_long_name() const;
  void construct_header();
  void construct_long_name();
  void now();
  HD_USTAR *get_header_for_long_name();
  HD_USTAR *get_header_for_data();
  void fill_name(HD_USTAR *header, const char *name, size_t n);
  void fill_mode(HD_USTAR *header, int mode);
  void fill_uid(HD_USTAR *header, int uid);
  void fill_gid(HD_USTAR *header, int gid);
  void fill_size(HD_USTAR *header, size_t size);
  void fill_mtime(HD_USTAR *header);
  void fill_chksum(HD_USTAR *header);
  void fill_linkflag(HD_USTAR *header, char linkflag);
  void fill_linkname(HD_USTAR *header, const char *linkname, size_t n);
  void fill_magic(HD_USTAR *header);
  void fill_version(HD_USTAR *header);
  void fill_uname(HD_USTAR *header, const char *uname, size_t n);
  void fill_gname(HD_USTAR *header, const char *gname, size_t n);
  void fill_devmajor(HD_USTAR *header, int devmajor);
  void fill_devminor(HD_USTAR *header, int devminor);
  void fill_prefix(HD_USTAR *header, const char *prefix, size_t n);

private:
  std::string name_ = "";
  size_t size_ = 0u;
  std::optional<std::string> real_path_ = std::nullopt;
  unsigned long mtime_ = 0;
  std::vector<char> block_;
  std::streampos block_begin_pos_ = 0;
  std::streampos block_end_pos_ = 0;
  std::streampos data_begin_pos_ = 0;
  std::streampos data_end_pos_ = 0;
};
} // namespace morphizen
