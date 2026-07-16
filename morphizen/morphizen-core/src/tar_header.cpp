/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./tar_header.hpp"
#include "morphizen/env_config.hpp"
#include <glog/logging.h>
DEF_ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= n)
namespace morphizen {
static constexpr size_t BLOCKSIZE = 512;
static constexpr int DEFAULT_MODE = 0644;
static constexpr int DEFAULT_UID = 30870;
static constexpr int DEFAULT_GID = 10585;
static constexpr char DEFAULT_UNAME[] = "chunywan";
static constexpr char DEFAULT_GNAME[] = "chunywan";

TarHeader::TarHeader(const std::string &name, size_t size)
    : name_(name), size_(size) {}
std::string TarHeader::to_string() const {
  std::ostringstream str;
  str << "TarHeader{"
      << "path=\"" << name_ << "\""                                     //
      << (real_path_ ? (std::string("->\"") + *real_path_ + "\"") : "") //
      << " block_begin_pos=" << block_begin_pos_                        //
      << ", block_end_pos=" << block_end_pos_                           //
      << ", data_begin_pos=" << data_begin_pos_                         //
      << ", data_end_pos=" << data_end_pos_                             //
      << ", size=" << size_
      << " bytes" //
      // << ", mode=" << mode_ << ", uid=" << uid_ << ", gid=" << gid_
      // << ", mtime=" << mtime_ << ", checksum=" << checksum_
      // << ", linkflag=" << linkflag_ << ", magic=\"" << magic_ << "\""
      // << ", uname=\"" << uname_ << "\""
      // << ", gname=\"" << gname_
      << "\""
      // Add other fields as needed
      << "}";
  return str.str();
}
bool TarHeader::is_long_name() const {
  // Check if the name exceeds the maximum length for a tar header
  // The maximum length for a tar header name is 100 bytes
  // in the POSIX tar format, so we can use that as a threshold.
  // we use `>=`, not `>`, because the name is not null-terminated.
  // clang-format off
  /*
00000c00: 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31  01234567890123456789012345678901
00000c20: 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33  23456789012345678901234567890123
00000c40: 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35  45678901234567890123456789012345
00000c60: 36 37 38 39 30 30 30 30 37 37 37 00 30 30 37 34 32 32 36 00 30 30 32 34 35 33 31 00 30 30 30 30  67890000777.0074226.0024531.0000
00000c80: 30 30 30 30 30 30 30 00 31 34 37 37 34 31 31 31 35 35 34 00 30 32 35 35 36 35 00 20 32 61 2e 74  0000000.14774111554.025565. 2a.t
00000ca0: 78 74 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  xt..............................
00000d00: 00 75 73 74 61 72 20 20 00 63 68 75 6e 79 77 61 6e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  .ustar  .chunywan...............
00000d20: 00 00 00 00 00 00 00 00 00 76 69 74 69 73 2d 61 69 2d 75 73 65 72 73 00 00 00 00 00 00 00 00 00  .........vitis-ai-users.........
*/
  // clang-format on
  return name_.size() >= 100;
}
void TarHeader::allocate_block() {
  if (is_long_name()) {
    // Allocate two blocks for long name
    auto num_of_blocks = (name_.size() + BLOCKSIZE - 1) / BLOCKSIZE;
    // BLOCK FOR HEADER 512 bytes
    // DATA FOR LONG NAME num_of_blocks * 512 bytes
    // BLOCK FOR DATA HEADER 512 bytes
    block_.resize((num_of_blocks + 2) *
                  BLOCKSIZE); // Allocate two more blocks for long name
  } else {
    block_.resize(BLOCKSIZE); // Allocate one block
  }
  std::fill(std::begin(block_), std::end(block_), static_cast<char>(0));
}
bool TarHeader::write_header(std::ostream &os) {
  construct_header();
  block_begin_pos_ = os.tellp();
  os.write(block_.data(), block_.size());
  block_end_pos_ = os.tellp();
  data_begin_pos_ = os.tellp();
  if (os.good() == false) {
    MY_LOG(1) << "Failed to write tar header";
  }
  data_end_pos_ = data_begin_pos_ + std::streampos(size());
  block_.clear();
  if (ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE) >= 1) {
    MY_LOG(1) << "write tar header: " << to_string()
              << " block_begin_pos_=" << block_begin_pos_
              << " streampos=" << os.tellp();
    os.flush();
  }
  return os.good();
}
HD_USTAR *TarHeader::get_header_for_long_name() {
  return reinterpret_cast<HD_USTAR *>(block_.data());
}
HD_USTAR *TarHeader::get_header_for_data() {
  auto ret = reinterpret_cast<HD_USTAR *>(&block_[block_.size() - BLOCKSIZE]);
  return ret;
}
void TarHeader::construct_long_name() {
  auto header = get_header_for_long_name();
  char link[] =
      "././@LongLink"; // size will be 14 (13 characters + null terminator)
  fill_name(header, link, sizeof(link));
  fill_mode(header, DEFAULT_MODE);
  fill_uid(header, DEFAULT_UID);
  fill_gid(header, DEFAULT_GID);
  fill_size(header, name_.size());
  fill_mtime(header);
  fill_linkflag(header, 'L'); // extension for long name
  fill_linkname(header, "", 0);
  fill_magic(header);
  fill_version(header);
  fill_uname(header, DEFAULT_UNAME, sizeof(DEFAULT_UNAME));
  fill_gname(header, DEFAULT_GNAME, sizeof(DEFAULT_GNAME));
  fill_devmajor(header, 0);
  fill_devminor(header, 0);
  fill_prefix(header, "", 0);
  fill_chksum(header);
  std::copy(name_.begin(), name_.end(), &block_[BLOCKSIZE]);
}
void TarHeader::construct_header() {
  allocate_block();
  now();
  if (is_long_name()) {
    construct_long_name();
  }
  auto header = get_header_for_data();
  // name probably is truncated
  fill_name(header, name_.data(), name_.size());
  fill_mode(header, DEFAULT_MODE);
  fill_uid(header, DEFAULT_UID);
  fill_gid(header, DEFAULT_GID);
  fill_size(header, size_);
  fill_mtime(header);
  fill_linkname(header, "", 0);
  if (real_path_) {
    if (real_path_->size() >= 100) {
      CHECK(false) << "Link name is too long: " << real_path_.value();
    }
    fill_linkflag(header, '2'); // symbol link
    fill_linkname(header, real_path_->data(), real_path_->size());
  } else {
    fill_linkflag(header, '0'); // normal file
    fill_linkname(header, "", 0);
  }
  fill_magic(header);
  fill_version(header);
  fill_uname(header, DEFAULT_UNAME, sizeof(DEFAULT_UNAME));
  fill_gname(header, DEFAULT_GNAME, sizeof(DEFAULT_GNAME));
  fill_devmajor(header, 0);
  fill_devminor(header, 0);
  fill_prefix(header, "", 0);
  fill_chksum(header);
  return;
}
void TarHeader::now() {
  auto now = std::chrono::system_clock::now();
  mtime_ =
      static_cast<unsigned long>(std::chrono::system_clock::to_time_t(now));
}
void TarHeader::fill_name(HD_USTAR *header, const char *name, size_t n) {
  // NOTE: name is not null-terminated
  auto len = std::min(n, sizeof(header->name));
  std::copy(name, name + len, &header->name[0]);
}
void TarHeader::fill_mode(HD_USTAR *header, int mode) {
  snprintf(header->mode, sizeof(header->mode), "%07o", mode);
  header->mode[7] = '\0'; // Ensure null termination
}
void TarHeader::fill_uid(HD_USTAR *header, int uid) {
  snprintf(header->uid, sizeof(header->uid), "%07o", uid);
  header->uid[7] = '\0'; // Ensure null termination
}
void TarHeader::fill_gid(HD_USTAR *header, int gid) {
  snprintf(header->gid, sizeof(header->gid), "%07o", gid);
  header->gid[7] = '\0'; // Ensure null termination
}
void TarHeader::fill_size(HD_USTAR *header, size_t size) {
  if (size <= 077777777777ULL) {
    // Standard POSIX ustar: 11-digit octal, supports up to 8 GB
    snprintf(header->size, sizeof(header->size), "%011llo",
             static_cast<unsigned long long>(size));
    header->size[11] = '\0';
  } else {
    // GNU tar base-256 extension: first byte 0x80, remaining 11 bytes
    // big-endian binary. Supports files up to ~4096 PB.
    header->size[0] = static_cast<char>(0x80u);
    size_t tmp = size;
    for (int i = 11; i >= 1; --i) {
      header->size[i] = static_cast<char>(tmp & 0xFFu);
      tmp >>= 8;
    }
  }
}
void TarHeader::fill_mtime(HD_USTAR *header) {
  snprintf(header->mtime, sizeof(header->mtime), "%011lo", mtime_);
  header->mtime[11] = '\0'; // Ensure null termination
}
void TarHeader::fill_chksum(HD_USTAR *header) {
  auto spaces = std::string(sizeof(header->chksum), ' ');
  std::copy(spaces.begin(), spaces.end(), header->chksum);
  unsigned int checksum_value = 0;
  const unsigned char *block = reinterpret_cast<const unsigned char *>(header);
  for (unsigned int i = 0; i < BLOCKSIZE; ++i) {
    checksum_value += block[i];
  }
  snprintf(header->chksum, sizeof(header->chksum), "%06o", checksum_value);
}
void TarHeader::fill_linkflag(HD_USTAR *header, char c) {
  header->typeflag = c;
}
void TarHeader::fill_linkname(HD_USTAR *header, const char *name, size_t n) {
  // NOTE: name is not null-terminated
  auto len = std::min(n, sizeof(header->linkname));
  std::copy(name, name + len, &header->linkname[0]);
}
void TarHeader::fill_magic(HD_USTAR *header) {

  // clang-format off
  /*
 00000d00: 00 75 73 74 61 72 20 20 00 63 68 75 6e 79 77 61 6e 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  .ustar  .chunywan...............
 */
  // clang-format on
  // TODO: check the magic
  const char magic[] = "ustar  ";
  std::copy(&magic[0], &magic[0] + sizeof(magic), &header->magic[0]);
}
void TarHeader::fill_version(HD_USTAR * /*header*/) {
  // const char version[] = "  ";
  // std::copy(&version[0], &version[0] + sizeof(version), &header->version[0]);
}
void TarHeader::fill_uname(HD_USTAR *header, const char *uname, size_t n) {
  // NOTE: uname is not null-terminated
  auto len = std::min(n, sizeof(header->uname));
  std::copy(uname, uname + len, &header->uname[0]);
}
void TarHeader::fill_gname(HD_USTAR *header, const char *gname, size_t n) {
  // NOTE: gname is not null-terminated
  auto len = std::min(n, sizeof(header->gname));
  std::copy(gname, gname + len, &header->gname[0]);
}
void TarHeader::fill_devmajor(HD_USTAR *header, int devmajor) {
  snprintf(header->devmajor, sizeof(header->devmajor), "%07o", devmajor);
  header->devmajor[7] = '\0'; // Ensure null termination
}
void TarHeader::fill_devminor(HD_USTAR *header, int devminor) {
  snprintf(header->devminor, sizeof(header->devminor), "%07o", devminor);
  header->devminor[7] = '\0'; // Ensure null termination
}
void TarHeader::fill_prefix(HD_USTAR *header, const char *prefix, size_t n) {
  // NOTE: prefix is not null-terminated
  auto len = std::min(n, sizeof(header->prefix));
  std::copy(prefix, prefix + len, &header->prefix[0]);
}

/**
 * @brief Calculates and verifies the TAR header checksum
 *
 * This function computes the checksum of a TAR header according to the POSIX
 * TAR specification and compares it with the stored checksum to verify the
 * integrity of the TAR header. It supports both POSIX and Sun tar checksum
 * variations.
 *
 * @param header Pointer to the TAR header
 * @return int Checksum validation result:
 *         - 1  : Checksum is valid (computed value matches the stored value)
 *         - -1 : Checksum is invalid (computed value does not match, possibly a
 * corrupted TAR file)
 *         - 0  : The TAR header might be empty (unlikely but possible)
 */
static int tar_checksum(HD_USTAR *header) {
  int unsigned_sum = 0; /* the POSIX one :-) */
  int signed_sum = 0;   /* the Sun one :-( */
  int recorded_sum;
  char *p = (char *)header;

  for (auto i = sizeof *header; i-- != 0;) {
    unsigned_sum += (unsigned char)*p;
    signed_sum += (signed char)(*p++);
  }

  if (unsigned_sum == 0)
    return 0;

  /* Adjust checksum to count the "chksum" field as blanks.  */

  for (auto i = sizeof header->chksum; i-- != 0;) {
    unsigned_sum -= (unsigned char)header->chksum[i];
    signed_sum -= (signed char)(header->chksum[i]);
  }
  unsigned_sum += (int)(' ' * sizeof header->chksum);
  signed_sum += (int)(' ' * sizeof header->chksum);

  auto parsed_sum = std::stoll(header->chksum, nullptr, 8);
  if (parsed_sum < 0)
    return -1;

  recorded_sum = (int)parsed_sum;

  if (unsigned_sum != recorded_sum && signed_sum != recorded_sum)
    return -1;

  return 1;
}

std::optional<TarHeader> TarHeader::read_header(std::istream &is) {
  auto ret = std::make_optional<TarHeader>("", 0);
  MY_LOG(1) << " trying to read a tar block at " << is.tellg();
  char block[BLOCKSIZE];
  is.read(block, BLOCKSIZE);
  if (is.gcount() != BLOCKSIZE) {
    if (is.eof()) {
      MY_LOG(1) << "End of tar file";
      is.clear();
      is.seekg(0, std::ios::end);
      return std::nullopt;
    } else {
      MY_LOG(1) << "Failed to read tar header";
      return std::nullopt;
    }
  }
  ret->block_begin_pos_ = is.tellg() - std::streampos(BLOCKSIZE);
  auto header = reinterpret_cast<HD_USTAR *>(block);
  auto check_sum = tar_checksum(header);
  if (check_sum == -1) {
    MY_LOG(1) << "Invalid tar header checksum";
    return std::nullopt;
  } else if (check_sum == 0) {
    MY_LOG(1) << "end of tar file at " << is.tellg();
    return std::nullopt;
  }
  if (header->typeflag == 'L') {
    // long name
    auto long_name_size = std::stoul(header->size, nullptr, 8);
    ret->name_.resize(long_name_size);
    is.read(block, BLOCKSIZE);
    if (is.gcount() != BLOCKSIZE) {
      MY_LOG(1) << "Failed to read tar long name";
      return std::nullopt;
    }
    std::copy(block, block + long_name_size, ret->name_.data());
    is.read(block, BLOCKSIZE);
    if (is.gcount() != BLOCKSIZE) {
      MY_LOG(1) << "Failed to read tar long name";
      return std::nullopt;
    }
    check_sum = tar_checksum(header);
    if (check_sum == -1) {
      MY_LOG(1) << "Invalid tar header checksum for long name: " << ret->name_;
      return std::nullopt;
    } else if (check_sum == 0) {
      MY_LOG(1) << "Empty tar header for long name: " << ret->name_;
      return std::nullopt;
    }
    header = reinterpret_cast<HD_USTAR *>(block);
  } else {
    // header->name might not be null-terminated, however there must be a zero
    // in the header, so trucate the name;
    ret->name_ = std::string(&header->name[0]);
    ret->name_.resize(std::min(sizeof(header->name), ret->name_.size()));
  }
  if (header->typeflag == '2') {
    // symbol link
    ret->real_path_ = std::string(&header->linkname[0]);
    ret->real_path_->resize(
        std::min(sizeof(header->linkname), ret->real_path_.value().size()));
  }
  ret->block_end_pos_ = is.tellg();
  if (static_cast<unsigned char>(header->size[0]) & 0x80u) {
    // GNU tar base-256 encoding: remaining 11 bytes are big-endian binary
    size_t val = 0;
    for (int i = 1; i <= 11; ++i) {
      val = (val << 8) | static_cast<unsigned char>(header->size[i]);
    }
    ret->size_ = val;
  } else {
    ret->size_ = std::stoull(header->size, nullptr, 8);
  }
  ret->data_begin_pos_ = is.tellg();
  ret->data_end_pos_ = ret->data_begin_pos_ + (std::streamoff)ret->size_;
  is.seekg(round_up_to_block_size(ret->data_end_pos_), std::ios::beg);
  MY_LOG(1) << " read a tar entry." << ret->to_string()
            << " streampos=" << is.tellg();
  return ret;
}
} // namespace morphizen
