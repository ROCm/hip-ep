/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "binary_journal.h"
#include "spool_writer.h"

#include <stdexcept>

namespace fs = std::filesystem;

namespace ktrace::collector {
namespace {
struct JournalHeader {
  std::uint32_t magic = 0x4A52544B; // KTRJ
  std::uint16_t version = 1;
  std::uint16_t reserved = 0;
  std::uint32_t record_size = sizeof(ipc::Record);
};
} // namespace

BinaryJournal::BinaryJournal(fs::path path)
    : path_(std::move(path)), buffer_(8 * 1024 * 1024) {
  if (_wfopen_s(&file_, path_.c_str(), L"wb") != 0 || !file_)
    throw std::runtime_error("cannot create binary journal: " + path_.string());
  setvbuf(file_, buffer_.data(), _IOFBF, buffer_.size());
  const JournalHeader header;
  if (fwrite(&header, sizeof(header), 1, file_) != 1) {
    Close();
    throw std::runtime_error("cannot write binary journal header");
  }
}

BinaryJournal::~BinaryJournal() { Close(); }

void BinaryJournal::Write(const ipc::Record &record) {
  if (fwrite(&record, sizeof(record), 1, file_) != 1)
    throw std::runtime_error("cannot append binary journal");
}

void BinaryJournal::Close() {
  if (file_) {
    fclose(file_);
    file_ = nullptr;
  }
}

void BinaryJournal::Replay(SpoolWriter *writer) {
  Close();
  FILE *input = nullptr;
  if (_wfopen_s(&input, path_.c_str(), L"rb") != 0 || !input)
    throw std::runtime_error("cannot reopen binary journal: " + path_.string());
  JournalHeader header{};
  if (fread(&header, sizeof(header), 1, input) != 1 ||
      header.magic != 0x4A52544B || header.version != 1 ||
      header.record_size != sizeof(ipc::Record)) {
    fclose(input);
    throw std::runtime_error("invalid binary journal header");
  }
  ipc::Record record{};
  while (fread(&record, sizeof(record), 1, input) == 1)
    writer->Write(record);
  if (!feof(input)) {
    fclose(input);
    throw std::runtime_error("failed reading binary journal");
  }
  fclose(input);
}

void BinaryJournal::Remove() {
  Close();
  std::error_code error;
  fs::remove(path_, error);
}
} // namespace ktrace::collector
