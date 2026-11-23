/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "pftrace_writer.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <dbghelp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ktrace {
namespace {
using Bytes = std::string;

void Varint(Bytes &out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

void Key(Bytes &out, uint32_t field, uint32_t wire) {
  Varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

void UInt(Bytes &out, uint32_t field, uint64_t value) {
  Key(out, field, 0);
  Varint(out, value);
}

void Fixed64(Bytes &out, uint32_t field, uint64_t value) {
  Key(out, field, 1);
  out.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

void Double(Bytes &out, uint32_t field, double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  Fixed64(out, field, bits);
}

void Message(Bytes &out, uint32_t field, const Bytes &value) {
  Key(out, field, 2);
  Varint(out, value.size());
  out += value;
}

void Text(Bytes &out, uint32_t field, const std::string &value) {
  Message(out, field, value);
}

std::string JsonText(const rapidjson::Value &value) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  value.Accept(writer);
  return {buffer.GetString(), buffer.GetSize()};
}

void Annotation(Bytes &event, const std::string &name,
                const rapidjson::Value &value) {
  Bytes annotation;
  Text(annotation, 10, name);
  if (value.IsBool())
    UInt(annotation, 2, value.GetBool() ? 1 : 0);
  else if (value.IsUint64())
    UInt(annotation, 3, value.GetUint64());
  else if (value.IsInt64())
    UInt(annotation, 4, static_cast<uint64_t>(value.GetInt64()));
  else if (value.IsDouble())
    Double(annotation, 5, value.GetDouble());
  else if (value.IsString())
    Text(annotation, 6, value.GetString());
  else
    Text(annotation, 9, JsonText(value));
  Message(event, 4, annotation);
}

uint64_t TrackUuid(int64_t pid, int64_t tid) {
  return 0x8000000000000000ull ^ (static_cast<uint64_t>(pid) << 32) ^
         static_cast<uint64_t>(tid);
}

uint64_t ProcessUuid(int64_t pid) {
  return 0x7000000000000000ull ^ static_cast<uint64_t>(pid);
}

uint64_t CounterUuid(int64_t pid, const std::string &name) {
  uint64_t hash = 1469598103934665603ull;
  for (const unsigned char ch : name) {
    hash ^= ch;
    hash *= 1099511628211ull;
  }
  return 0x6000000000000000ull ^ (static_cast<uint64_t>(pid) << 32) ^ hash;
}

int64_t I64(const rapidjson::Value &object, const char *key,
            int64_t fallback = 0) {
  const auto found = object.FindMember(key);
  if (found == object.MemberEnd())
    return fallback;
  if (found->value.IsInt64())
    return found->value.GetInt64();
  if (found->value.IsUint64())
    return static_cast<int64_t>(found->value.GetUint64());
  return fallback;
}

double Number(const rapidjson::Value &object, const char *key,
              double fallback = 0) {
  const auto found = object.FindMember(key);
  return found != object.MemberEnd() && found->value.IsNumber()
             ? found->value.GetDouble()
             : fallback;
}

std::string StringValue(const rapidjson::Value &object, const char *key,
                        const char *fallback = "") {
  const auto found = object.FindMember(key);
  return found != object.MemberEnd() && found->value.IsString()
             ? found->value.GetString()
             : fallback;
}

std::string DosPath(std::string path) {
  if (path.rfind("\\SystemRoot\\", 0) == 0) {
    char root[MAX_PATH]{};
    const DWORD size = GetEnvironmentVariableA("SystemRoot", root, MAX_PATH);
    if (size && size < MAX_PATH)
      return std::string(root, size) + path.substr(11);
  }
  char device[MAX_PATH]{};
  for (char drive = 'A'; drive <= 'Z'; ++drive) {
    char name[3]{drive, ':', '\0'};
    if (!QueryDosDeviceA(name, device, MAX_PATH))
      continue;
    const size_t length = std::strlen(device);
    if (path.size() >= length && _strnicmp(path.c_str(), device, length) == 0)
      return std::string(name) + path.substr(length);
  }
  return path;
}

class SymbolResolver {
public:
  explicit SymbolResolver(const fs::path &images) {
    if (!images.empty()) {
      CsvReader reader(images);
      CsvRow row;
      while (reader.IsOpen() && reader.Next(row)) {
        Image image;
        image.pid = ToI64(row["Process_Id"]);
        image.base = ToU64(row["Image_Base"]);
        image.size = ToU64(row["Image_Size"]);
        image.path = DosPath(row["File_Name"]);
        by_pid_[image.pid].push_back(std::move(image));
      }
      for (auto &[pid, entries] : by_pid_) {
        (void)pid;
        std::sort(
            entries.begin(), entries.end(),
            [](const Image &a, const Image &b) { return a.base < b.base; });
      }
    }
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES |
                  SYMOPT_FAIL_CRITICAL_ERRORS);
    const char *search = std::getenv("_NT_SYMBOL_PATH");
    initialized_ =
        SymInitialize(
            GetCurrentProcess(),
            search
                ? search
                : "srv*C:\\Symbols*https://msdl.microsoft.com/download/symbols",
            FALSE) != FALSE;
  }

  ~SymbolResolver() {
    if (initialized_)
      SymCleanup(GetCurrentProcess());
  }

  std::string Resolve(int64_t pid, const std::string &text) {
    char *end = nullptr;
    const uint64_t address = std::strtoull(text.c_str(), &end, 0);
    if (end == text.c_str())
      return text;
    const auto key = std::make_pair(pid, address);
    const auto cached = cache_.find(key);
    if (cached != cache_.end())
      return cached->second;
    Image *image = Find(pid, address);
    if (!image)
      return cache_[key] = text;
    const std::string module = fs::path(image->path).filename().string();
    std::ostringstream fallback;
    fallback << module << "!+0x" << std::hex << address - image->base;
    std::string result = fallback.str();
    if (initialized_ && !image->loaded && fs::exists(image->path)) {
      SymLoadModuleEx(GetCurrentProcess(), nullptr, image->path.c_str(),
                      nullptr, image->base, static_cast<DWORD>(image->size),
                      nullptr, 0);
      image->loaded = true;
    }
    if (initialized_) {
      alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 2048]{};
      auto *symbol = reinterpret_cast<SYMBOL_INFO *>(storage);
      symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol->MaxNameLen = 2048;
      DWORD64 displacement = 0;
      if (SymFromAddr(GetCurrentProcess(), address, &displacement, symbol)) {
        std::ostringstream named;
        named << module << '!' << std::string(symbol->Name, symbol->NameLen);
        if (displacement)
          named << "+0x" << std::hex << displacement;
        result = named.str();
      }
    }
    return cache_[key] = result;
  }

private:
  struct Image {
    int64_t pid = 0;
    uint64_t base = 0;
    uint64_t size = 0;
    std::string path;
    bool loaded = false;
  };

  Image *Find(int64_t pid, uint64_t address) {
    for (const int64_t candidate : {pid, int64_t{0}}) {
      auto found = by_pid_.find(candidate);
      if (found == by_pid_.end())
        continue;
      auto &entries = found->second;
      auto it = std::upper_bound(entries.begin(), entries.end(), address,
                                 [](uint64_t value, const Image &image) {
                                   return value < image.base;
                                 });
      if (it == entries.begin())
        continue;
      --it;
      if (address < it->base + it->size)
        return &*it;
    }
    return nullptr;
  }

  bool initialized_ = false;
  std::map<int64_t, std::vector<Image>> by_pid_;
  std::map<std::pair<int64_t, uint64_t>, std::string> cache_;
};
} // namespace

struct PftraceWriter::Impl {
  Impl(PftraceWriter *writer, const fs::path &images)
      : writer(writer), symbols(images) {}

  void Packet(const Bytes &payload) {
    Bytes trace;
    Message(trace, 1, payload);
    writer->output_.write(trace.data(),
                          static_cast<std::streamsize>(trace.size()));
    ++writer->packet_count_;
  }

  void Descriptor(const Bytes &descriptor) {
    Bytes packet;
    UInt(packet, 10, 1);
    Message(packet, 60, descriptor);
    Packet(packet);
  }

  void ProcessDescriptor(int64_t pid) {
    Bytes process;
    UInt(process, 1, static_cast<uint64_t>(pid));
    const auto name = process_names.find(pid);
    if (name != process_names.end())
      Text(process, 6, name->second);
    UInt(process, 3, static_cast<uint64_t>(pid));
    Bytes descriptor;
    UInt(descriptor, 1, ProcessUuid(pid));
    if (name != process_names.end())
      Text(descriptor, 2, name->second);
    UInt(descriptor, 12, static_cast<uint64_t>(pid));
    Message(descriptor, 3, process);
    Descriptor(descriptor);
  }

  void TrackDescriptor(int64_t pid, int64_t tid) {
    const auto key = std::make_pair(pid, tid);
    if (!described.insert(key).second)
      return;
    const auto named = thread_names.find(key);
    const std::string name = named == thread_names.end()
                                 ? "thread " + std::to_string(tid)
                                 : named->second;
    Bytes thread;
    UInt(thread, 1, static_cast<uint64_t>(pid));
    UInt(thread, 2, static_cast<uint64_t>(tid));
    Text(thread, 5, name);
    UInt(thread, 3, static_cast<uint64_t>(tid));
    Bytes descriptor;
    UInt(descriptor, 1, TrackUuid(pid, tid));
    Text(descriptor, 2, name);
    UInt(descriptor, 12, static_cast<uint64_t>(tid));
    Message(descriptor, 4, thread);
    Descriptor(descriptor);
  }

  void CounterDescriptor(int64_t pid, const std::string &name) {
    const uint64_t uuid = CounterUuid(pid, name);
    if (!counter_described.insert(uuid).second)
      return;
    Bytes counter;
    UInt(counter, 3, 0); // custom unit, value matches JSON exactly
    Text(counter, 6, "MB");
    Bytes descriptor;
    UInt(descriptor, 1, uuid);
    Text(descriptor, 2, name);
    UInt(descriptor, 5, ProcessUuid(pid));
    Message(descriptor, 8, counter);
    Descriptor(descriptor);
  }

  void TrackEvent(uint64_t timestamp_ns, const Bytes &event) {
    Bytes packet;
    UInt(packet, 8, timestamp_ns);
    UInt(packet, 10, 1);
    Message(packet, 11, event);
    Packet(packet);
  }

  void Metadata(const rapidjson::Value &metadata) {
    constexpr uint64_t uuid = 0x5f4b545241434500ull;
    Bytes root;
    UInt(root, 1, 0);
    UInt(root, 19, 1);
    UInt(root, 20, 1);
    Descriptor(root);
    Bytes descriptor;
    UInt(descriptor, 1, uuid);
    Text(descriptor, 2, "ktrace metadata");
    Descriptor(descriptor);
    Bytes event;
    UInt(event, 9, 3);
    UInt(event, 11, uuid);
    Text(event, 23, "ktrace metadata");
    for (auto member = metadata.MemberBegin(); member != metadata.MemberEnd();
         ++member)
      Annotation(event, member->name.GetString(), member->value);
    TrackEvent(0, event);
  }

  void Event(const rapidjson::Value &value) {
    const std::string phase = StringValue(value, "ph");
    const int64_t pid = I64(value, "pid");
    const int64_t tid = I64(value, "tid");
    const std::string name = StringValue(value, "name");
    if (phase == "M") {
      const auto args = value.FindMember("args");
      if (name == "process_name" && args != value.MemberEnd() &&
          args->value.IsObject()) {
        process_names[pid] = StringValue(args->value, "name");
        ProcessDescriptor(pid);
      } else if (name == "thread_name" && args != value.MemberEnd() &&
                 args->value.IsObject()) {
        thread_names[{pid, tid}] = StringValue(args->value, "name");
        TrackDescriptor(pid, tid);
      }
      return;
    }

    const uint64_t timestamp_ns =
        static_cast<uint64_t>(std::llround(Number(value, "ts") * 1000.0));
    if (phase == "C") {
      CounterDescriptor(pid, name);
      Bytes event;
      UInt(event, 9, 4);
      UInt(event, 11, CounterUuid(pid, name));
      const auto args = value.FindMember("args");
      if (args != value.MemberEnd() && args->value.IsObject() &&
          args->value.MemberCount()) {
        const auto first = args->value.MemberBegin();
        if (first->value.IsNumber())
          Double(event, 44, first->value.GetDouble());
        for (auto member = args->value.MemberBegin();
             member != args->value.MemberEnd(); ++member)
          Annotation(event, member->name.GetString(), member->value);
      }
      TrackEvent(timestamp_ns, event);
      return;
    }

    TrackDescriptor(pid, tid);
    uint32_t type = 3;
    if (phase == "X")
      type = 1;
    Bytes event;
    UInt(event, 9, type);
    UInt(event, 11, TrackUuid(pid, tid));
    if (!name.empty())
      Text(event, 23, name);
    const std::string category = StringValue(value, "cat");
    if (!category.empty())
      Text(event, 22, category);
    const auto args = value.FindMember("args");
    if (args != value.MemberEnd() && args->value.IsObject()) {
      for (auto member = args->value.MemberBegin();
           member != args->value.MemberEnd(); ++member)
        Annotation(event, member->name.GetString(), member->value);
      const auto stack = args->value.FindMember("stack");
      if (stack != args->value.MemberEnd() && stack->value.IsString()) {
        std::vector<std::string> frames;
        std::string text = stack->value.GetString();
        size_t begin = 0;
        while (begin <= text.size()) {
          const size_t end = text.find(';', begin);
          frames.push_back(text.substr(begin, end - begin));
          if (end == std::string::npos)
            break;
          begin = end + 1;
        }
        Bytes callstack;
        for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame) {
          Bytes item;
          const int64_t process =
              args != value.MemberEnd() && args->value.IsObject()
                  ? I64(args->value, "process_id")
                  : 0;
          Text(item, 1, symbols.Resolve(process, *frame));
          Message(callstack, 1, item);
        }
        if (!frames.empty())
          Message(event, 55, callstack);
      }
    }
    const auto id = value.FindMember("id");
    if (id != value.MemberEnd() && id->value.IsUint64()) {
      const bool finish = phase == "f";
      Fixed64(event, finish ? 48 : 47, id->value.GetUint64());
    }
    TrackEvent(timestamp_ns, event);
    if (phase == "X") {
      Bytes end;
      UInt(end, 9, 2);
      UInt(end, 11, TrackUuid(pid, tid));
      const uint64_t duration_ns =
          static_cast<uint64_t>(std::llround(Number(value, "dur") * 1000.0));
      TrackEvent(timestamp_ns + duration_ns, end);
    }
  }

  PftraceWriter *writer;
  std::map<int64_t, std::string> process_names;
  std::map<std::pair<int64_t, int64_t>, std::string> thread_names;
  std::set<std::pair<int64_t, int64_t>> described;
  std::set<uint64_t> counter_described;
  SymbolResolver symbols;
};

PftraceWriter::PftraceWriter(const fs::path &output, const fs::path &images,
                             bool asynchronous)
    : asynchronous_(asynchronous), impl_(std::make_unique<Impl>(this, images)) {
  output_.open(output, std::ios::binary | std::ios::trunc);
  if (!output_)
    throw std::runtime_error(output.string() + ": cannot create pftrace");
  filling_.reserve(4096);
  if (asynchronous_)
    worker_ = std::thread(&PftraceWriter::Worker, this);
}

PftraceWriter::~PftraceWriter() {
  try {
    Close();
  } catch (...) {
  }
}

void PftraceWriter::Metadata(const std::string &json) {
  rapidjson::Document document;
  document.Parse(json.data(), json.size());
  if (document.HasParseError() || !document.IsObject())
    throw std::runtime_error("cannot encode pftrace metadata");
  impl_->Metadata(document);
}

void PftraceWriter::Event(const char *json, size_t size) {
  if (!asynchronous_) {
    Process(std::string(json, size));
    return;
  }
  filling_.emplace_back(json, size);
  if (filling_.size() >= 4096) {
    Batch batch;
    batch.swap(filling_);
    filling_.reserve(4096);
    Enqueue(std::move(batch));
  }
}

void PftraceWriter::Enqueue(Batch batch) {
  std::unique_lock<std::mutex> lock(mutex_);
  space_.wait(lock, [&] { return pending_.size() < 8 || error_; });
  if (error_)
    return;
  pending_.push_back(std::move(batch));
  ready_.notify_one();
}

void PftraceWriter::Worker() {
  for (;;) {
    Batch batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [&] { return stop_ || !pending_.empty(); });
      if (pending_.empty() && stop_)
        return;
      batch = std::move(pending_.front());
      pending_.pop_front();
      space_.notify_one();
    }
    try {
      for (const auto &json : batch)
        Process(json);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
      stop_ = true;
      pending_.clear();
      space_.notify_all();
      return;
    }
  }
}

void PftraceWriter::Process(const std::string &json) {
  rapidjson::Document document;
  document.Parse(json.data(), json.size());
  if (document.HasParseError() || !document.IsObject())
    throw std::runtime_error("cannot encode pftrace event");
  impl_->Event(document);
}

void PftraceWriter::Close() {
  if (closed_)
    return;
  closed_ = true;
  if (asynchronous_) {
    if (!filling_.empty())
      Enqueue(std::move(filling_));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }
  output_.flush();
  const bool write_failed = !output_;
  output_.close();
  if (error_)
    std::rethrow_exception(error_);
  if (write_failed)
    throw std::runtime_error("pftrace write failed");
}
} // namespace ktrace
