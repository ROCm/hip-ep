/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ktrace_ipc.h"

#include <new>

namespace ktrace::ipc {
namespace {
std::wstring Win32Error(DWORD code) {
  wchar_t *text = nullptr;
  const DWORD size = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<wchar_t *>(&text), 0, nullptr);
  std::wstring result =
      size && text ? std::wstring(text, size) : L"unknown error";
  if (text)
    LocalFree(text);
  while (!result.empty() && (result.back() == L'\r' || result.back() == L'\n'))
    result.pop_back();
  return result;
}
} // namespace

std::size_t MappingSize() { return sizeof(SharedSession); }

std::wstring MappingName(const std::wstring &session_name) {
  return L"Local\\ktrace_" + session_name;
}

bool Create(const std::wstring &session_name, Mapping *mapping,
            std::wstring *error) {
  mapping->name = MappingName(session_name);
  const std::uint64_t size = MappingSize();
  mapping->handle =
      CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                         static_cast<DWORD>(size >> 32),
                         static_cast<DWORD>(size), mapping->name.c_str());
  if (!mapping->handle) {
    *error = L"CreateFileMappingW failed: " + Win32Error(GetLastError());
    return false;
  }
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    *error = L"IPC session already exists: " + mapping->name;
    Close(mapping);
    return false;
  }
  mapping->session = static_cast<SharedSession *>(
      MapViewOfFile(mapping->handle, FILE_MAP_ALL_ACCESS, 0, 0, size));
  if (!mapping->session) {
    *error = L"MapViewOfFile failed: " + Win32Error(GetLastError());
    Close(mapping);
    return false;
  }

  new (mapping->session) SharedSession();
  mapping->session->magic = kMagic;
  mapping->session->abi_version = kAbiVersion;
  mapping->session->capacity = kRingCapacity;
  mapping->session->launcher_pid = GetCurrentProcessId();
  for (std::uint64_t i = 0; i < kRingCapacity; ++i)
    mapping->session->slots[i].sequence.store(i, std::memory_order_relaxed);
  return true;
}

bool Open(const std::wstring &session_name, Mapping *mapping,
          std::wstring *error) {
  mapping->name = MappingName(session_name);
  mapping->handle =
      OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, mapping->name.c_str());
  if (!mapping->handle) {
    *error = L"OpenFileMappingW failed: " + Win32Error(GetLastError());
    return false;
  }
  mapping->session = static_cast<SharedSession *>(
      MapViewOfFile(mapping->handle, FILE_MAP_ALL_ACCESS, 0, 0, MappingSize()));
  if (!mapping->session) {
    *error = L"MapViewOfFile failed: " + Win32Error(GetLastError());
    Close(mapping);
    return false;
  }
  if (mapping->session->magic != kMagic ||
      mapping->session->abi_version != kAbiVersion ||
      mapping->session->capacity != kRingCapacity) {
    *error = L"ktrace IPC ABI mismatch";
    Close(mapping);
    return false;
  }
  return true;
}

void Close(Mapping *mapping) {
  if (mapping->session) {
    UnmapViewOfFile(mapping->session);
    mapping->session = nullptr;
  }
  if (mapping->handle) {
    CloseHandle(mapping->handle);
    mapping->handle = nullptr;
  }
  mapping->name.clear();
}

bool Push(Mapping *mapping, const Record &record) {
  if (!mapping || !mapping->session)
    return false;
  SharedSession *session = mapping->session;
  std::uint64_t pos = session->enqueue_pos.load(std::memory_order_relaxed);
  Slot *slot = nullptr;
  for (;;) {
    slot = &session->slots[pos % kRingCapacity];
    const std::uint64_t sequence =
        slot->sequence.load(std::memory_order_acquire);
    const std::int64_t difference = static_cast<std::int64_t>(sequence - pos);
    if (difference == 0) {
      if (session->enqueue_pos.compare_exchange_weak(pos, pos + 1,
                                                     std::memory_order_relaxed))
        break;
    } else if (difference < 0) {
      session->dropped.fetch_add(1, std::memory_order_relaxed);
      return false;
    } else {
      pos = session->enqueue_pos.load(std::memory_order_relaxed);
    }
  }
  std::memcpy(&slot->record, &record, sizeof(record));
  slot->sequence.store(pos + 1, std::memory_order_release);
  session->pushed.fetch_add(1, std::memory_order_relaxed);
  const std::uint64_t occupancy =
      pos + 1 - session->dequeue_pos.load(std::memory_order_acquire);
  std::uint64_t peak = session->high_watermark.load(std::memory_order_relaxed);
  while (occupancy > peak && !session->high_watermark.compare_exchange_weak(
                                 peak, occupancy, std::memory_order_relaxed)) {
  }
  return true;
}

bool Pop(Mapping *mapping, Record *record) {
  if (!mapping || !mapping->session || !record)
    return false;
  SharedSession *session = mapping->session;
  std::uint64_t pos = session->dequeue_pos.load(std::memory_order_relaxed);
  Slot *slot = nullptr;
  for (;;) {
    slot = &session->slots[pos % kRingCapacity];
    const std::uint64_t sequence =
        slot->sequence.load(std::memory_order_acquire);
    const std::int64_t difference =
        static_cast<std::int64_t>(sequence - (pos + 1));
    if (difference == 0) {
      if (session->dequeue_pos.compare_exchange_weak(pos, pos + 1,
                                                     std::memory_order_relaxed))
        break;
    } else if (difference < 0) {
      return false;
    } else {
      pos = session->dequeue_pos.load(std::memory_order_relaxed);
    }
  }
  std::memcpy(record, &slot->record, sizeof(*record));
  slot->sequence.store(pos + kRingCapacity, std::memory_order_release);
  return true;
}

bool Empty(const Mapping &mapping) {
  if (!mapping.session)
    return true;
  return mapping.session->dequeue_pos.load(std::memory_order_acquire) ==
         mapping.session->enqueue_pos.load(std::memory_order_acquire);
}
} // namespace ktrace::ipc
