#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Resolve captured Windows instruction addresses with DbgHelp."""

from __future__ import annotations

import csv
import ctypes
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Image:
    process: int
    base: int
    size: int
    path: str


class _SymbolInfo(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", ctypes.c_ulong),
        ("TypeIndex", ctypes.c_ulong),
        ("Reserved", ctypes.c_ulonglong * 2),
        ("Index", ctypes.c_ulong),
        ("Size", ctypes.c_ulong),
        ("ModBase", ctypes.c_ulonglong),
        ("Flags", ctypes.c_ulong),
        ("Value", ctypes.c_ulonglong),
        ("Address", ctypes.c_ulonglong),
        ("Register", ctypes.c_ulong),
        ("Scope", ctypes.c_ulong),
        ("Tag", ctypes.c_ulong),
        ("NameLen", ctypes.c_ulong),
        ("MaxNameLen", ctypes.c_ulong),
        ("Name", ctypes.c_char * 1),
    ]


def _device_paths() -> list[tuple[str, str]]:
    if os.name != "nt":
        return []
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    result = []
    buffer = ctypes.create_unicode_buffer(32768)
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:"
        if kernel32.QueryDosDeviceW(drive, buffer, len(buffer)):
            result.append((buffer.value, drive))
    return sorted(result, key=lambda item: len(item[0]), reverse=True)


def _dos_path(path: str, devices: list[tuple[str, str]]) -> str:
    low = path.lower()
    if low.startswith("\\systemroot\\"):
        return os.environ.get("SystemRoot", r"C:\Windows") + path[11:]
    for device, drive in devices:
        if low.startswith(device.lower()):
            return drive + path[len(device) :]
    return path


class SymbolResolver:
    def __init__(self, images_csv: Path | None):
        self.images: dict[int, list[Image]] = {}
        self.bases: dict[int, list[int]] = {}
        self.cache: dict[tuple[int, int], str] = {}
        self.loaded: set[tuple[int, int]] = set()
        self.handles: dict[int, int] = {}
        self.devices = _device_paths()
        self.dbghelp = None
        if images_csv and images_csv.exists():
            with images_csv.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    image = Image(
                        int(row["Process_Id"]),
                        int(row["Image_Base"], 0),
                        int(row["Image_Size"], 0),
                        _dos_path(row["File_Name"], self.devices),
                    )
                    self.images.setdefault(image.process, []).append(image)
            for pid, images in self.images.items():
                images.sort(key=lambda image: image.base)
                self.bases[pid] = [image.base for image in images]
        if os.name == "nt":
            self._initialize_dbghelp()

    def _initialize_dbghelp(self) -> None:
        self.dbghelp = ctypes.WinDLL("dbghelp", use_last_error=True)
        self.dbghelp.SymSetOptions(
            0x00000004  # SYMOPT_DEFERRED_LOADS
            | 0x00000002  # SYMOPT_UNDNAME
            | 0x00000010  # SYMOPT_LOAD_LINES
            | 0x00000200
        )  # SYMOPT_FAIL_CRITICAL_ERRORS
        self.dbghelp.SymInitializeW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_bool,
        ]
        self.dbghelp.SymLoadModuleExW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_ulonglong,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self.dbghelp.SymLoadModuleExW.restype = ctypes.c_ulonglong
        self.dbghelp.SymFromAddr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_void_p,
        ]

    def _handle(self, pid: int) -> int:
        if pid in self.handles:
            return self.handles[pid]
        handle = 0x4B5400000000 | (pid & 0xFFFFFFFF)
        search = os.environ.get(
            "_NT_SYMBOL_PATH",
            r"SRV*C:\Symbols*https://msdl.microsoft.com/download/symbols",
        )
        if not self.dbghelp.SymInitializeW(handle, search, False):
            return 0
        self.handles[pid] = handle
        return handle

    def _image(self, pid: int, address: int) -> Image | None:
        for candidate_pid in (pid, 0):
            bases = self.bases.get(candidate_pid)
            if not bases:
                continue
            index = bisect_right(bases, address) - 1
            if index < 0:
                continue
            image = self.images[candidate_pid][index]
            if address < image.base + image.size:
                return image
        return None

    def resolve(self, pid: int, address_text: str) -> str:
        try:
            address = int(address_text, 0)
        except ValueError:
            return address_text
        key = pid, address
        if key in self.cache:
            return self.cache[key]
        image = self._image(pid, address)
        if image is None:
            result = address_text
        else:
            module = Path(image.path).name or image.path
            result = f"{module}!+0x{address - image.base:x}"
            handle = self._handle(pid) if self.dbghelp else 0
            load_key = pid, image.base
            if handle and load_key not in self.loaded and Path(image.path).exists():
                self.dbghelp.SymLoadModuleExW(
                    handle, None, image.path, None, image.base, image.size, None, 0
                )
                self.loaded.add(load_key)
            if handle:
                storage = ctypes.create_string_buffer(ctypes.sizeof(_SymbolInfo) + 2048)
                symbol = ctypes.cast(storage, ctypes.POINTER(_SymbolInfo))
                symbol.contents.SizeOfStruct = ctypes.sizeof(_SymbolInfo)
                symbol.contents.MaxNameLen = 2048
                displacement = ctypes.c_ulonglong()
                if self.dbghelp.SymFromAddr(
                    handle, address, ctypes.byref(displacement), symbol
                ):
                    start = ctypes.addressof(storage) + _SymbolInfo.Name.offset
                    name = ctypes.string_at(start, symbol.contents.NameLen).decode(
                        "utf-8", errors="replace"
                    )
                    result = f"{module}!{name}"
                    if displacement.value:
                        result += f"+0x{displacement.value:x}"
        self.cache[key] = result
        return result
