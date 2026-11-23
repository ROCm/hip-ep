#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Write a native Perfetto TrackEvent protobuf while retaining Chrome JSON."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

from rocprofv3_reader import ensure_capture
from symbol_resolver import SymbolResolver
from to_chrome_trace import main as chrome_main


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _uint(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _fixed64(field: int, value: int) -> bytes:
    return _key(field, 1) + int(value).to_bytes(8, "little", signed=False)


def _bytes(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _text(field: int, value: str) -> bytes:
    return _bytes(field, value.encode("utf-8"))


def _packet(payload: bytes) -> bytes:
    return _bytes(1, payload)


def _descriptor(uuid: int, name: str, pid: int, tid: int) -> bytes:
    thread = _uint(1, pid) + _uint(2, tid) + _text(5, name)
    descriptor = _uint(1, uuid) + _text(2, name) + _uint(12, tid) + _bytes(4, thread)
    packet = _uint(10, 1) + _bytes(60, descriptor)
    return _packet(packet)


def _process_descriptor(pid: int, name: str) -> bytes:
    uuid = 0x7000000000000000 ^ pid
    process = _uint(1, pid) + _text(6, name) + _uint(3, pid)
    descriptor = _uint(1, uuid) + _text(2, name) + _uint(12, pid) + _bytes(3, process)
    return _packet(_uint(10, 1) + _bytes(60, descriptor))


def _annotation(name: str, value) -> bytes:
    annotation = _text(10, name)
    if isinstance(value, bool):
        annotation += _uint(2, int(value))
    elif isinstance(value, int) and value >= 0:
        annotation += _uint(3, value)
    elif isinstance(value, int):
        annotation += _uint(4, value)
    elif isinstance(value, float):
        annotation += _key(5, 1) + struct.pack("<d", value)
    elif isinstance(value, str):
        annotation += _text(6, value)
    else:
        annotation += _text(
            9, json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
    return _bytes(4, annotation)


def _event(
    timestamp_ns: int,
    track_uuid: int,
    event_type: int,
    name: str = "",
    category: str = "",
    flow: int | None = None,
    terminating: bool = False,
    callstack: list[str] | None = None,
    args: dict | None = None,
) -> bytes:
    track_event = _uint(9, event_type) + _uint(11, track_uuid)
    if name:
        track_event += _text(23, name)
    if category:
        track_event += _text(22, category)
    if flow is not None:
        track_event += _fixed64(48 if terminating else 47, flow)
    if callstack:
        inline = b"".join(_bytes(1, _text(1, frame)) for frame in callstack)
        track_event += _bytes(55, inline)
    for key, value in (args or {}).items():
        track_event += _annotation(str(key), value)
    packet = _uint(8, timestamp_ns) + _uint(10, 1) + _bytes(11, track_event)
    return _packet(packet)


def _counter_descriptor(uuid: int, name: str) -> bytes:
    counter = _text(6, "MB")
    descriptor = _uint(1, uuid) + _text(2, name) + _bytes(8, counter)
    return _packet(_uint(10, 1) + _bytes(60, descriptor))


def _track_uuid(pid: int, tid: int) -> int:
    return ((pid & 0xFFFFFFFF) << 32) | (tid & 0xFFFFFFFF) | (1 << 63)


def _counter_uuid(pid: int, name: str) -> int:
    value = 1469598103934665603
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value ^ ((pid & 0xFFFFFFFF) << 32) ^ (6 << 60)


class PftraceSink:
    def __init__(self, output_path: Path, images_path: Path | None = None):
        self.output = output_path.open("wb")
        self.symbols = SymbolResolver(images_path)
        self.names: dict[tuple[int, int], str] = {}
        self.processes: set[int] = set()
        self.described: set[tuple[int, int]] = set()
        self.counters: set[int] = set()
        self.packet_count = 0

    def _write(self, packet: bytes) -> None:
        self.output.write(packet)
        self.packet_count += 1

    def _ensure_track(self, pid: int, tid: int) -> int:
        key = pid, tid
        uuid = _track_uuid(pid, tid)
        if key not in self.described:
            self.described.add(key)
            self._write(
                _descriptor(
                    uuid, self.names.get(key, f"pid {pid} / tid {tid}"), pid, tid
                )
            )
        return uuid

    def metadata(self, metadata: dict) -> None:
        uuid = 0x5F4B545241434500
        root = _uint(1, 0) + _uint(19, 1) + _uint(20, 1)
        self._write(_packet(_uint(10, 1) + _bytes(60, root)))
        self._write(
            _packet(
                _uint(10, 1) + _bytes(60, _uint(1, uuid) + _text(2, "ktrace metadata"))
            )
        )
        self._write(_event(0, uuid, 3, "ktrace metadata", "metadata", args=metadata))

    def event(self, event: dict) -> None:
        phase = event.get("ph")
        pid, tid = int(event.get("pid", 0)), int(event.get("tid", 0))
        if phase == "M":
            if event.get("name") == "process_name":
                process_name = str(event.get("args", {}).get("name", f"pid {pid}"))
                if pid not in self.processes:
                    self.processes.add(pid)
                    self._write(_process_descriptor(pid, process_name))
            elif event.get("name") == "thread_name":
                self.names[(pid, tid)] = str(
                    event.get("args", {}).get("name", f"thread {tid}")
                )
                self._ensure_track(pid, tid)
            return
        timestamp = int(round(float(event.get("ts", 0)) * 1000.0))
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        args = dict(event.get("args", {}))
        if phase == "C":
            uuid = _counter_uuid(pid, name)
            if uuid not in self.counters:
                self.counters.add(uuid)
                self._write(_counter_descriptor(uuid, name))
            value = next(
                (value for value in args.values() if isinstance(value, (int, float))), 0
            )
            track_event = _uint(9, 4) + _uint(11, uuid)
            track_event += _key(44, 1) + struct.pack("<d", float(value))
            for key, item in args.items():
                track_event += _annotation(str(key), item)
            self._write(
                _packet(_uint(8, timestamp) + _uint(10, 1) + _bytes(11, track_event))
            )
            return
        uuid = self._ensure_track(pid, tid)
        if phase == "X":
            self._write(_event(timestamp, uuid, 1, name, category, args=args))
            duration = int(round(float(event.get("dur", 0)) * 1000.0))
            self._write(_event(timestamp + duration, uuid, 2))
        elif phase == "i":
            stack_text = str(args.get("stack", ""))
            process_id = int(args.get("process_id", 0))
            callstack = list(
                reversed(
                    [
                        self.symbols.resolve(process_id, frame)
                        for frame in stack_text.split(";")
                        if frame
                    ]
                )
            )
            self._write(
                _event(
                    timestamp, uuid, 3, name, category, callstack=callstack, args=args
                )
            )
        elif phase in ("s", "f"):
            self._write(
                _event(
                    timestamp,
                    uuid,
                    3,
                    name,
                    category,
                    int(event.get("id", 0)),
                    phase == "f",
                    args=args,
                )
            )

    def close(self) -> int:
        self.output.close()
        return self.packet_count


def convert_chrome(
    chrome_path: Path, output_path: Path, images_path: Path | None = None
) -> int:
    document = json.loads(chrome_path.read_text(encoding="utf-8"))
    sink = PftraceSink(output_path, images_path)
    sink.metadata(document.get("otherData", {}))
    for event in document.get("traceEvents", []):
        sink.event(event)
    return sink.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ktrace capture -> native Perfetto TrackEvent protobuf"
    )
    parser.add_argument("capture")
    parser.add_argument("out_proto", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--flows", action="store_true")
    args = parser.parse_args(argv)

    capture = ensure_capture(args.capture)
    output = Path(args.out_proto or capture.auto_output("_ktrace_perfetto.pftrace"))
    images = None if capture.is_legacy else capture.domain_path("cpu_images")
    sink = PftraceSink(output, images)
    with tempfile.TemporaryDirectory(prefix="ktrace-perfetto-") as temp:
        placeholder = Path(temp) / "suppressed.json"
        chrome_args = [args.capture, str(placeholder)]
        if args.all:
            chrome_args.append("--all")
        if args.flows:
            chrome_args.append("--flows")
        chrome_main(chrome_args, event_sink=sink, suppress_json=True)
    count = sink.close()
    print(f"-> {output} ({output.stat().st_size / 1e6:.1f} MB, {count:,} packets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
