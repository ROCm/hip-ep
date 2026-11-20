#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""SQTT hardware token stream for gfx11 / gfx11.5 (header version 3).

A port of AMD's rocprof-trace-decoder token parser (rocm-systems,
projects/rocprof-trace-decoder: gfx10parser.h/gfx10token.cpp for the bit reader
and token table, gfx11token.* for the gfx11 overrides, rdna_sqtt.cpp for the
parse loop, trace_parser.hpp for PopulateDispatch). Only what dispatch, event,
occupancy and realtime records need is decoded; instruction-level tokens are
skipped.

The stream is LSB-first and every token length is a multiple of 4 bits, so the
decoder keeps a 64-bit window reloaded from the byte buffer at the consumed
position. Tokens are reordered by time through a small lookahead deque, exactly
as the reference does, because WAVE_READY is stamped one tick early.
"""

from __future__ import annotations

from collections import deque

MASK64 = (1 << 64) - 1
MASK32 = 0xFFFFFFFF
MASK48 = (1 << 48) - 1
MASK36 = (1 << 36) - 1

# Token types, in the reference's RdnaType enum order.
(
    UNKNOWN,
    EVENT,
    EVENT_SYNC,
    REG,
    REG_INIT,
    VALU_INST,
    IMM_ONE,
    IMMEDIATE,
    WAVE_READY,
    NEW_PC_GFX10,
    WAVE_END,
    WAVE_START,
    WAVE_START_EXT,
    WAVE_ALLOC,
    SHADER_DATA,
    SHADER_DATA_SHORT,
    UTIL_COUNTER_GFX10,
    TIME,
    NOP,
    MISC_GFX10,
    TIMESTAMP,
    HEADER,
    INST,
    UTIL_COUNTER_GFX11,
    EXEC_POPCOUNT1,
    EXEC_POPCOUNT3,
    NEW_PC_GFX12,
    LDS_CONFIG,
    MEDIUM_TIME,
) = range(29)

SKIPPED = frozenset(
    (
        INST,
        VALU_INST,
        IMM_ONE,
        IMMEDIATE,
        NEW_PC_GFX10,
        NEW_PC_GFX12,
        WAVE_READY,
        SHADER_DATA,
        SHADER_DATA_SHORT,
    )
)

# Decoder event types (trace_decoder_types.h).
EV_CS_PARTIAL_FLUSH = 1
EV_BOTTOM_OF_PIPE_TS = 2
EV_SAVE_CONTEXT = 3
EV_CACHE_FLUSH = 5
EV_PACKET_LOSS = 6
EV_TT_STALL_BEGIN = 9
EV_TT_STALL_END = 10
EV_TT_FLUSH = 11
EV_DIDT_STALL_BEGIN = 12
EV_DIDT_STALL_END = 13
EV_RGP_MARKER = 15  # reserved id; one USERDATA_2/3 marker dword (see markers.py)
EV_GC_RINSE = 16
EV_SPM_SAMPLE = 17

# RGP writes its SQTT markers two dwords per packet into these user-data registers.
USERDATA_2 = 0xC342
USERDATA_3 = 0xC343

_EVENT_IDS = {
    0x7: EV_CS_PARTIAL_FLUSH,
    0x6: EV_CACHE_FLUSH,
    0x4: EV_CACHE_FLUSH,
    0x16: EV_CACHE_FLUSH,
    0x14: EV_CACHE_FLUSH,
    0x28: EV_BOTTOM_OF_PIPE_TS,
    0x36: EV_TT_FLUSH,
}


def _lookup() -> list[tuple[int, int, int, int]]:
    """Low byte -> (type, length in bits, time field begin, time field end)."""
    tbl = [(UNKNOWN, 4, 0, 0)] * 256

    def add(typ, pattern, plen, length, tb, te):
        for i in range(pattern, 256, 1 << plen):
            tbl[i] = (typ, length, tb, te)

    add(INST, 0b010, 3, 20, 4, 7)
    add(VALU_INST, 0b011, 3, 12, 3, 6)
    add(IMM_ONE, 0b1101, 4, 12, 4, 7)
    add(IMMEDIATE, 0b00100, 5, 24, 5, 8)
    add(WAVE_READY, 0b10100, 5, 24, 5, 8)
    add(NEW_PC_GFX10, 0b0100001, 7, 64, 8, 11)
    add(WAVE_START, 0b01100, 5, 32, 5, 7)
    add(WAVE_START_EXT, 0b11100, 5, 48, 5, 7)
    add(WAVE_ALLOC, 0b00101, 5, 20, 5, 8)
    add(WAVE_END, 0b10101, 5, 20, 5, 8)
    add(SHADER_DATA, 0b00110, 5, 52, 5, 8)
    add(SHADER_DATA_SHORT, 0b10110, 5, 28, 5, 8)
    add(UTIL_COUNTER_GFX10, 0b0110001, 7, 64, 7, 9)
    add(TIME, 0b1000, 4, 8, 4, 8)
    add(NOP, 0b0000, 4, 4, 0, 0)
    add(MISC_GFX10, 0b1010001, 7, 24, 7, 16)
    add(EVENT, 0b01100001, 8, 24, 8, 11)
    add(EVENT_SYNC, 0b11100001, 8, 32, 8, 11)
    add(REG, 0b1001, 4, 64, 4, 7)
    add(REG_INIT, 0b1110001, 7, 64, 7, 10)
    add(TIMESTAMP, 0b0000001, 7, 64, 16, 64)
    add(HEADER, 0b0010001, 7, 64, 0, 0)
    # gfx11 overrides
    add(MISC_GFX10, 0b1010001, 7, 24, 7, 16)
    add(UTIL_COUNTER_GFX11, 0b0110001, 7, 48, 7, 9)
    add(TIMESTAMP, 0b0000001, 7, 48, 12, 48)
    return tbl


LOOKUP = _lookup()


class TokenStream:
    """Yields (time, raw 64-bit window, type) in time order."""

    __slots__ = (
        "advance",
        "buf",
        "clock",
        "cur",
        "fill",
        "lost",
        "nbits",
        "pending",
        "realtime",
    )

    def __init__(self, blob: bytes):
        self.buf = memoryview(bytes(blob) + b"\x00" * 16)
        self.nbits = len(blob) * 8
        self.fill = 0  # bit position of the window end; consumed = fill - 64
        self.advance = 64  # bits the current token occupies
        self.cur = 0
        self.clock = 0  # shader clock
        self.pending: deque = deque()
        self.realtime: list[tuple[int, int]] = []  # (shader clock, 100 MHz realtime)
        self.lost = False

    def _load(self) -> None:
        self.fill += self.advance
        pos = self.fill - 64
        i = pos >> 3
        self.cur = (int.from_bytes(self.buf[i : i + 9], "little") >> (pos & 7)) & MASK64

    def _padded(self) -> bool:
        return self.fill + 64 < self.nbits

    def _tail(self) -> bool:
        return self.fill < self.nbits or self.cur != 0

    def more(self) -> bool:
        return self._padded() or bool(self.pending) or self.cur != 0

    def _tick(self, info) -> int:
        """Advance the shader clock; return a realtime sample (0 if none)."""
        c = self.cur
        typ = info[0]
        if typ == TIMESTAMP:
            lost = (c >> 8) & 1
            is_rt = (c >> 9) & 1
            val = (c >> 12) & MASK36
            if lost and not is_rt:
                self.lost = True
            if not is_rt:
                self.clock += val
                return 0
            return val if not lost else 0
        tb, te = info[2], info[3]
        self.clock += ((c >> tb) & ((1 << (te - tb)) - 1)) + (4 if typ == TIME else 0)
        return 0

    def _merge_realtime(self, real: int) -> None:
        rl = self.realtime
        if rl:
            sh, rt = rl[-1]
            if rt >= real or sh >= self.clock:
                rl[-1] = ((sh + self.clock) // 2, (rt + real) // 2)
                return
        rl.append((self.clock, real))

    def next(self):
        pend = self.pending
        lookup = LOOKUP
        # Padded region: realtime samples are appended directly.
        while self._padded() or pend:
            if pend and (pend[0][0] < pend[-1][0] or not self._padded()):
                return pend.popleft()
            self._load()
            info = lookup[self.cur & 0xFF]
            typ = info[0]
            if typ == NOP:
                self.advance = 4
                continue
            self.advance = info[1]
            real = self._tick(info)
            if typ == TIMESTAMP or typ == TIME:
                if real:
                    self.realtime.append((self.clock, real))
                continue
            t = self.clock - 1 if typ == WAVE_READY else self.clock
            tok = (t, self.cur, typ)
            if pend and t < pend[-1][0]:
                pend.appendleft(tok)
            else:
                pend.append(tok)
        # Tail region: realtime samples are merged when out of order.
        while self._tail() or pend:
            if pend and (pend[0][0] < pend[-1][0] or not self._tail()):
                return pend.popleft()
            self._load()
            info = lookup[self.cur & 0xFF]
            typ = info[0]
            if typ == NOP:
                self.advance = 4
                continue
            self.advance = info[1]
            real = self._tick(info)
            if typ == TIMESTAMP or typ == TIME:
                if real:
                    self._merge_realtime(real)
                continue
            t = self.clock - 1 if typ == WAVE_READY else self.clock
            tok = (t, self.cur, typ)
            if pend and t < pend[-1][0]:
                pend.appendleft(tok)
            else:
                pend.append(tok)
        return (0, 0, TIMESTAMP)  # end sentinel


class ComputeRegs:
    """COMPUTE_* register shadow; snapshots a dispatch on REG_INIT."""

    __slots__ = ("nx", "ny", "nz", "pgm", "pkt", "rsrc1", "rsrc2", "rsrc3")

    def __init__(self):
        self.pgm = [[0] * 4, [0] * 4]
        self.pkt = [[0] * 4, [0] * 4]
        self.nx = self.ny = self.nz = 0
        self.rsrc1 = self.rsrc2 = self.rsrc3 = 0

    def write(self, addr: int, data: int, me: int, pipe: int) -> None:
        d = data & MASK32
        m = me & 1
        if addr == 0xC:  # COMPUTE_PGM_LO
            self.pgm[m][pipe] = (self.pgm[m][pipe] & ~MASK32) | d
        elif addr == 0xD:  # COMPUTE_PGM_HI
            self.pgm[m][pipe] = (self.pgm[m][pipe] & MASK32) | (d << 32)
        elif addr == 0x7:
            self.nx = d
        elif addr == 0x8:
            self.ny = d
        elif addr == 0x9:
            self.nz = d
        elif addr == 0x12:
            self.rsrc1 = d
        elif addr == 0x13:
            self.rsrc2 = d
        elif addr == 0x2D:
            self.rsrc3 = d
        elif addr == 0xE:  # COMPUTE_DISPATCH_PKT_LO
            self.pkt[m][pipe] = (self.pkt[m][pipe] & ~MASK32) | d
        elif addr == 0xF:
            self.pkt[m][pipe] = (self.pkt[m][pipe] & MASK32) | (d << 32)

    def pc(self, me: int, pipe: int) -> int:
        return (self.pgm[me & 1][pipe] << 8) & MASK48

    def dispatch(self, t: int, me: int, pipe: int, se: int) -> dict:
        r1, r2 = self.rsrc1, self.rsrc2
        flags = (
            ((r1 >> 10) & 1)
            | (((r1 >> 11) & 1) << 1)
            | (((r1 >> 14) & 1) << 2)
            | ((r2 & 1) << 3)
        )
        return {
            "se": se,
            "me": me,
            "pipe": pipe,
            "t": t,
            "pc": self.pc(me, pipe),
            "vgpr": (r1 & 0x3F) * 8 + 8,
            "sgpr": 128,
            "lds": ((r2 >> 15) & 0x1FF) * 512,
            "wg": self.nx * self.ny * self.nz,
            "flags": flags,
        }


def parse_engine(blob: bytes, se: int, broadcast: bool, launches: bool = False):
    """One shader engine's stream -> (dispatches, events, realtime, occupancy).

    Dispatches and events are broadcast into every engine's stream and are read
    from engine 0 only (``broadcast``); ``launches`` collects just the dispatch
    records so another engine can report the same launches on its own clock.
    Occupancy rows are [time, is_start, cu, simd, wave_id, me, pipe, pc] in
    delivery order, with synthetic pre-capture starts pushed to the front.
    """
    ts = TokenStream(blob)
    regs = ComputeRegs()
    running: dict[int, int] = {}
    saved: dict[int, int] = {}
    occ: deque = deque()
    disps: list[dict] = []
    events: list[tuple] = []
    want_disp = broadcast or launches

    while ts.more():
        t, c, typ = ts.next()
        if typ == MISC_GFX10:
            if broadcast:
                f = (c >> 16) & 0xFF
                if (f >> 3) & 1:
                    events.append((EV_SAVE_CONTEXT, t, 0, 0, 0, 0))
                if (f >> 4) & 1:
                    events.append((EV_TT_STALL_BEGIN, t, 0, 0, 0, 0))
                if (f >> 5) & 1:
                    events.append((EV_TT_STALL_END, t, 0, 0, 0, 0))
                if (f >> 6) & 1:
                    events.append((EV_DIDT_STALL_BEGIN, t, 0, 0, 0, 0))
                if (f >> 7) & 1:
                    events.append((EV_DIDT_STALL_END, t, 0, 0, 0, 0))
                if (f >> 1) & 1:
                    events.append((EV_GC_RINSE, t, 0, 0, 0, 0))
                if f & 1:
                    events.append((EV_SPM_SAMPLE, t, 0, 0, 0, 0))
        elif typ == HEADER:
            pass
        elif typ == WAVE_START or typ == WAVE_START_EXT:
            sa, simd, wgp = (c >> 7) & 1, (c >> 8) & 3, (c >> 10) & 7
            wid, pipe, me = (c >> 13) & 0x1F, (c >> 21) & 3, (c >> 23) & 1
            count = (c >> 25) & 0x7F
            cu = (sa << 7) + (wgp & 0x7F)
            loc = (cu << 7) | (simd << 5) | wid
            if 66 <= count < 90:
                # Context save / restore of an already-running wave.
                if (0b0000110011 >> (count - 66)) & 1:
                    if loc in running:
                        occ.append([t, 0, cu, simd, wid, me, pipe, running[loc]])
                        saved[loc] = running[loc]
                elif (0b1111001100 >> (count - 66)) & 1:
                    addr = saved.pop(loc, 0)
                    running[loc] = addr
                    occ.append([t, 1, cu, simd, wid, me, pipe, addr])
                continue
            if ts.lost and loc in running:
                occ.append([t - 1, 0, cu, simd, wid, 0, 0, 0])
            pc = regs.pc(me, pipe)
            running[loc] = pc
            occ.append([t, 1, cu, simd, wid, me, pipe, pc])
        elif typ == WAVE_END:
            sa, simd, wgp, wid = (
                (c >> 8) & 1,
                (c >> 9) & 3,
                (c >> 11) & 7,
                (c >> 15) & 0x1F,
            )
            cu = (sa << 7) + (wgp & 0x7F)
            loc = (cu << 7) | (simd << 5) | wid
            if loc in running:
                pc = running.pop(loc)
            else:
                pc = 0
                occ.appendleft([0, 1, cu, simd, wid, 0, 0, 0])
            occ.append([t, 0, cu, simd, wid, 0, 0, pc])
        elif typ == REG:
            addr, data = (c >> 16) & 0xFFFF, (c >> 32) & MASK32
            me, pipe = (c >> 9) & 3, (c >> 7) & 3
            if (c >> 15) & 1:
                regs.write(addr, data, me, pipe)
            elif broadcast and (addr == USERDATA_2 or addr == USERDATA_3):
                events.append(
                    (EV_RGP_MARKER, t, me, pipe, data, 1 if addr == USERDATA_3 else 0)
                )
        elif typ == REG_INIT:
            if want_disp and ((c >> 18) & 3) == 2 and ((c >> 20) & 1) == 1:
                disps.append(regs.dispatch(t, (c >> 16) & 3, (c >> 14) & 3, se))
        elif typ == EVENT:
            ev = _EVENT_IDS.get((c >> 18) & 0x3F)
            if ev is not None and broadcast:
                flags = 0x1 | (0x2 if (c >> 11) & 1 else 0)
                events.append((ev, t, ((c >> 16) & 3) & 1, (c >> 14) & 3, 0, flags))
        elif typ in SKIPPED:
            pass
        else:
            if typ == TIMESTAMP and c == 0:
                break
            if ts.lost and broadcast:
                events.append((EV_PACKET_LOSS, t, 0, 0, 0, 0))
                ts.lost = False
    return disps, events, ts.realtime, list(occ)
