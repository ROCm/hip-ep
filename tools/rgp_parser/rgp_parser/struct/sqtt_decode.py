#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Pure-Python SQTT hardware-token decoder for gfx11 (RDNA3 / Strix Halo).

This is a faithful port of AMD's ``rocprof-trace-decoder`` (the C++
``rocprof_trace_decoder_parse_data`` token parser), plus the post-processing this
pipeline needs (wave-to-dispatch attribution + wall-clock mapping), so the whole
pipeline runs in Python.

Scope: it produces the record set the rest of the pipeline consumes -- DISPATCH,
EVENT, OCCUPANCY (wave start/end) and REALTIME -- for gfx11
(``header.version == 3``). The full instruction-level wave stitching
(INST/VALU/NEW_PC/SHADER_DATA -> per-wave instruction timelines) is intentionally
skipped: it is not needed for the dispatch / occupancy / event trace.

Source of truth (all in-tree under rocm-systems/projects/rocprof-trace-decoder):
  * source/gfx10/gfx10parser.h, gfx10token.cpp   -- bit reader + token table
  * source/gfx11/gfx11token.{h,cpp}, gfx11parser.h -- gfx11 overrides
  * source/gfx10/rdna_sqtt.cpp                    -- the parse loop
  * source/trace_parser.hpp                       -- CSRegisterHandler.PopulateDispatch
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1
MASK32 = 0xFFFFFFFF
BITMASK48 = (1 << 48) - 1
INF = float("inf")

# --- RdnaType (token_types.h enum order) ------------------------------------
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

# tokens the parse loop explicitly cases but that we skip (no dispatch/occ/event
# effect); everything not handled and not here falls to the packet-loss default.
_SKIP = frozenset(
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

# --- decoder event types (trace_decoder_types.h) ----------------------------
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
EV_GC_RINSE = 16
EV_SPM_SAMPLE = 17
EV_RGP_MARKER = (
    15  # RESERVED id; carries one USERDATA_2/3 marker dword (build._parse_markers)
)

# RGP streams its SQTT markers 2 dwords/packet into the SQ thread-trace user-data
# registers (PAL palubercapturemgr -> CmdInsertRgpTraceMarker). Reassembling these
# in capture order reproduces the Mesa ac_sqtt.h rgp_sqtt_marker_* stream, which
# yields per-dispatch grid dims (exact wavefronts) and exact barrier cache/stall bits.
USERDATA_2 = 0xC342  # even marker dwords
USERDATA_3 = 0xC343  # odd  marker dwords (barrier_end cache bits live here)

# hardware EVENT-token ids (sqtt_event_type_t) -> decoder event type
_EVENT_ID_MAP = {
    0x7: EV_CS_PARTIAL_FLUSH,
    0x6: EV_CACHE_FLUSH,
    0x4: EV_CACHE_FLUSH,
    0x16: EV_CACHE_FLUSH,
    0x14: EV_CACHE_FLUSH,
    0x28: EV_BOTTOM_OF_PIPE_TS,
    0x36: EV_TT_FLUSH,
}


def _build_lookup():
    """256-entry (type, length, time_begin, time_end) table for gfx11."""
    tbl = [(UNKNOWN, 4, 0, 0)] * 256

    def add(typ, pattern, plen, length, tb, te):
        info = (typ, length, tb, te)
        step = 1 << plen
        for i in range(pattern, 256, step):
            tbl[i] = info

    # gfx10 base encodings
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


_LOOKUP = _build_lookup()
_TS36 = (1 << 36) - 1


class TokenGen:
    """gfx11 token generator: bit reader + lookahead time-reorder (rdna_sqtt).

    A single 64-bit window ``current`` slides over the LSB-first bit stream.
    Because every gfx1x token length is a multiple of 4 bits, the C++ nibble/byte
    shifting is exactly equivalent to reloading ``current`` from the byte buffer
    at the running consumed-bit position -- which is what ``_reload`` does.
    """

    __slots__ = (
        "mv",
        "nbits",
        "bit_ptr",
        "bits_toread",
        "current",
        "globaltime",
        "lookahead",
        "realtime",
        "packetlost",
        "lookup",
    )

    def __init__(self, blob):
        self.mv = memoryview(bytes(blob) + b"\x00" * 16)
        self.nbits = len(blob) * 8
        self.bit_ptr = 0  # fill pointer; consumed pos == bit_ptr - 64
        self.bits_toread = 64
        self.current = 0
        self.globaltime = 0
        self.lookahead = __import__("collections").deque()
        self.realtime = []  # (shader_clock, realtime_clock)
        self.packetlost = False
        self.lookup = _LOOKUP

    def _reload(self):
        # consume previous token, load window at new consumed position
        self.bit_ptr += self.bits_toread
        pos = self.bit_ptr - 64
        idx = pos >> 3
        val = int.from_bytes(self.mv[idx : idx + 9], "little")
        self.current = (val >> (pos & 7)) & MASK64

    def buffer_padded(self):
        return self.bit_ptr + 64 < self.nbits

    def buffer_valid_unsafe(self):
        return self.bit_ptr < self.nbits or self.current != 0

    def next_valid(self):
        return self.buffer_padded() or self.lookahead or self.current != 0

    def _time(self, info):
        """Advance globaltime; return realtime value (0 if none)."""
        c = self.current
        typ = info[0]
        if typ == TIMESTAMP:
            pl = (c >> 8) & 1
            rt = (c >> 9) & 1
            tm = (c >> 12) & _TS36
            if pl == 1 and rt == 0:
                self.packetlost = True
            if rt == 0:
                self.globaltime += tm
                return 0
            if pl == 0:
                return tm
            return 0
        tb = info[2]
        te = info[3]
        mask = (1 << (te - tb)) - 1
        delta = ((c >> tb) & mask) + (4 if typ == TIME else 0)
        self.globaltime += delta
        return 0

    def _add_realtime(self, real):
        rl = self.realtime
        if rl:
            sh_back, rt_back = rl[-1]
            if rt_back >= real or sh_back >= self.globaltime:
                rl[-1] = ((sh_back + self.globaltime) // 2, (rt_back + real) // 2)
                return
        rl.append((self.globaltime, real))

    def next(self):
        la = self.lookahead
        lookup = self.lookup
        # Loop 1: padded region (direct realtime emplace).
        while self.buffer_padded() or la:
            if la:
                if la[0][0] < la[-1][0] or not self.buffer_padded():
                    return la.popleft()
            self._reload()
            info = lookup[self.current & 0xFF]
            typ = info[0]
            if typ == NOP:
                self.bits_toread = 4
                continue
            self.bits_toread = info[1]
            real = self._time(info)
            if typ == TIMESTAMP or typ == TIME:
                if real != 0:
                    self.realtime.append((self.globaltime, real))
                continue
            t = self.globaltime - 1 if typ == WAVE_READY else self.globaltime
            tok = (t, self.current, typ)
            if la and t < la[-1][0]:
                la.appendleft(tok)
            else:
                la.append(tok)
        # Loop 2: tail region (deduped realtime).
        while self.buffer_valid_unsafe() or la:
            if la:
                if la[0][0] < la[-1][0] or not self.buffer_valid_unsafe():
                    return la.popleft()
            self._reload()
            info = lookup[self.current & 0xFF]
            typ = info[0]
            if typ == NOP:
                self.bits_toread = 4
                continue
            self.bits_toread = info[1]
            real = self._time(info)
            if typ == TIMESTAMP or typ == TIME:
                if real != 0:
                    self._add_realtime(real)
                continue
            t = self.globaltime - 1 if typ == WAVE_READY else self.globaltime
            tok = (t, self.current, typ)
            if la and t < la[-1][0]:
                la.appendleft(tok)
            else:
                la.append(tok)
        return (0, 0, TIMESTAMP)  # sentinel


class CSRegister:
    """Compute-shader register state -> dispatch snapshot (PopulateDispatch)."""

    __slots__ = (
        "wave_start_addr",
        "dispatch_pkt",
        "num_x",
        "num_y",
        "num_z",
        "rsrc1",
        "rsrc2",
        "rsrc3",
        "tt_version",
    )

    def __init__(self):
        self.wave_start_addr = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.dispatch_pkt = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.num_x = self.num_y = self.num_z = 0
        self.rsrc1 = self.rsrc2 = self.rsrc3 = 0
        self.tt_version = 0

    def update_cs(self, regaddr, regdata, me, pipe):
        rd = regdata & MASK32
        m = me & 1
        if regaddr == 0xC:  # COMPUTE_PGM_LO
            e = self.wave_start_addr[m][pipe]
            self.wave_start_addr[m][pipe] = (e & ~MASK32) | rd
        elif regaddr == 0xD:  # COMPUTE_PGM_HI
            e = self.wave_start_addr[m][pipe]
            self.wave_start_addr[m][pipe] = (e & MASK32) | (rd << 32)
        elif regaddr == 0x7:  # COMPUTE_NUM_THREAD_X
            self.num_x = rd
        elif regaddr == 0x8:
            self.num_y = rd
        elif regaddr == 0x9:
            self.num_z = rd
        elif regaddr == 0x12:  # COMPUTE_PGM_RSRC1
            self.rsrc1 = rd
        elif regaddr == 0x13:  # COMPUTE_PGM_RSRC2
            self.rsrc2 = rd
        elif regaddr == 0x2D:  # COMPUTE_PGM_RSRC3
            self.rsrc3 = rd
        elif regaddr == 0xE:  # COMPUTE_DISPATCH_PKT_LO
            e = self.dispatch_pkt[m][pipe]
            self.dispatch_pkt[m][pipe] = (e & ~MASK32) | rd
        elif regaddr == 0xF:  # COMPUTE_DISPATCH_PKT_HI
            e = self.dispatch_pkt[m][pipe]
            self.dispatch_pkt[m][pipe] = (e & MASK32) | (rd << 32)

    def wave_start_pc(self, me, pipe):
        return (self.wave_start_addr[me & 1][pipe] << 8) & BITMASK48

    def populate_dispatch(self, time, me, pipe, se):
        r1 = self.rsrc1
        r2 = self.rsrc2
        pc = (self.wave_start_addr[me & 1][pipe] << 8) & BITMASK48
        lds = ((r2 >> 15) & 0x1FF) * 512
        vgpr = (r1 & 0x3F) * 8 + 8
        sgpr = 128
        # tt_version 3 (gfx11): no <=1 / ==1 / >=5 adjustments
        flags = 0
        if (r1 >> 10) & 1:
            flags |= 0x1
        if (r1 >> 11) & 1:
            flags |= 0x2
        if (r1 >> 14) & 1:
            flags |= 0x4
        if r2 & 1:
            flags |= 0x8
        return {
            "se": se,
            "me": me,
            "pipe": pipe,
            "t": time,
            "pc": pc,
            "vgpr": vgpr,
            "sgpr": sgpr,
            "lds": lds,
            "wg": self.num_x * self.num_y * self.num_z,
            "flags": flags,
        }


def parse_se(blob, se_index, collect_de):
    """One shader-engine parse pass.

    Returns (dispatches, events, realtime, occupancy). ``dispatches``/``events``/
    ``realtime`` are only populated when ``collect_de`` is set (they are global
    and read from SE0 only, mirroring the C++ parse loop). ``occupancy`` is always
    collected: a list of [time, start, cu, simd, wid, me, pipe, pc] in the exact
    delivery order (synthetic pre-capture starts pushed to the front).
    """
    gen = TokenGen(blob)
    cs = CSRegister()
    running = {}  # gpu_location -> start pc
    saved = {}
    occ = __import__("collections").deque()
    disps = []
    events = []
    tt_version = 0

    while gen.next_valid():
        t, c, typ = gen.next()

        if typ == MISC_GFX10:
            fields = (c >> 16) & 0xFF
            if collect_de:
                # emission order matches rdna_sqtt.cpp
                if (fields >> 3) & 1:
                    events.append((EV_SAVE_CONTEXT, t, 0, 0, 0, 0))
                if (fields >> 4) & 1:
                    events.append((EV_TT_STALL_BEGIN, t, 0, 0, 0, 0))
                if (fields >> 5) & 1:
                    events.append((EV_TT_STALL_END, t, 0, 0, 0, 0))
                if (fields >> 6) & 1:
                    events.append((EV_DIDT_STALL_BEGIN, t, 0, 0, 0, 0))
                if (fields >> 7) & 1:
                    events.append((EV_DIDT_STALL_END, t, 0, 0, 0, 0))
                if (fields >> 1) & 1:
                    events.append((EV_GC_RINSE, t, 0, 0, 0, 0))
                if fields & 1:
                    events.append((EV_SPM_SAMPLE, t, 0, 0, 0, 0))

        elif typ == HEADER:
            tt_version = (c >> 7) & 0x3F
            cs.tt_version = tt_version

        elif typ == WAVE_START or typ == WAVE_START_EXT:
            sa = (c >> 7) & 1
            simd = (c >> 8) & 3
            wgp = (c >> 10) & 7
            wid = (c >> 13) & 0x1F
            pipe = (c >> 21) & 3
            me = (c >> 23) & 1
            count = (c >> 25) & 0x7F
            sacu = (sa << 7) + (wgp & 0x7F)
            loc = (sacu << 7) | (simd << 5) | wid
            if 66 <= count < 90:
                is_save = (0b0000110011 >> (count - 66)) & 1
                is_resr = (0b1111001100 >> (count - 66)) & 1
                if is_save:
                    if loc in running:
                        occ.append([t, 0, sacu, simd, wid, me, pipe, running[loc]])
                        saved[loc] = running[loc]
                elif is_resr:
                    addr = saved.pop(loc, 0)
                    running[loc] = addr
                    occ.append([t, 1, sacu, simd, wid, me, pipe, addr])
                continue
            if gen.packetlost and loc in running:
                occ.append([t - 1, 0, sacu, simd, wid, 0, 0, 0])
            wave_addr = cs.wave_start_pc(me, pipe)
            running[loc] = wave_addr
            occ.append([t, 1, sacu, simd, wid, me, pipe, wave_addr])

        elif typ == WAVE_END:
            sa = (c >> 8) & 1
            simd = (c >> 9) & 3
            wgp = (c >> 11) & 7
            wid = (c >> 15) & 0x1F
            sacu = (sa << 7) + (wgp & 0x7F)
            loc = (sacu << 7) | (simd << 5) | wid
            if loc in running:
                startpc = running.pop(loc)
            else:
                startpc = 0
                occ.appendleft([0, 1, sacu, simd, wid, 0, 0, 0])
            occ.append([t, 0, sacu, simd, wid, 0, 0, startpc])

        elif typ == REG:
            regaddr = (c >> 16) & 0xFFFF
            regdata = (c >> 32) & MASK32
            me = (c >> 9) & 3
            pipe = (c >> 7) & 3
            cs_bit = (c >> 15) & 1
            if cs_bit:
                cs.update_cs(regaddr, regdata, me, pipe)
            elif collect_de and (regaddr == USERDATA_2 or regaddr == USERDATA_3):
                # RGP SQTT marker dword (see USERDATA_2/3 note above). Surface each
                # write as EVENT type 15 (payload = regdata); bop tags USERDATA_3 so
                # the reader can verify the 2/3 interleave. build._parse_markers walks
                # the concatenated stream into per-dispatch grids + barrier fields.
                bop = 1 if regaddr == USERDATA_3 else 0
                events.append((EV_RGP_MARKER, t, me, pipe, regdata, bop))

        elif typ == REG_INIT:
            rtype = (c >> 18) & 3
            data = (c >> 20) & MASK32
            if rtype == 2 and (data & 1) == 1 and collect_de:
                me = (c >> 16) & 3
                pipe = (c >> 14) & 3
                disps.append(cs.populate_dispatch(t, me, pipe, se_index))

        elif typ == EVENT:
            eid = (c >> 18) & 0x3F
            ty = _EVENT_ID_MAP.get(eid)
            if ty is not None and collect_de:
                me = (c >> 16) & 3
                pipe = (c >> 14) & 3
                bop = (c >> 11) & 1
                flags = 0x1 | (0x2 if bop else 0)  # per_pipe + bop
                events.append((ty, t, me & 1, pipe, 0, flags))

        elif typ in _SKIP:
            pass

        else:
            if typ == TIMESTAMP and c == 0:
                break  # sentinel
            if gen.packetlost and collect_de:
                events.append((EV_PACKET_LOSS, t, 0, 0, 0, 0))
                gen.packetlost = False

    return disps, events, gen.realtime, list(occ)


def decode(blobs, wave_size=32):
    """Decode one SQTT blob per shader engine into raw records.

    Returns a dict of raw decode records:
    ``rt_freq, rt0, wave_size, dispatches, events, occupancy``.
    """
    import bisect

    disps, ev_raw, rts, occ0 = parse_se(blobs[0], 0, True)
    occ_all = [occ0]
    for i in range(1, len(blobs)):
        _, _, _, occ_i = parse_se(blobs[i], i, False)
        occ_all.append(occ_i)

    rt_freq = 100_000_000  # RGP capture: decoder emits no RT_FREQUENCY -> fallback

    disps.sort(key=lambda d: d["t"])
    for d in disps:
        d["exp_waves"] = 0
        d["wf"] = 0
        d["min_start"] = INF
        d["max_end"] = 0

    # --- phase 2: pair occupancy waves per hw slot + build per-SE band ------
    slot_pend = {}
    pipe_waves = {}
    occ_ev = [[] for _ in blobs]
    for se, occ in enumerate(occ_all):
        oe = occ_ev[se]
        for rec in occ:
            time, start, cu, simd, wid, me, pipe, pc = rec
            sk = (se << 24) | ((cu & 0x7F) << 16) | ((simd & 0xF) << 8) | (wid & 0xFF)
            oe.append((time, 1 if start else -1))
            if start:
                slot_pend[sk] = (time, pc, me, pipe)
            else:
                p = slot_pend.pop(sk, None)
                if p is None:
                    continue
                st, ppc, pme, ppipe = p
                key = (ppc << 8) | (((pme & 7) << 4) | (ppipe & 0xF))
                lst = pipe_waves.get(key)
                if lst is None:
                    pipe_waves[key] = [(st, time)]
                else:
                    lst.append((st, time))

    # --- exp_waves from RGP markers (USERDATA_2/3), per (me,pipe) ------------
    # Reassemble the type-15 marker dwords in capture
    # order, walk EVENT markers (ident 0x0 w/ dims) for grid workgroups, and pair
    # them 1:1 with dispatches in launch order on the pipe. exp_waves =
    # grid_workgroups * ceil(wg/wave). This gates the wave-attribution cap below so
    # a sparse same-PC dispatch cannot absorb waves from the idle gap before its
    # next invocation and inflate its wave-span (the small-dispatch duration bug).
    def _mkey(me, pipe):
        return ((me & 7) << 4) | (pipe & 0xF)

    pipe_marker_dw = {}
    for e in ev_raw:  # ev_raw is already in capture (seq) order
        if e[0] == EV_RGP_MARKER:
            pipe_marker_dw.setdefault(_mkey(e[2], e[3]), []).append(e[4] & 0xFFFFFFFF)
    pipe_grids = {}
    for mk, dw in pipe_marker_dw.items():
        grids = []
        i, n = 0, len(dw)
        while i < n:
            d0 = dw[i]
            ident = d0 & 0xF
            ext = (d0 >> 4) & 0x7
            if ident == 0x0:  # EVENT (per dispatch)
                has = (d0 >> 31) & 1
                grids.append(
                    dw[i + 3] * dw[i + 4] * dw[i + 5] if (has and i + 5 < n) else 0
                )
                i += 3 + (3 if has else 0) + ext
            elif ident == 0x1:  # CB_START
                i += 4 + ext
            elif ident == 0x2 or ident == 0xC:  # CB_END / BIND_PIPELINE
                i += 3 + ext
            elif ident == 0x3 or ident == 0x4 or ident == 0x9:  # BARRIER_*/LAYOUT
                i += 2 + ext
            else:
                i += 1 + ext
        pipe_grids[mk] = grids
    pipe_disps = {}
    for i, d in enumerate(disps):
        pipe_disps.setdefault(_mkey(d["me"], d["pipe"]), []).append(i)
    for mk, idxs in pipe_disps.items():
        grids = pipe_grids.get(mk, [])
        for k, di in enumerate(idxs):
            d = disps[di]
            wg = d["wg"] or 1
            wpw = (wg + wave_size - 1) // wave_size
            grid = grids[k] if k < len(grids) else 0
            d["exp_waves"] = grid * wpw

    # --- attribute waves to dispatches by (me,pipe,pc) + launch window ------
    pc_disps = {}
    for i, d in enumerate(disps):
        key = (d["pc"] << 8) | (((d["me"] & 7) << 4) | (d["pipe"] & 0xF))
        pc_disps.setdefault(key, []).append(i)
    for key, idxs in pc_disps.items():
        wv = pipe_waves.get(key)
        if not wv:
            continue
        times = [disps[i]["t"] for i in idxs]
        wv.sort(key=lambda w: w[0])
        for ws, we in wv:
            k = bisect.bisect_right(times, ws) - 1
            if k < 0:
                k = 0
            d = disps[idxs[k]]
            if d["exp_waves"] > 0 and d["wf"] >= d["exp_waves"]:
                continue
            d["wf"] += 1
            if ws < d["min_start"]:
                d["min_start"] = ws
            if we > d["max_end"]:
                d["max_end"] = we

    # --- wall-clock mapping -------------------------------------------------
    rts_sorted = sorted(rts, key=lambda r: r[0])
    rt_sh = [r[0] for r in rts_sorted]

    def sc_to_rt(sc):
        if not rts_sorted:
            return float(sc)
        if sc <= rts_sorted[0][0]:
            return float(rts_sorted[0][1])
        if sc >= rts_sorted[-1][0]:
            return float(rts_sorted[-1][1])
        j = bisect.bisect_left(rt_sh, sc)
        a = rts_sorted[j - 1]
        b = rts_sorted[j]
        f = (sc - a[0]) / max(1, b[0] - a[0])
        return a[1] + f * (b[1] - a[1])

    sc0 = INF
    for d in disps:
        if d["t"] < sc0:
            sc0 = d["t"]
    for e in ev_raw:
        if e[1] < sc0:
            sc0 = e[1]
    if sc0 == INF:
        sc0 = 0
    rt0 = sc_to_rt(sc0)
    US = 1e6 / rt_freq

    def to_us(sc):
        return (sc_to_rt(sc) - rt0) * US

    out_disp = []
    for d in disps:
        has = d["wf"] > 0 and d["min_start"] != INF
        ws = d["min_start"] if has else d["t"]
        dur = round(to_us(d["max_end"]) - to_us(d["min_start"]), 4) if has else -1.0
        out_disp.append(
            {
                "se": d["se"],
                "me": d["me"],
                "pipe": d["pipe"],
                "ts": round(to_us(ws), 4),
                "launch_ts": round(to_us(d["t"]), 4),
                "dur": dur,
                "wavefronts": d["wf"],
                "exp_waves": d["exp_waves"],
                "flags": d["flags"],
                "vgpr": d["vgpr"],
                "sgpr": d["sgpr"],
                "lds": d["lds"],
                "wg": d["wg"],
                "pc": "0x%x" % d["pc"],
            }
        )

    out_ev = []
    for seq, e in enumerate(ev_raw):
        ty, rawt, me, pipe, payload, flags = e
        out_ev.append(
            {
                "me": me,
                "pipe": pipe,
                "ts": round(to_us(rawt), 4),
                "rawt": rawt,
                "type": ty,
                "flags": flags,
                "payload": payload,
                "seq": seq,
            }
        )

    out_occ = _occupancy_band(occ_ev, len(blobs), to_us)

    return {
        "rt_freq": rt_freq,
        "rt0": int(rt0),
        "wave_size": wave_size,
        "dispatches": out_disp,
        "events": out_ev,
        "occupancy": out_occ,
    }


def _occupancy_band(occ_ev, n_se, to_us):
    """Peak alive-wave count per ~2us bucket, per SE (wavefront-occupancy band)."""
    ns = min(n_se, 8)
    tmin = INF
    tmax = -INF
    for s in range(ns):
        for time, _ in occ_ev[s]:
            if time < tmin:
                tmin = time
            if time > tmax:
                tmax = time
    if not (tmax > tmin):
        return []
    span_us = to_us(tmax) - to_us(tmin)
    nsamp = int(span_us / 2.0)
    if nsamp < 4000:
        nsamp = 4000
    if nsamp > 120000:
        nsamp = 120000
    for s in range(ns):
        occ_ev[s].sort()
    cur = [0] * ns
    idx = [0] * ns
    span = tmax - tmin
    band = []
    for k in range(nsamp):
        t1 = tmin + int(span * (k + 1) / nsamp)
        tmid = tmin + int(span * (k + 0.5) / nsamp)
        row = {"ts": round(to_us(tmid), 4)}
        for s in range(ns):
            E = occ_ev[s]
            ci = cur[s]
            peak = ci if ci > 0 else 0
            i = idx[s]
            ne = len(E)
            while i < ne and E[i][0] < t1:
                ci += E[i][1]
                if ci > peak:
                    peak = ci
                i += 1
            cur[s] = ci
            idx[s] = i
            row["se%d" % s] = peak
        band.append(row)
    return band
