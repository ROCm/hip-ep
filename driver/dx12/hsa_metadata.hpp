/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Parses the AMDHSA kernel metadata (msgpack in the AMDGPU ELF .note section)
 * so a kernel's argument ABI can be recovered from the code object itself
 * rather than hard-coded per kernel.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hip_ep {
namespace dx12 {
namespace hsa_md {

enum class ArgKind  { GlobalBuffer, ByValue, Hidden, Other };
enum class ArgAccess{ ReadOnly, WriteOnly, ReadWrite, Unknown };

struct KernelArg
{
    std::size_t offset = 0;
    std::size_t size   = 0;
    ArgKind     kind   = ArgKind::Other;
    ArgAccess   access = ArgAccess::Unknown;
    std::string value_kind;
    std::string name;
};

struct KernelAbi
{
    std::string             name;
    std::size_t             kernarg_segment_size    = 0;
    std::uint32_t           max_flat_workgroup_size = 256;
    std::size_t             group_segment_fixed_size = 0;
    std::vector<KernelArg>  args;

    std::size_t explicit_arg_count() const
    {
        std::size_t n = 0;
        for(const auto& a : args)
            if(a.kind != ArgKind::Hidden) ++n;
        return n;
    }
};

// ---------------------------------------------------------------------------
// Minimal msgpack reader (only the subset AMDHSA metadata uses).
// ---------------------------------------------------------------------------
namespace detail {

struct Cursor { const std::uint8_t* d; std::size_t n; std::size_t p; };

inline std::uint64_t be_read(const std::uint8_t* p, int nbytes)
{
    std::uint64_t v = 0;
    for(int i = 0; i < nbytes; ++i) v = (v << 8) | p[i];
    return v;
}

inline bool peek(Cursor& c, std::uint8_t& b)
{
    if(c.p >= c.n) return false;
    b = c.d[c.p];
    return true;
}

inline bool read_uint(Cursor& c, std::uint64_t& out)
{
    std::uint8_t b;
    if(!peek(c, b)) return false;
    if(b <= 0x7f)                 { out = b; c.p += 1; return true; }
    if(b == 0xcc && c.p + 1 < c.n){ out = c.d[c.p+1]; c.p += 2; return true; }
    if(b == 0xcd && c.p + 2 < c.n){ out = be_read(c.d + c.p + 1, 2); c.p += 3; return true; }
    if(b == 0xce && c.p + 4 < c.n){ out = be_read(c.d + c.p + 1, 4); c.p += 5; return true; }
    if(b == 0xcf && c.p + 8 < c.n){ out = be_read(c.d + c.p + 1, 8); c.p += 9; return true; }
    return false;
}

inline bool read_str(Cursor& c, std::string& out)
{
    std::uint8_t b;
    if(!peek(c, b)) return false;
    std::size_t len = 0, hdr = 0;
    if((b & 0xe0) == 0xa0)               { len = b & 0x1fu; hdr = 1; }
    else if(b == 0xd9 && c.p + 1 < c.n)  { len = c.d[c.p+1]; hdr = 2; }
    else if(b == 0xda && c.p + 2 < c.n)  { len = static_cast<std::size_t>(be_read(c.d + c.p + 1, 2)); hdr = 3; }
    else if(b == 0xdb && c.p + 4 < c.n)  { len = static_cast<std::size_t>(be_read(c.d + c.p + 1, 4)); hdr = 5; }
    else return false;
    if(c.p + hdr + len > c.n) return false;
    out.assign(reinterpret_cast<const char*>(c.d + c.p + hdr), len);
    c.p += hdr + len;
    return true;
}

inline bool read_map_hdr(Cursor& c, std::size_t& count)
{
    std::uint8_t b;
    if(!peek(c, b)) return false;
    if((b & 0xf0) == 0x80)             { count = b & 0x0fu; c.p += 1; return true; }
    if(b == 0xde && c.p + 2 < c.n)     { count = static_cast<std::size_t>(be_read(c.d + c.p + 1, 2)); c.p += 3; return true; }
    if(b == 0xdf && c.p + 4 < c.n)     { count = static_cast<std::size_t>(be_read(c.d + c.p + 1, 4)); c.p += 5; return true; }
    return false;
}

inline bool read_arr_hdr(Cursor& c, std::size_t& count)
{
    std::uint8_t b;
    if(!peek(c, b)) return false;
    if((b & 0xf0) == 0x90)             { count = b & 0x0fu; c.p += 1; return true; }
    if(b == 0xdc && c.p + 2 < c.n)     { count = static_cast<std::size_t>(be_read(c.d + c.p + 1, 2)); c.p += 3; return true; }
    if(b == 0xdd && c.p + 4 < c.n)     { count = static_cast<std::size_t>(be_read(c.d + c.p + 1, 4)); c.p += 5; return true; }
    return false;
}

bool skip_value(Cursor& c);

inline bool skip_n(Cursor& c, std::size_t n)
{
    for(std::size_t i = 0; i < n; ++i)
        if(!skip_value(c)) return false;
    return true;
}

inline bool skip_value(Cursor& c)
{
    std::uint8_t b;
    if(!peek(c, b)) return false;

    if(b <= 0x7f || b >= 0xe0) { c.p += 1; return true; }          // fixint
    if((b & 0xe0) == 0xa0)     { std::string s; return read_str(c, s); }
    if((b & 0xf0) == 0x90)     { std::size_t n; return read_arr_hdr(c, n) && skip_n(c, n); }
    if((b & 0xf0) == 0x80)     { std::size_t n; return read_map_hdr(c, n) && skip_n(c, 2 * n); }

    switch(b)
    {
    case 0xc0: case 0xc2: case 0xc3: c.p += 1; return true;        // nil/false/true
    case 0xcc: case 0xd0:            c.p += 2; return true;
    case 0xcd: case 0xd1:            c.p += 3; return true;
    case 0xce: case 0xd2: case 0xca: c.p += 5; return true;
    case 0xcf: case 0xd3: case 0xcb: c.p += 9; return true;
    default: break;
    }
    if(b == 0xd9 || b == 0xda || b == 0xdb) { std::string s; return read_str(c, s); }
    if(b == 0xdc || b == 0xdd)              { std::size_t n; return read_arr_hdr(c, n) && skip_n(c, n); }
    if(b == 0xde || b == 0xdf)              { std::size_t n; return read_map_hdr(c, n) && skip_n(c, 2 * n); }
    return false;
}

// Locate the AMDGPU note descriptor bytes inside an ELF64 image.
inline bool find_amdgpu_note(const std::vector<std::uint8_t>& elf,
                             const std::uint8_t*& desc, std::size_t& desc_size)
{
    if(elf.size() < 64) return false;
    if(elf[0] != 0x7F || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F') return false;

    std::uint64_t shoff = 0;
    std::uint16_t shentsize = 0, shnum = 0;
    std::memcpy(&shoff,     elf.data() + 40, 8);
    std::memcpy(&shentsize, elf.data() + 58, 2);
    std::memcpy(&shnum,     elf.data() + 60, 2);
    if(shoff == 0 || shentsize < 64 || shnum == 0) return false;

    for(std::uint16_t i = 0; i < shnum; ++i)
    {
        const std::uint64_t so = shoff + static_cast<std::uint64_t>(shentsize) * i;
        if(so + 64 > elf.size()) return false;
        const std::uint8_t* sh = elf.data() + so;

        std::uint32_t sh_type = 0;
        std::uint64_t sh_off = 0, sh_size = 0;
        std::memcpy(&sh_type, sh + 4, 4);
        std::memcpy(&sh_off, sh + 24, 8);
        std::memcpy(&sh_size, sh + 32, 8);
        if(sh_type != 7 /*SHT_NOTE*/) continue;                     // not a note
        if(sh_off + sh_size > elf.size() || sh_size < 12) continue;

        std::uint32_t namesz = 0, descsz = 0;
        std::memcpy(&namesz, elf.data() + sh_off,     4);
        std::memcpy(&descsz, elf.data() + sh_off + 4, 4);
        const std::size_t name_aligned = (namesz + 3u) & ~3u;
        if(12 + name_aligned + descsz > sh_size) continue;
        if(std::strncmp(reinterpret_cast<const char*>(elf.data() + sh_off + 12), "AMDGPU", 6) != 0)
            continue;

        desc      = elf.data() + sh_off + 12 + name_aligned;
        desc_size = descsz;
        return true;
    }
    return false;
}

} // namespace detail

// Parse every kernel's ABI out of an AMDGPU code object. Empty on failure.
inline std::vector<KernelAbi> parse(const std::vector<std::uint8_t>& elf)
{
    std::vector<KernelAbi> out;

    const std::uint8_t* desc = nullptr;
    std::size_t desc_size = 0;
    if(!detail::find_amdgpu_note(elf, desc, desc_size)) return out;

    detail::Cursor c{desc, desc_size, 0};
    std::size_t top_n = 0;
    if(!detail::read_map_hdr(c, top_n)) return out;

    for(std::size_t i = 0; i < top_n; ++i)
    {
        std::string key;
        if(!detail::read_str(c, key)) return out;
        if(key != "amdhsa.kernels") { if(!detail::skip_value(c)) return out; continue; }

        std::size_t nk = 0;
        if(!detail::read_arr_hdr(c, nk)) return out;

        for(std::size_t k = 0; k < nk; ++k)
        {
            std::size_t nf = 0;
            if(!detail::read_map_hdr(c, nf)) return out;

            KernelAbi abi;
            for(std::size_t f = 0; f < nf; ++f)
            {
                std::string fk;
                if(!detail::read_str(c, fk)) return out;

                if(fk == ".name" || fk == ".symbol")
                {
                    std::string v;
                    if(!detail::read_str(c, v)) return out;
                    if(fk == ".name") abi.name = v;
                }
                else if(fk == ".kernarg_segment_size")
                {
                    std::uint64_t v = 0;
                    if(!detail::read_uint(c, v)) return out;
                    abi.kernarg_segment_size = static_cast<std::size_t>(v);
                }
                else if(fk == ".max_flat_workgroup_size")
                {
                    std::uint64_t v = 0;
                    if(!detail::read_uint(c, v)) return out;
                    abi.max_flat_workgroup_size = static_cast<std::uint32_t>(v);
                }
                else if(fk == ".group_segment_fixed_size")
                {
                    std::uint64_t v = 0;
                    if(!detail::read_uint(c, v)) return out;
                    abi.group_segment_fixed_size = static_cast<std::size_t>(v);
                }
                else if(fk == ".args")
                {
                    std::size_t na = 0;
                    if(!detail::read_arr_hdr(c, na)) return out;
                    for(std::size_t a = 0; a < na; ++a)
                    {
                        std::size_t naf = 0;
                        if(!detail::read_map_hdr(c, naf)) return out;
                        KernelArg arg;
                        for(std::size_t af = 0; af < naf; ++af)
                        {
                            std::string ak;
                            if(!detail::read_str(c, ak)) return out;
                            if(ak == ".offset" || ak == ".size")
                            {
                                std::uint64_t v = 0;
                                if(!detail::read_uint(c, v)) return out;
                                if(ak == ".offset") arg.offset = static_cast<std::size_t>(v);
                                else                arg.size   = static_cast<std::size_t>(v);
                            }
                            else if(ak == ".value_kind")
                            {
                                if(!detail::read_str(c, arg.value_kind)) return out;
                            }
                            else if(ak == ".name")
                            {
                                if(!detail::read_str(c, arg.name)) return out;
                            }
                            else if(ak == ".actual_access")
                            {
                                std::string v;
                                if(!detail::read_str(c, v)) return out;
                                if(v == "read_only")       arg.access = ArgAccess::ReadOnly;
                                else if(v == "write_only") arg.access = ArgAccess::WriteOnly;
                                else if(v == "read_write") arg.access = ArgAccess::ReadWrite;
                            }
                            else if(!detail::skip_value(c)) return out;
                        }
                        if(arg.value_kind == "global_buffer")      arg.kind = ArgKind::GlobalBuffer;
                        else if(arg.value_kind == "by_value")      arg.kind = ArgKind::ByValue;
                        else if(arg.value_kind.rfind("hidden_", 0) == 0) arg.kind = ArgKind::Hidden;
                        abi.args.push_back(std::move(arg));
                    }
                }
                else if(!detail::skip_value(c)) return out;
            }
            out.push_back(std::move(abi));
        }
        return out;
    }
    return out;
}

} // namespace hsa_md
} // namespace dx12
} // namespace hip_ep
