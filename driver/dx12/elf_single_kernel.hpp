/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Rebuilds a multi-kernel AMDGPU ET_REL object into a single-kernel ET_REL.
 * PAL InitFromHsaAbiBinary resolves one kernel via its .kd symbol; handing it a
 * multi-kernel object removes the device rather than returning an error.
 */
#pragma once

#include "hsa_metadata.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hip_ep {
namespace dx12 {
namespace elf_split {

namespace detail {

struct Section
{
    std::string   name;
    std::uint32_t type = 0;
    std::uint64_t off  = 0;
    std::uint64_t size = 0;
    std::uint64_t link = 0;
    std::uint64_t entsize = 0;
};

inline bool read_sections(const std::vector<std::uint8_t>& elf, std::vector<Section>& out)
{
    if(elf.size() < 64) return false;
    std::uint64_t shoff = 0;
    std::uint16_t shentsize = 0, shnum = 0, shstrndx = 0;
    std::memcpy(&shoff,     elf.data() + 40, 8);
    std::memcpy(&shentsize, elf.data() + 58, 2);
    std::memcpy(&shnum,     elf.data() + 60, 2);
    std::memcpy(&shstrndx,  elf.data() + 62, 2);
    if(shoff == 0 || shentsize < 64 || shnum == 0 || shstrndx >= shnum) return false;

    const std::uint8_t* shstr_hdr = elf.data() + shoff + static_cast<std::uint64_t>(shentsize) * shstrndx;
    std::uint64_t shstr_off = 0, shstr_size = 0;
    std::memcpy(&shstr_off,  shstr_hdr + 24, 8);
    std::memcpy(&shstr_size, shstr_hdr + 32, 8);
    if(shstr_off + shstr_size > elf.size()) return false;

    out.clear();
    out.reserve(shnum);
    for(std::uint16_t i = 0; i < shnum; ++i)
    {
        const std::uint64_t so = shoff + static_cast<std::uint64_t>(shentsize) * i;
        if(so + 64 > elf.size()) return false;
        const std::uint8_t* sh = elf.data() + so;
        std::uint32_t nameoff = 0;
        Section s;
        std::memcpy(&nameoff,   sh +  0, 4);
        std::memcpy(&s.type,    sh +  4, 4);
        std::memcpy(&s.off,     sh + 24, 8);
        std::memcpy(&s.size,    sh + 32, 8);
        std::uint32_t link32 = 0;
        std::memcpy(&link32,    sh + 40, 4);
        s.link = link32;
        std::memcpy(&s.entsize, sh + 56, 8);
        if(nameoff < shstr_size)
        {
            const char* p = reinterpret_cast<const char*>(elf.data() + shstr_off + nameoff);
            s.name.assign(p, ::strnlen(p, static_cast<std::size_t>(shstr_size - nameoff)));
        }
        out.push_back(std::move(s));
    }
    return true;
}

struct SymRef
{
    std::uint64_t file_off = 0;
    std::uint64_t size     = 0;
    bool          found    = false;
};

// Locate a symbol by name in .symtab and resolve it to a file offset.
inline bool find_symbol(const std::vector<std::uint8_t>& elf,
                        const std::vector<Section>& secs,
                        const std::string& want,
                        SymRef& out,
                        std::string* cuid_name = nullptr)
{
    for(const auto& st : secs)
    {
        if(st.type != 2 /*SHT_SYMTAB*/) continue;
        if(st.link >= secs.size()) continue;
        const Section& strtab = secs[st.link];
        if(strtab.off + strtab.size > elf.size()) continue;

        const std::uint64_t ent = st.entsize >= 24 ? st.entsize : 24u;
        const std::uint64_t n   = st.size / ent;
        for(std::uint64_t i = 0; i < n; ++i)
        {
            const std::uint64_t so = st.off + i * ent;
            if(so + 24 > elf.size()) break;
            const std::uint8_t* sym = elf.data() + so;
            std::uint32_t st_name = 0;
            std::uint16_t st_shndx = 0;
            std::uint64_t st_value = 0, st_size = 0;
            std::memcpy(&st_name,  sym +  0, 4);
            std::memcpy(&st_shndx, sym +  6, 2);
            std::memcpy(&st_value, sym +  8, 8);
            std::memcpy(&st_size,  sym + 16, 8);
            if(st_name == 0 || st_name >= strtab.size) continue;

            const char* np = reinterpret_cast<const char*>(elf.data() + strtab.off + st_name);
            const std::string nm(np, ::strnlen(np, static_cast<std::size_t>(strtab.size - st_name)));

            if(cuid_name != nullptr && nm.rfind("__hip_cuid_", 0) == 0)
                *cuid_name = nm;

            if(nm != want) continue;
            if(st_shndx == 0 || st_shndx >= secs.size()) continue;

            out.file_off = secs[st_shndx].off + st_value;   // ET_REL: st_value is section-relative
            out.size     = st_size;
            out.found    = true;
            if(cuid_name == nullptr) return true;
        }
    }
    return out.found;
}

inline void push32(std::vector<std::uint8_t>& v, std::uint32_t x)
{
    v.push_back(static_cast<std::uint8_t>(x & 0xff));
    v.push_back(static_cast<std::uint8_t>((x >> 8) & 0xff));
    v.push_back(static_cast<std::uint8_t>((x >> 16) & 0xff));
    v.push_back(static_cast<std::uint8_t>((x >> 24) & 0xff));
}

// Rewrite the note msgpack so amdhsa.kernels holds only the requested kernel.
inline std::vector<std::uint8_t> filter_note(const std::uint8_t* desc, std::size_t desc_size,
                                             const std::string& kernel_name)
{
    namespace md = hip_ep::dx12::hsa_md::detail;
    md::Cursor c{desc, desc_size, 0};
    std::size_t root_n = 0;
    const std::size_t root_start = c.p;
    if(!md::read_map_hdr(c, root_n)) return {};
    const std::size_t root_hdr_size = c.p - root_start;

    std::vector<std::uint8_t> out;
    out.reserve(desc_size);
    out.insert(out.end(), desc, desc + root_hdr_size);

    bool wrote_kernels = false;
    for(std::size_t i = 0; i < root_n; ++i)
    {
        const std::size_t koff = c.p;
        std::string key;
        if(!md::read_str(c, key)) return {};
        const std::size_t ksize = c.p - koff;

        const std::size_t voff = c.p;
        if(key != "amdhsa.kernels")
        {
            if(!md::skip_value(c)) return {};
            out.insert(out.end(), desc + koff, desc + c.p);
            continue;
        }

        std::size_t nk = 0;
        if(!md::read_arr_hdr(c, nk)) return {};
        std::size_t hit_off = 0, hit_size = 0;
        for(std::size_t k = 0; k < nk; ++k)
        {
            const std::size_t eoff = c.p;
            md::Cursor probe = c;
            std::size_t nf = 0;
            if(!md::read_map_hdr(probe, nf)) return {};
            std::string this_name;
            for(std::size_t f = 0; f < nf; ++f)
            {
                std::string fk;
                if(!md::read_str(probe, fk)) return {};
                if(fk == ".name")
                {
                    if(!md::read_str(probe, this_name)) return {};
                }
                else if(!md::skip_value(probe)) return {};
            }
            if(!md::skip_value(c)) return {};
            if(this_name == kernel_name) { hit_off = eoff; hit_size = c.p - eoff; }
        }
        if(hit_size == 0) return {};

        out.insert(out.end(), desc + koff, desc + koff + ksize);
        out.push_back(0x91); // fixarray(1)
        out.insert(out.end(), desc + hit_off, desc + hit_off + hit_size);
        wrote_kernels = true;
        (void)voff;
    }
    if(!wrote_kernels) return {};
    return out;
}

} // namespace detail

// Returns a single-kernel ET_REL for kernel_name, or empty on any failure.
inline std::vector<std::uint8_t> build_single_kernel(const std::vector<std::uint8_t>& src,
                                                     const std::string& kernel_name)
{
    if(src.size() < 64) return {};
    if(src[0] != 0x7F || src[1] != 'E' || src[2] != 'L' || src[3] != 'F') return {};

    std::vector<detail::Section> secs;
    if(!detail::read_sections(src, secs)) return {};

    detail::SymRef kern{}, kd{};
    std::string cuid = "__hip_cuid_000000000000";
    if(!detail::find_symbol(src, secs, kernel_name, kern, &cuid)) return {};
    if(!detail::find_symbol(src, secs, kernel_name + ".kd", kd)) return {};
    if(kern.size == 0) return {};

    constexpr std::uint64_t k_kd_size = 64u;
    if(kern.file_off + kern.size > src.size()) return {};
    if(kd.file_off + k_kd_size > src.size()) return {};

    const std::uint8_t* note_desc = nullptr;
    std::size_t note_desc_size = 0;
    if(!hip_ep::dx12::hsa_md::detail::find_amdgpu_note(src, note_desc, note_desc_size)) return {};
    std::vector<std::uint8_t> new_note = detail::filter_note(note_desc, note_desc_size, kernel_name);
    if(new_note.empty()) return {};

    std::uint16_t e_machine = 0;
    std::uint32_t e_flags   = 0;
    std::memcpy(&e_machine, src.data() + 18, 2);
    std::memcpy(&e_flags,   src.data() + 48, 4);

    // .strtab
    const std::string kd_name = kernel_name + ".kd";
    std::vector<std::uint8_t> strtab{0};
    const std::uint32_t kern_str = static_cast<std::uint32_t>(strtab.size());
    strtab.insert(strtab.end(), kernel_name.begin(), kernel_name.end()); strtab.push_back(0);
    const std::uint32_t kd_str = static_cast<std::uint32_t>(strtab.size());
    strtab.insert(strtab.end(), kd_name.begin(), kd_name.end()); strtab.push_back(0);
    const std::uint32_t cuid_str = static_cast<std::uint32_t>(strtab.size());
    strtab.insert(strtab.end(), cuid.begin(), cuid.end()); strtab.push_back(0);

    // .shstrtab
    std::vector<std::uint8_t> shstr{0};
    auto add_shstr = [&](const char* s) {
        const std::uint32_t idx = static_cast<std::uint32_t>(shstr.size());
        while(*s) shstr.push_back(static_cast<std::uint8_t>(*s++));
        shstr.push_back(0);
        return idx;
    };
    const std::uint32_t n_strtab   = add_shstr(".strtab");
    const std::uint32_t n_text     = add_shstr(".text");
    const std::uint32_t n_rodata   = add_shstr(".rodata");
    const std::uint32_t n_bss      = add_shstr(".bss");
    const std::uint32_t n_note     = add_shstr(".note");
    const std::uint32_t n_shstrtab = add_shstr(".shstrtab");
    const std::uint32_t n_symtab   = add_shstr(".symtab");

    // .note section payload
    std::vector<std::uint8_t> note_sec;
    detail::push32(note_sec, 7u);                                            // namesz
    detail::push32(note_sec, static_cast<std::uint32_t>(new_note.size()));   // descsz
    detail::push32(note_sec, 32u);                                           // NT_AMDGPU_METADATA
    const std::uint8_t nm[8] = {'A','M','D','G','P','U',0,0};
    note_sec.insert(note_sec.end(), nm, nm + 8);
    note_sec.insert(note_sec.end(), new_note.begin(), new_note.end());
    while(note_sec.size() % 4) note_sec.push_back(0);

    auto align_up = [](std::uint64_t v, std::uint64_t a) { return (v + a - 1) & ~(a - 1); };
    constexpr std::uint64_t EHDR = 64u, SHDR = 64u;
    constexpr std::uint16_t NSECS = 8u;

    const std::uint64_t text_off   = align_up(EHDR, 256u);
    const std::uint64_t rodata_off = align_up(text_off + kern.size, 64u);
    const std::uint64_t bss_off    = rodata_off + k_kd_size;
    const std::uint64_t note_off   = align_up(bss_off, 4u);
    const std::uint64_t strtab_off = note_off + note_sec.size();
    const std::uint64_t shstr_off  = strtab_off + strtab.size();
    const std::uint64_t shdr_off   = align_up(shstr_off + shstr.size(), 8u);
    const std::uint64_t symtab_off = align_up(shdr_off + NSECS * SHDR, 8u);
    const std::uint64_t total      = symtab_off + 4u * 24u;

    std::vector<std::uint8_t> out(static_cast<std::size_t>(total), 0);
    auto w16 = [&](std::uint64_t o, std::uint16_t v) { std::memcpy(out.data() + o, &v, 2); };
    auto w32 = [&](std::uint64_t o, std::uint32_t v) { std::memcpy(out.data() + o, &v, 4); };
    auto w64 = [&](std::uint64_t o, std::uint64_t v) { std::memcpy(out.data() + o, &v, 8); };

    out[0] = 0x7F; out[1] = 'E'; out[2] = 'L'; out[3] = 'F';
    out[4] = 2; out[5] = 1; out[6] = 1;
    out[7] = 0x40;        // ELFOSABI_AMDGPU_HSA
    out[8] = src[8];      // preserve code-object ABI version
    w16(16, 1u);          // ET_REL
    w16(18, e_machine);
    w32(20, 1u);
    w64(40, shdr_off);
    w32(48, e_flags);
    w16(52, static_cast<std::uint16_t>(EHDR));
    w16(58, static_cast<std::uint16_t>(SHDR));
    w16(60, NSECS);
    w16(62, 6u);          // .shstrtab index

    std::memcpy(out.data() + text_off,   src.data() + kern.file_off, static_cast<std::size_t>(kern.size));
    std::memcpy(out.data() + rodata_off, src.data() + kd.file_off,   static_cast<std::size_t>(k_kd_size));
    std::memcpy(out.data() + note_off,   note_sec.data(),            note_sec.size());
    std::memcpy(out.data() + strtab_off, strtab.data(),              strtab.size());
    std::memcpy(out.data() + shstr_off,  shstr.data(),               shstr.size());

    auto shdr = [&](std::uint16_t i, std::uint32_t name, std::uint32_t type, std::uint64_t flags,
                    std::uint64_t off, std::uint64_t size, std::uint32_t link, std::uint32_t info,
                    std::uint64_t align, std::uint64_t entsz) {
        const std::uint64_t b = shdr_off + static_cast<std::uint64_t>(i) * SHDR;
        w32(b + 0, name); w32(b + 4, type); w64(b + 8, flags); w64(b + 16, 0);
        w64(b + 24, off); w64(b + 32, size); w32(b + 40, link); w32(b + 44, info);
        w64(b + 48, align); w64(b + 56, entsz);
    };
    shdr(1, n_strtab,   3, 0, strtab_off, strtab.size(),  0, 0,   1,  0);
    shdr(2, n_text,     1, 6, text_off,   kern.size,      0, 0, 256,  0);
    shdr(3, n_rodata,   1, 2, rodata_off, k_kd_size,      0, 0,  64,  0);
    shdr(4, n_bss,      8, 3, bss_off,    1,              0, 0,   1,  0);
    shdr(5, n_note,     7, 2, note_off,   note_sec.size(),0, 0,   4,  0);
    shdr(6, n_shstrtab, 3, 0, shstr_off,  shstr.size(),   0, 0,   1,  0);
    shdr(7, n_symtab,   2, 0, symtab_off, 4u * 24u,       1, 1,   8, 24);

    auto sym = [&](std::uint8_t i, std::uint32_t name, std::uint8_t info,
                   std::uint16_t shndx, std::uint64_t size) {
        const std::uint64_t b = symtab_off + static_cast<std::uint64_t>(i) * 24u;
        w32(b + 0, name); out[b + 4] = info; out[b + 5] = 0; w16(b + 6, shndx);
        w64(b + 8, 0); w64(b + 16, size);
    };
    sym(0, 0,        0x00, 0, 0);
    sym(1, kern_str, 0x12, 2, kern.size);   // GLOBAL FUNC   -> .text
    sym(2, kd_str,   0x11, 3, k_kd_size);   // GLOBAL OBJECT -> .rodata
    sym(3, cuid_str, 0x11, 4, 1);           // GLOBAL OBJECT -> .bss

    return out;
}

} // namespace elf_split
} // namespace dx12
} // namespace hip_ep
