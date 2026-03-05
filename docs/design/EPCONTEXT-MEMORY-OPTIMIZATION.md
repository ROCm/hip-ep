<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# EPContext Memory Optimization

**Date:** 2026-02-13
**Document Type:** Design
**Review Status:** Draft
**Related:** CONSTANT-HANDLING-DESIGN.md, ARCHITECTURE.md

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Current Approach: MemoryModule (2x Memory)](#current-approach-memorymodule-2x-memory)
- [Proposed Optimization: Direct mmap Execution (1x Memory)](#proposed-optimization-direct-mmap-execution-1x-memory)
- [Technical Deep Dive](#technical-deep-dive)
  - [Why 2x Memory with MemoryModule](#why-2x-memory-with-memorymodule)
  - [How Standard LoadLibrary Works](#how-standard-loadlibrary-works)
  - [RIP-Relative Addressing on x64](#rip-relative-addressing-on-x64)
  - [Copy-on-Write Behavior](#copy-on-write-behavior)
- [Implementation Approach](#implementation-approach)
- [Open Questions](#open-questions)
- [References](#references)

---

## Problem Statement

**Context:** EPContext stores compiled DLL in a tar file. For non-embed mode, the tar file exists on disk and is referenced by filename.

**WebNN Constraint:** No disk writes allowed (cannot create temp files).

**Current Memory Consumption:**

```
Tar file on disk: 100MB
  └─ model.dll (embedded at 4K-aligned offset)

Loading with MemoryModule:
  ├─ Step 1: mmap tar file          → 100MB (virtual, file-backed)
  ├─ Step 2: VirtualAlloc(SizeOfImage) → 100MB (physical, private)
  ├─ Step 3: Copy sections           → Now both exist in memory
  └─ Step 4: Apply relocations       → Modify copied memory

Peak physical memory: ~200MB (2x DLL size)
```

**Goal:** Reduce memory consumption to ~1x DLL size by executing directly from mmap'd memory without copying.

---

## Current Approach: MemoryModule (2x Memory)

[MemoryModule](https://github.com/fancycode/MemoryModule) is a library that loads DLLs from memory buffers. It works by manually performing the PE loader's job.

### How MemoryModule Works

From [Loading a DLL from memory tutorial](https://www.joachim-bauch.de/tutorials/loading-a-dll-from-memory/):

```cpp
// Current approach (simplified)
HMEMORYMODULE MemoryLoadLibrary(const void* dll_data) {
    // 1. Parse PE headers
    IMAGE_DOS_HEADER* dos_hdr = (IMAGE_DOS_HEADER*)dll_data;
    IMAGE_NT_HEADERS* nt_hdr = (IMAGE_NT_HEADERS*)((char*)dll_data + dos_hdr->e_lfanew);

    // 2. Allocate new memory at preferred base (or anywhere)
    size_t image_size = nt_hdr->OptionalHeader.SizeOfImage;
    void* code = VirtualAlloc(NULL, image_size,
                              MEM_RESERVE | MEM_COMMIT,
                              PAGE_READWRITE);  // ← Allocates physical memory

    // 3. Copy headers
    memcpy(code, dll_data, nt_hdr->OptionalHeader.SizeOfHeaders);

    // 4. Copy each section from source to destination
    IMAGE_SECTION_HEADER* sections = IMAGE_FIRST_SECTION(nt_hdr);
    for (int i = 0; i < nt_hdr->FileHeader.NumberOfSections; i++) {
        void* dest = (char*)code + sections[i].VirtualAddress;
        void* src = (char*)dll_data + sections[i].PointerToRawData;
        memcpy(dest, src, sections[i].SizeOfRawData);  // ← COPY happens here
    }

    // 5. Perform base relocations (if needed)
    PerformBaseRelocation(code, nt_hdr);

    // 6. Set memory protections
    FinalizeSections(code, nt_hdr);  // .text=RX, .data=RW, etc.

    // 7. Resolve imports, call DllMain
    // ...

    return (HMEMORYMODULE)code;
}
```

**Memory at this point:**
- Source buffer (`dll_data`): Still in memory (mmap'd tar file)
- Destination (`code`): New VirtualAlloc'd memory with copied sections
- **Total: 2x DLL size**

### Why Can't We Use Standard LoadLibrary?

From [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa):

> `LoadLibrary` and `LoadLibraryEx` only work with files on the filesystem.

We cannot use `LoadLibrary(path)` because:
1. WebNN constraint: No temp file creation
2. DLL is embedded in tar file, not standalone

---

## Proposed Optimization: Direct mmap Execution (1x Memory)

### Core Idea

Instead of copying PE sections, **execute directly from mmap'd tar file memory** by:
1. mmap the tar file (read-only)
2. Locate DLL at 4K-aligned offset
3. Change memory protections in-place using `VirtualProtect`
4. Only copy pages that require relocation (minimal on x64)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tar file on disk (cache.tar)                               │
│   Offset 0:    Tar header                                   │
│   Offset 4096: model.dll (4K-aligned) ← DLL starts here     │
│                  - PE headers                                │
│                  - .text section (code)                      │
│                  - .rdata section (read-only data)           │
│                  - .data section (read/write data)           │
└─────────────────────────────────────────────────────────────┘
                         ↓ mmap(tar_file)
┌─────────────────────────────────────────────────────────────┐
│ Process virtual memory                                      │
│                                                              │
│   [mmap'd region - file-backed]                             │
│   ├─ 0x7FFE0000: Tar header (READ)                          │
│   ├─ 0x7FFE1000: DLL headers (READ) ← dll_ptr               │
│   ├─ 0x7FFE2000: .text section                              │
│   │              VirtualProtect(PAGE_EXECUTE_READ)          │
│   │              → No copy if no relocation!                │
│   ├─ 0x7FFE8000: .rdata section                             │
│   │              VirtualProtect(PAGE_READONLY)              │
│   │              → No copy (already read-only)              │
│   └─ 0x7FFEA000: .data section                              │
│                  VirtualProtect(PAGE_READWRITE)             │
│                  → Copy-on-write (first write triggers copy)│
│                                                              │
│ Physical memory consumption: ~1x DLL size                   │
│   - .text: Shared with file (no private pages)             │
│   - .rdata: Shared with file (no private pages)            │
│   - .data: Private copy on write (initialized data)        │
│   - Relocations: Private copy for modified pages (minimal) │
└─────────────────────────────────────────────────────────────┘
```

### Conceptual Code

```cpp
// Proposed approach (conceptual)
HMEMORYMODULE LoadDllFromMmappedTar(const char* tar_path) {
    // 1. mmap the tar file (read-only)
    HANDLE tar_file = CreateFile(tar_path, GENERIC_READ,
                                 FILE_SHARE_READ, NULL,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    HANDLE tar_mapping = CreateFileMapping(tar_file, NULL,
                                           PAGE_READONLY, 0, 0, NULL);

    void* tar_mmap = MapViewOfFile(tar_mapping, FILE_MAP_READ, 0, 0, 0);

    // 2. Locate DLL at 4K-aligned offset
    size_t dll_offset = 4096;  // First 4K block in tar
    void* dll_ptr = (char*)tar_mmap + dll_offset;

    // 3. Parse PE headers
    IMAGE_DOS_HEADER* dos_hdr = (IMAGE_DOS_HEADER*)dll_ptr;
    IMAGE_NT_HEADERS* nt_hdr = (IMAGE_NT_HEADERS*)((char*)dll_ptr + dos_hdr->e_lfanew);

    // 4. Set memory protections on mmap'd sections (NO COPY!)
    IMAGE_SECTION_HEADER* sections = IMAGE_FIRST_SECTION(nt_hdr);
    for (int i = 0; i < nt_hdr->FileHeader.NumberOfSections; i++) {
        void* section_addr = (char*)dll_ptr + sections[i].VirtualAddress;
        DWORD section_size = sections[i].Misc.VirtualSize;

        // Determine protection based on section characteristics
        DWORD protect = 0;
        if (sections[i].Characteristics & IMAGE_SCN_MEM_EXECUTE) {
            if (sections[i].Characteristics & IMAGE_SCN_MEM_WRITE) {
                protect = PAGE_EXECUTE_READWRITE;  // Rare
            } else {
                protect = PAGE_EXECUTE_READ;  // .text section
            }
        } else if (sections[i].Characteristics & IMAGE_SCN_MEM_WRITE) {
            protect = PAGE_READWRITE;  // .data section
        } else {
            protect = PAGE_READONLY;  // .rdata section
        }

        DWORD old_protect;
        // KEY: Change protection on mmap'd memory, don't copy!
        VirtualProtect(section_addr, section_size, protect, &old_protect);
    }

    // 5. Handle relocations if base address doesn't match
    void* actual_base = dll_ptr;
    void* preferred_base = (void*)nt_hdr->OptionalHeader.ImageBase;

    if (actual_base != preferred_base) {
        // Apply relocations - this WILL trigger COW for modified pages
        ApplyRelocations(dll_ptr, nt_hdr);
    }

    // 6. Resolve imports
    ResolveImports(dll_ptr, nt_hdr);

    // 7. Call DllMain
    DllMain_t DllMain = (DllMain_t)((char*)dll_ptr + nt_hdr->OptionalHeader.AddressOfEntryPoint);
    DllMain((HINSTANCE)dll_ptr, DLL_PROCESS_ATTACH, NULL);

    return (HMEMORYMODULE)dll_ptr;
}
```

### Memory Consumption Analysis

**CRITICAL: The benefit depends on relocation density!**

Relocations trigger COW. If relocations are dense (many pages), COW negates the optimization. If relocations are sparse (few pages), the optimization works.

**Scenario 1: No relocations needed** (ideal case - DLL loads at preferred address)
```
.text section (50MB):  PAGE_EXECUTE_READ on mmap → Shared with file (0 private pages)
.rdata section (30MB): PAGE_READONLY on mmap → Shared with file (0 private pages)
.data section (20MB):  PAGE_READWRITE on mmap → COW (20MB private on first write)

Total private memory: ~20MB (only .data section)
Total virtual memory: ~100MB (file-backed)
Memory savings vs MemoryModule: 80MB (100MB - 20MB)
```

**Scenario 2: Sparse relocations** (x64 typical - like ntdll with 560 bytes)
```
.text section (50MB):
  - 560 bytes of relocations scattered across section
  - Worst case: 56 relocations × 10 pages each = 560 pages touched
  - More likely: Clustered in ~10-20 pages
  - COW on relocated pages: ~40-80KB private
  - Rest of pages: Shared with file

.rdata section (30MB): Shared with file (0 private pages)
.data section (20MB):  COW (20MB private on first write)

Total private memory: ~20MB (data) + ~40-80KB (relocated code)
Total virtual memory: ~100MB (file-backed)
Memory savings vs MemoryModule: ~80MB
```

**Scenario 3: Dense relocations** (x86-style or LLVM generates many relocations)
```
.text section (50MB):
  - 18KB of relocation data (like x86 ntdll)
  - Relocations scattered across most pages
  - COW on majority of .text pages: ~40-50MB private

.rdata section (30MB): Some relocations possible: ~10MB private
.data section (20MB):  COW: 20MB private

Total private memory: ~70-80MB
Total virtual memory: ~100MB (file-backed)
Memory savings vs MemoryModule: ~20-30MB (minimal benefit)
```

**Key Insight:**

The optimization is only worth it if LLVM-generated x64 code has **sparse relocations** like system DLLs. If LLVM generates **dense relocations**, the COW overhead negates most benefits.

**This MUST be measured empirically** - cannot assume without testing.

---

## Technical Deep Dive

### Why 2x Memory with MemoryModule

From [Reflective DLL Loading](https://trustedsec.com/blog/loading-dlls-reflections) and [PE File Loading Process](https://deepwiki.com/fancycode/MemoryModule/2.1-pe-file-loading-process):

**Standard MemoryModule Process:**
1. **VirtualAlloc** allocates physical memory (`MEM_COMMIT`)
2. **memcpy** copies all sections from source to destination
3. Both buffers remain in memory until source is freed

**Why VirtualAlloc instead of mmap:**
- Need executable memory (`PAGE_EXECUTE_READ`)
- Need to apply relocations (modify code)
- MemoryModule assumes source buffer might be temporary

**The Copy:**
```cpp
// From MemoryModule source
for each section:
    dest = VirtualAlloc(base + section->VirtualAddress, size, MEM_COMMIT, PAGE_READWRITE);
    memcpy(dest, source + section->PointerToRawData, section->SizeOfRawData);
    // ↑ This copies data from source buffer to new memory
```

### How Standard LoadLibrary Works

From [CreateFileMapping with SEC_IMAGE documentation](https://learn.microsoft.com/en-us/answers/questions/1025940/createfilemapping-function-with-the-sec-image-flag):

**Windows PE Loader Process:**
```cpp
// Simplified internal loader behavior
LoadLibrary(path) internally does:
    1. CreateFile(path) → HANDLE file
    2. CreateFileMapping(file, SEC_IMAGE) → HANDLE mapping
       // SEC_IMAGE tells Windows to validate PE structure
    3. MapViewOfFile(mapping) → void* base
       // OS maps sections according to PE headers
    4. Apply relocations (if needed)
    5. Resolve imports
    6. Call DllMain
```

**Key: SEC_IMAGE flag**
- Validates PE structure
- Automatically sets correct page protections (.text=RX, .data=RW)
- Uses file-backed mapping (not a copy)
- Shares .text section across processes

**Memory consumption**: ~1x DLL size (code sections shared, data sections private via COW)

**Why we can't use this:**
- `SEC_IMAGE` requires a valid file `HANDLE` from `CreateFile`
- Cannot use `INVALID_HANDLE_VALUE` (memory buffer) with `SEC_IMAGE` (security restriction)
- Our DLL is embedded in tar, not a standalone file

### RIP-Relative Addressing on x64

From [x64 DLL relocations analysis](http://www.nynaeve.net/?p=192):

**What is RIP:**
- RIP = Register Instruction Pointer (64-bit instruction pointer in x64)
- Equivalent to EIP in x86, IP in 16-bit

**RIP-Relative Addressing Mode:**
```asm
; x86 (32-bit) - Absolute addressing
mov eax, [0x12345678]          ; Hardcoded absolute address
; Problem: If DLL loads at different base, address is wrong
; Solution: Base relocation (patch all absolute addresses)

; x64 - RIP-relative addressing
mov eax, [rip + 0x1000]        ; Relative to current instruction
; Advantage: Offset is constant regardless of load address
; No relocation needed!
```

**Real-world Impact:**
- **Windows Vista x86 ntdll.dll**: 18,092 bytes of relocations
- **Windows Vista x64 ntdll.dll**: 560 bytes of relocations (32x smaller!)

**Why this matters:**
- Fewer relocations = fewer pages to modify
- Fewer modified pages = less COW overhead
- Most x64 code is position-independent by default

From [Position Independent Code on x64](https://eli.thegreenplace.net/2011/11/11/position-independent-code-pic-in-shared-libraries-on-x64):

> AMD did the right thing with AMD64 and added RIP (instruction pointer) relative addressing, which means you get position independent code pretty much for free.

**For our use case:**
- LLVM-generated x64 code should have minimal relocations
- Most code references use RIP-relative addressing
- Applying relocations affects only small number of pages

### Copy-on-Write Behavior

**CRITICAL: Files on disk are NEVER modified by relocations!**

When relocations are applied, they modify **in-memory copies only**. The DLL file (or tar file) on disk remains unchanged. This is guaranteed by the OS's copy-on-write mechanism.

From [Windows Memory Management](https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-using-large-pages) and [Shared Memory in DLLs](https://learn.microsoft.com/en-us/windows/win32/dlls/using-shared-memory-in-a-dynamic-link-library):

**Copy-on-Write (COW) Mechanism:**
```
Initial state:
  Process A maps file → Page backed by file (shared)
  Process B maps file → Same physical page (shared)

After Process A writes to page:
  Process A → Private copy of page (modified)
  Process B → Still using original file-backed page (shared)
```

**Triggers for COW:**
1. **First write** to a page mapped with `PAGE_READWRITE`
2. **Modification** during relocation (changing code bytes)

**Does not trigger COW:**
- Changing protection from `PAGE_READONLY` to `PAGE_EXECUTE_READ` (no modification)
- Reading from pages
- Multiple processes reading same file-backed pages

**Expected behavior for our approach:**
```
.text section:
  - mmap'd as PAGE_READONLY
  - VirtualProtect → PAGE_EXECUTE_READ (just changes permission bits)
  - No modification → No COW → Shared with file ✓
  - Unless relocations needed → Modifies bytes → COW for those pages

.rdata section:
  - mmap'd as PAGE_READONLY
  - VirtualProtect → PAGE_READONLY (no change)
  - No modification → No COW → Shared with file ✓

.data section:
  - mmap'd as PAGE_READONLY
  - VirtualProtect → PAGE_READWRITE
  - First write by program → COW triggers
  - Private copy created (expected for writable data)
```

---

## Implementation Approach

### Phase 1: Research & Validation

1. **Measure relocation counts** in LLVM-generated DLLs:
   ```bash
   # Build sample DLL
   cmake --build ../../build/onnx-hipdnn-ep --target sample_model

   # Check relocations
   dumpbin /RELOCATIONS sample_model.dll
   dumpbin /RELOCATIONS sample_model.dll | find /C "OFFSET"

   # Expected: Very low count on x64 (similar to 560 bytes for ntdll)
   ```

2. **Test VirtualProtect on file mappings:**
   ```cpp
   // Test program
   HANDLE file = CreateFile("test.dll", GENERIC_READ, FILE_SHARE_READ, ...);
   HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY, ...);
   void* view = MapViewOfFile(mapping, FILE_MAP_READ, ...);

   // Try changing protection
   DWORD old_protect;
   BOOL result = VirtualProtect(view, size, PAGE_EXECUTE_READ, &old_protect);
   // Does this work? Does it trigger COW?

   // Measure private bytes before/after
   PROCESS_MEMORY_COUNTERS pmc;
   GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
   // Check pmc.WorkingSetSize for actual memory consumption
   ```

3. **Verify 4K alignment** in tar file generation:
   ```bash
   # Ensure DLL starts at 4K boundary in tar
   tar tvf cache.tar
   # Check offset of model.dll - should be multiple of 4096
   ```

### Phase 2: Prototype

Create modified MemoryModule that:
1. Accepts mmap'd pointer instead of copying
2. Uses `VirtualProtect` on existing pages
3. Handles relocations in-place (COW acceptable for modified pages)
4. Tracks actual memory consumption

### Phase 3: Benchmark

Compare memory consumption:
- Current approach (MemoryModule with copy)
- Optimized approach (direct mmap execution)
- Measure: Private bytes, working set, virtual memory

Test with various DLL sizes: 10MB, 100MB, 500MB

### Phase 4: Decision

Based on:
- Actual memory savings
- Implementation complexity
- Relocation overhead on x64
- Compatibility across Windows versions

---

## Open Questions

### Q1: Does VirtualProtect on file-mapped memory trigger COW?

**Hypothesis:** Changing protection without modifying data should NOT trigger COW.

**Test needed:** Empirical measurement with sample DLL.

**References:**
- [Windows Memory Management](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile)
- [VirtualProtect documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualprotect)

### Q2: How many relocations do LLVM-generated x64 DLLs have?

**Hypothesis:** LLVM uses RIP-relative addressing, should have minimal relocations similar to system DLLs.

**Test needed:** Build sample DLL and measure with `dumpbin /RELOCATIONS`.

**Expected:** < 1KB of relocations for typical model DLL.

### Q3: Can we apply relocations in-place on mmap'd memory?

**Scenario:** DLL loads at different address than preferred base.

**Options:**
- **Option A:** Apply relocations in-place → Triggers COW for modified pages (acceptable if few pages)
- **Option B:** Fall back to MemoryModule copy approach (safe fallback)

**Decision criteria:** If relocations affect < 5% of pages, in-place is acceptable.

### Q4: Linux mmap vs Windows CreateFileMapping behavior differences?

**Need to verify:**
- Does `mprotect` (Linux) behave same as `VirtualProtect` (Windows)?
- Does Linux COW behavior match Windows for this use case?

**Cross-platform considerations:**
```cpp
#ifdef _WIN32
    // CreateFileMapping + MapViewOfFile + VirtualProtect
#else
    // mmap + mprotect
#endif
```

---

## References

### Windows API Documentation

- [LoadLibraryEx function](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa) - Standard DLL loading API
- [CreateFileMapping function](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createfilemappinga) - File mapping API
- [MapViewOfFile function](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile) - Map file to memory
- [VirtualProtect function](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualprotect) - Change memory protection
- [CreateFileMapping with SEC_IMAGE](https://learn.microsoft.com/en-us/answers/questions/1025940/createfilemapping-function-with-the-sec-image-flag) - SEC_IMAGE flag discussion
- [NtCreateSection function](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntcreatesection) - Low-level section API

### In-Memory DLL Loading

- [MemoryModule GitHub](https://github.com/fancycode/MemoryModule) - Reference implementation
- [Loading a DLL from memory tutorial](https://www.joachim-bauch.de/tutorials/loading-a-dll-from-memory/) - Detailed explanation
- [Reflective DLL Loading](https://trustedsec.com/blog/loading-dlls-reflections) - Security perspective
- [PE File Loading Process](https://deepwiki.com/fancycode/MemoryModule/2.1-pe-file-loading-process) - Technical details
- [Reflective PE Loading](https://skr1x.github.io/reflective-loading-portable-executable-memory/) - Implementation guide

### Position-Independent Code

- [RIP-relative addressing in x64](http://www.nynaeve.net/?p=192) - Analysis of x64 relocations (ntdll example)
- [Position Independent Code on x64](https://eli.thegreenplace.net/2011/11/11/position-independent-code-pic-in-shared-libraries-on-x64) - PIC explanation
- [Position Independent Code and x86-64](https://www.technovelty.org/c/position-independent-code-and-x86-64-libraries.html) - Technical details

### Memory Management

- [Windows Memory Mapped File IO](https://www.jeremyong.com/winapi/io/2024/11/03/windows-memory-mapped-file-io/) - Modern guide
- [Managing Memory-Mapped Files](https://www.labri.fr/perso/betrema/winnt/manamemo.html) - Detailed reference
- [Using Shared Memory in DLLs](https://learn.microsoft.com/en-us/windows/win32/dlls/using-shared-memory-in-a-dynamic-link-library) - Shared memory patterns

### Advanced Topics

- [Phantom DLL Hollowing](https://www.cyberark.com/resources/threat-research-blog/masking-malicious-memory-artifacts-part-i-phantom-dll-hollowing-2) - Advanced memory mapping techniques
- [NtCreateSection code injection](https://www.ired.team/offensive-security/code-injection-process-injection/ntcreatesection-+-ntmapviewofsection-code-injection) - Low-level section API usage

---

## Decision Status

**Current Status:** Deferred

**Rationale:**
- Current MemoryModule approach is simple and proven
- 2x memory overhead acceptable for initial deployment
- Optimization adds significant complexity
- Need to validate assumptions with real measurements

**Revisit When:**
1. **Large models** (>500MB DLL) make memory consumption critical
2. **WebNN deployment** has strict memory limits in production
3. **Profiling** shows memory consumption is a bottleneck vs computation time
4. **Customer feedback** indicates memory is limiting factor

**Next Action:** Measure relocation counts in LLVM-generated DLLs to validate feasibility.
