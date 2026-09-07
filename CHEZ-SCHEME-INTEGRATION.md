# Chez Scheme Integration Status

**Status:** ✅ COMPLETE

## What Works

- ✅ ChezScheme builds as CMake ExternalProject
- ✅ Both petite.boot (2.1M) and scheme.boot (1.2M) embedded as C arrays
- ✅ Chez C API properly linked (libkernel.a, liblz4.a, libz.a)
- ✅ Runtime initialization succeeds
- ✅ Scheme code execution works: `(* 6 7)` → `42`
- ✅ MLIR pass infrastructure using tablegen
- ✅ Zero external file dependencies - single DLL deployment

## Verification

```bash
$ /workspace/hip-ep/build/hip-ep-1/bin/hip-mlir-opt --scheme-print test.mlir

Initializing Chez Scheme runtime...
  Version: 10.5.0-pre-release.1
  Petite boot: 2165162 bytes
  Scheme boot: 1163174 bytes
Scheme test: (* 6 7) = 42
Chez Scheme runtime initialized successfully!
```

## Key Technical Details

### Boot File Loading Order

**CRITICAL:** petite.boot MUST be registered before scheme.boot.

```cpp
Sscheme_init(nullptr);
Sregister_boot_file_bytes("petite.boot", petite_boot_data, petite_boot_size);  // FIRST
Sregister_boot_file_bytes("scheme.boot", scheme_boot_data, scheme_boot_size);  // SECOND
Sbuild_heap("hip-mlir-opt", nullptr);
```

**Why:** ChezScheme's boot loading mechanism (`c/scheme.c:load()`):
- First registered boot becomes "base boot" (loaded with `base=1` flag)
- Base boot's 3rd object initializes `S_G.base_rtd` (base record-type descriptor)
- Subsequent boots (loaded with `base=0`) depend on `S_G.base_rtd` being set
- scheme.boot contains `fasl_type_base_rtd` references that require initialized `S_G.base_rtd`
- petite.boot defines `S_G.base_rtd`, scheme.boot uses it

### Investigation Findings

**Bug in ChezScheme** (`c/fasl.c:906-907`):
```c
case fasl_type_base_rtd: {
    ptr rtd;
    if ((rtd = S_G.base_rtd) == Sfalse) {
      if (!Srecordp(rtd)) S_error_abort("S_G.base-rtd has not been set");  // Logic error
    }
```

When `rtd == Sfalse`, the inner `if (!Srecordp(rtd))` check always triggers because `Sfalse` is not a record. This makes the error message confusing - the actual issue is that `base_rtd` wasn't initialized, not that it's an invalid record.

## Next Steps

1. ✅ ~~Integrate ChezScheme~~ DONE
2. ✅ ~~Initialize Scheme runtime~~ DONE
3. ✅ ~~Create MLIR pass infrastructure~~ DONE
4. 🔜 Add wcy123/rime library (Common Lisp-style loop macro)
5. 🔜 Create pattern DSL using Scheme define-syntax
6. 🔜 Implement MLIR transformation patterns in Scheme

## Files

- `lib/Dialect/Hipsr/Scheme/CMakeLists.txt` - Build and embed boot files
- `lib/Dialect/Hipsr/Scheme/SchemeRuntime.cpp` - Chez runtime initialization
- `lib/Dialect/Hipsr/Scheme/SchemePrintPass.cpp` - Example MLIR pass
- `include/hip/Dialect/Hipsr/Transforms/Passes.td` - Pass registration

## References

- ChezScheme repo: https://github.com/cisco/ChezScheme
- ChezScheme User's Guide: https://cisco.github.io/ChezScheme/csug/csug.html
- Foreign Interface: https://cisco.github.io/ChezScheme/csug/foreign.html
- Main.c reference: https://github.com/cisco/ChezScheme/blob/main/c/main.c
- rime library: https://github.com/wcy123/rime

## PR

https://github.com/ROCm/hip-ep/pull/912
