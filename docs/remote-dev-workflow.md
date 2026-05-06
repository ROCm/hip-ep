<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Remote Development Workflow

This project's full build + GPU test loop runs on a **remote Windows host with an AMD gfx1151 iGPU**, driven over SSH from a developer laptop. This doc collects the SSH/cmd-shell gotchas, the "must build on gfx1151" rationale, and the small set of conventions that make the loop fast.

User-specific facts (your hostname, username, paths) belong in your own Claude memory or your shell config — not here. This doc covers everything that's the **same for every developer** on this project.

## Why the remote host (gfx1151), not your laptop

The dynamic-sequence-length runtime path SEGVs on gfx1151 if a compiler pass routes a host-side `memref.store` into the GPU pool. On gfx1150 (and most other arches) `hipMalloc` happens to return UMA-mapped host-accessible memory, so the same antipattern silently works. **Local LIT or pytest runs on a non-gfx1151 box give a false sense of progress** — they don't exercise the runtime path that fails in production.

Concrete consequences:

- **Build, install, and run tests on the gfx1151 remote.** Local-machine test passes are not authoritative.
- **The CLAUDE.md "gfx1151 dynseqlen host-scalar SEGV" gotcha** explains the actual fix (separate runtime-owned host scratch buffer); this doc only explains why the architecture matters.
- **Vulkan baseline numbers and the 8B Llama DYN-vs-FIX baseline in CLAUDE.md were collected on gfx1151.** Running them on a different arch will give different numbers and is not comparable.

## SSH to a Windows host with default OpenSSH

The OpenSSH server on Windows defaults to `cmd.exe` as the shell — no cygwin, no MSYS, no `bash -lc` wrapper. Commands that come naturally on a Linux box need translation:

| Linux/bash | Windows/cmd over SSH |
|---|---|
| `cmd1 && cmd2` | `cmd1 & cmd2` (cmd's `&` is unconditional; use `&&` for short-circuit) |
| `cmd1; cmd2` | `cmd1 & cmd2` |
| `cat`, `head`, `tail`, `grep`, `sed`, `awk` | not available — use `type`, `findstr`, or transfer the file via the SMB share and read it locally |
| `'single quotes'` | only `"double quotes"` — no single-quote string literal |
| `~` | `%USERPROFILE%` |
| Forward-slash paths | both work for filesystem paths, but `cd /d` is needed to switch drives |

Practical patterns that come up constantly:

```bash
# Run a build over SSH. Quote the WHOLE remote command so cmd sees one string.
ssh user@host "cd /d C:\\path\\to\\repo && python build.py"

# Merge stderr into stdout BEFORE redirecting, so the SSH transport carries everything.
ssh user@host "cmd args 2>&1" > local_log.txt

# Embed Python / cmake snippets that contain backslashes — use forward slashes
# inside the embedded string to avoid double-escape hell.
ssh user@host "python -c \"import os; print(os.path.join('C:/foo', 'bar'))\""
```

**`cmd` swallows trailing whitespace inside a chained `set X=value && next` literally** — the value becomes `value ` (with a trailing space). Always quote: `set "X=value" && next`. This trap has caused silent CPU-fallback regressions in this project (`THEROCK_DIST` ending with a space → lld-link can't find `amdhip64.lib` → EP falls back to CPU and tests still pass cosine=1.0 because they compare CPU-vs-CPU). PowerShell does not have this issue.

## SMB-share + SSH split: what to do where

If your laptop maps the remote home directory as a network drive (e.g. via SMB), you get the best of both worlds:

- **Through the share** (e.g. `Z:\`): use your local editor, `Read`/`Edit`/`Write` tools, `git diff`, etc. Zero SSH per file. Fast.
- **Through SSH**: use only for things that **must execute on the remote**: build, run tests, run `model_benchmark.exe`, `tasklist`, etc.

This split is the single biggest workflow speed-up. SSH per command is slow; share-mounted file ops are not.

**Caveat: the share copies bytes over the network.** A `Read` of a 100 MB log over SMB is slower than `findstr` over SSH. For grep-style scans inside huge files (build logs, IR dumps), prefer `ssh ... findstr ...` and only `Read` the result.

## Driving GPU benchmarks from a remote shell

Rules from CLAUDE.md ("Python Performance Tests") apply, plus two SSH-specific gotchas:

1. **Don't background the SSH command** if it owns a `model_benchmark.exe` child. Killing the SSH session leaves the child running on the GPU, blocking the next benchmark. Always foreground with a long timeout.
2. **Always quote the whole remote command**, including the env var assignments. `ssh host "set X=Y && tool ..."` runs `set` and `tool` in one cmd; without the outer quotes only the local shell sees the assignment.

Verify the GPU is idle before launching:

```bash
ssh user@host "tasklist | findstr /I model_benchmark"
# (no output = idle)
```

Kill stragglers:

```bash
ssh user@host "taskkill /F /IM model_benchmark.exe"
```

## Conda env on the remote: which `python` runs

On a typical remote setup, **two Python interpreters** coexist on PATH:

1. The miniforge/conda BASE Python (usually `C:\tools\miniforge3\python.exe` or similar)
2. The project conda env Python (e.g. `C:\Users\<user>\envs\hipdnn-ep\python.exe`)

`build.py` requires the project env Python (its dependencies live there). If the BASE Python sits earlier in PATH, a bare `python build.py` over SSH silently runs the wrong interpreter and dies on missing imports.

**Always invoke the env Python by absolute path** when running over SSH:

```bash
ssh user@host "cd /d C:\\path\\to\\repo && C:\\Users\\<user>\\envs\\hipdnn-ep\\python.exe build.py"
```

Sub-tools that the env adds to PATH (lit, ninja, sccache, gh, ctest) work without a prefix because they're env-installed and not shadowed by BASE.

## Git on Windows: cygwin contamination warning

If `install/build/CMakeCache.txt` contains `GIT_EXECUTABLE:FILEPATH=C:/cygwin/bin/git.exe` (left over from a prior cygwin-era configure), `FetchContent` dependency clones will fail at fetch time:

```
git-remote-https.exe: error while loading shared libraries: cygcrypto-3.dll: cannot open shared object file
```

This is hard to debug because the rest of the build looks normal. Fix: wipe `install/build/` and reconfigure so cmake re-detects the system `git.exe` (typically `C:\Program Files\Git\cmd\git.exe`).

## OGA build (`build.py --build-oga`) — defensive measures in place

`build_oga()` in `build.py` defends against several non-obvious snags. Each defense has an inline comment explaining the snag it protects against. Quick reference:

| Defense | Snag it protects against |
|---|---|
| `PYBIND11_FINDPYTHON=ON` cmake flag | OGA vendors pybind11 v2.13.6 which uses removed `distutils.sysconfig` on Python 3.12+ |
| `Python_EXECUTABLE=<sys.executable>` (forward-slash path, no `3` suffix) | Stops cmake from auto-picking the wrong-ABI Python from PATH |
| Tolerate `PyPackageBuild` non-zero exit, then run `python -m pip wheel` manually | OGA's wheel-packaging ninja step emits a malformed `cmd /C "... && -m pip wheel ."` (missing `python` before `-m`). The artifacts you actually need are built BEFORE this failure |
| Copy `onnxruntime.dll` + `onnxruntime_providers_shared.dll` into `install/dist/bin/` after install | Otherwise Windows DLL search picks `C:\Windows\System32\onnxruntime.dll` (often v1.17 = API 17), and OGA built against ORT 1.24 (API 24) crashes immediately |
| `setuptools`, `wheel`, `packaging`, `requests` in `environment.yml` | OGA's dep fetcher imports `requests`; without it, build aborts before reaching cmake |

If `--build-oga` fails on a fresh env, walk through `build_oga()`'s comments in order — the failure is almost always one of these.
