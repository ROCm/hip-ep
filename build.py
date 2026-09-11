#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Native cross-platform build driver for onnx-hipdnn-ep.

Structured after ONNX Runtime's tools/ci_build/build.py (logging, run_subprocess,
update_submodules, --config/--cmake_generator/--cmake_extra_defines/
--skip_submodule_sync conventions) but deliberately plain: it ensures
submodules, sets up the platform compiler environment, picks the GPU arch, runs
the cmake configure/build/install for this project's artifacts -- the MorphiZen
EP shared library (libonnxruntime_morphizen_ep.so / .dll) plus the HIP MLIR
tools -- and (on Linux) bundles the runtime .so into a self-contained install/.

Most C++ dependency acquisition is delegated to cmake/deps.cmake:
LLVM/MLIR/LLD (from-source fallback), protobuf/flatbuffers (from source), ONNX
Runtime (official release download), and the TheRock ROCm SDK (auto-download).
When --enable_rocmlirtriton is selected, this driver additionally builds and
installs the fetched static rocMLIR::rockCompiler package before reconfiguring
hip-ep against it. CI can inject cached prefixes via --cmake_prefix_path. This
script does not build onnxruntime-genai (OGA).

Layout (siblings of the repo, matching docs/quick_start.md and CI):
    <workspace>/<repo>/          project source (this repo)
    <workspace>/build/<repo>/    cmake build tree (+ _therock, _deps)
    <workspace>/install/         install prefix (self-contained on Linux)

Usage:
    python build.py                       # real build, auto-detect arch
    python build.py --mock                # mock runtime (no GPU/HIP/TheRock)
    python build.py --hip_arch gfx1151    # explicit GPU arch
    python build.py --cmake_prefix_path "/x/llvm-install;/x/ort-install"
    python build.py --clean               # remove build/ + install/
"""

import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
PROJECT_NAME = REPO.name
IS_WINDOWS = os.name == "nt"

logging.basicConfig(format="[build] %(message)s", level=logging.INFO)
log = logging.getLogger("hipdnn-ep.build")


# ---------------------------------------------------------------------------
# Errors + subprocess helper (ONNX Runtime style)
# ---------------------------------------------------------------------------


class BuildError(Exception):
    pass


def run_subprocess(args, cwd=None, env=None, capture_stdout=False):
    """Run a command (sequence of str/Path), echoing it. Raises on failure."""
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")
    args = [str(a) for a in args]
    my_env = os.environ.copy()
    if env:
        my_env.update(env)
    log.info(" ".join(shlex.quote(a) for a in args))
    return subprocess.run(
        args,
        cwd=cwd,
        env=my_env,
        check=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        text=True if capture_stdout else None,
    )


def step(msg):
    bar = "=" * 68
    print(f"\n{bar}\n> {msg}\n{bar}", flush=True)


def have_tool(name):
    return shutil.which(name) is not None


def _check_python_version():
    if sys.version_info[:2] < (3, 8):
        raise BuildError(f"Python 3.8+ required; found {sys.version.split()[0]}")


# ---------------------------------------------------------------------------
# Submodules + platform toolchain
# ---------------------------------------------------------------------------


def update_submodules():
    # MorphiZen is vendored in-tree as a git subtree, so there are normally no
    # submodules to initialize; this stays a generic no-op guard for any future
    # submodule and returns early when none are uninitialized.
    log.info("Checking git submodules ...")
    r = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    uninitialized = any(
        ln.strip().startswith("-") for ln in r.stdout.splitlines() if ln.strip()
    )
    if not uninitialized:
        log.info("  submodules OK.")
        return
    log.info("  initializing submodules ...")
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=str(REPO))
    run_subprocess(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO)
    )


def default_generator():
    # Like ONNX Runtime: on Windows default to the Visual Studio generator so
    # CMake locates MSVC itself (no vcvarsall sourcing); Ninja elsewhere.
    return "Visual Studio 17 2022" if IS_WINDOWS else "Ninja"


def check_toolchain(generator):
    for tool in ("cmake", "git"):
        if not have_tool(tool):
            raise BuildError(f"required tool not found on PATH: {tool}")
    if generator == "Ninja" and not have_tool("ninja"):
        raise BuildError("Ninja generator selected but 'ninja' is not on PATH")
    if IS_WINDOWS:
        # The Visual Studio generator finds MSVC on its own. For Ninja we rely on
        # the caller having loaded the MSVC environment (run from an "x64 Native
        # Tools Command Prompt for VS"), exactly as ONNX Runtime's build expects.
        if generator == "Ninja" and not have_tool("cl"):
            log.warning(
                "cl.exe not on PATH; for the Ninja generator on Windows run from "
                "an 'x64 Native Tools Command Prompt for VS' (or pass "
                "--cmake_generator 'Visual Studio 17 2022')."
            )
    elif not (have_tool("c++") or have_tool("g++") or have_tool("clang++")):
        raise BuildError("no C++ compiler (c++/g++/clang++) on PATH")


def _msvc_build_environment():
    """Return an x64 MSVC environment for an isolated Ninja build."""
    if not IS_WINDOWS or (os.environ.get("LIB") and os.environ.get("INCLUDE")):
        return {}

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if not vswhere.exists():
        raise BuildError(
            "Visual Studio vswhere.exe was not found; install the Desktop "
            "development with C++ workload"
        )
    result = subprocess.run(
        [
            str(vswhere),
            "-all",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    attempted = []
    for path_line in result.stdout.splitlines():
        install_path = Path(path_line.strip())
        vcvars = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars.exists():
            continue
        attempted.append(str(vcvars))
        # shell=True is intentional on Windows: invoking a .bat file and then
        # reading its mutated environment must happen in one cmd.exe process.
        vcvars_result = subprocess.run(
            f'"{vcvars}" >nul && set',
            shell=True,
            capture_output=True,
            text=True,
        )
        if vcvars_result.returncode != 0:
            continue
        environment = {}
        for line in vcvars_result.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                key, value = line.split("=", 1)
                environment[key] = value
        if environment.get("LIB") and environment.get("INCLUDE"):
            return environment
    raise BuildError(
        "No Visual Studio installation produced a usable x64 build "
        f"environment; tried: {', '.join(attempted) or 'none'}"
    )


# ---------------------------------------------------------------------------
# GPU architecture detection (no ROCm/TheRock needed)
#   Linux:   amdgpu kernel sysfs (/sys/class/kfd)
#   Windows: the driver-provided HIP runtime (amdhip64_*.dll), via ctypes
# ---------------------------------------------------------------------------


def _detect_hip_arch_windows():
    """Read gcnArchName from the driver-provided HIP runtime via ctypes.

    Loads amdhip64_*.dll (shipped by the GPU driver in System32 -- no ROCm or
    TheRock needed) and reads gcnArchName through the fixed hipDeviceProp_t
    offsets, mirroring LLVM's offload-arch. Returns a gfx string or None.
    """
    import ctypes

    # gcnArchName offset matches HIP's R0600 (6.x+) hipDeviceProp_t layout.
    class _PropR0600(ctypes.Structure):
        _fields_ = [
            ("_pad", ctypes.c_char * 1160),
            ("gcnArchName", ctypes.c_char * 256),
            ("_pad2", ctypes.c_char * 56),
        ]

    for dll in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
        try:
            lib = ctypes.WinDLL(dll)
        except OSError:
            continue
        try:
            get_count = lib.hipGetDeviceCount
            get_props = lib.hipGetDevicePropertiesR0600
        except AttributeError:
            continue
        count = ctypes.c_int(0)
        if get_count(ctypes.byref(count)) != 0 or count.value <= 0:
            continue
        prop = _PropR0600()
        if get_props(ctypes.byref(prop), 0) != 0:
            continue
        name = prop.gcnArchName.decode("ascii", "ignore").strip()
        if name:
            return name.split(":", 1)[0]  # drop feature flags (gfx1151:xnack-)
    return None


def _detect_hip_arch_linux():
    """Read gfx_target_version from /sys/class/kfd topology (no ROCm needed).

    gfx_target_version encodes the arch as major*10000 + minor*100 + step
    (e.g. 110501 -> gfx1151, 90010 -> gfx90a). The CPU node reports 0.
    Returns a gfx string or None.
    """
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    if not nodes.is_dir():
        return None
    for props in sorted(nodes.glob("*/properties")):
        try:
            text = props.read_text()
        except OSError:
            continue
        ver = None
        for ln in text.splitlines():
            if ln.startswith("gfx_target_version "):
                ver = int(ln.split()[1])
                break
        if not ver:
            continue
        major, minor, step = ver // 10000, (ver // 100) % 100, ver % 100
        return f"gfx{major}{minor:x}{step:x}"
    return None


def detect_hip_arch():
    """Detect the local AMD GPU's gfx arch (TheRock-free). Returns gfx or None."""
    return _detect_hip_arch_windows() if IS_WINDOWS else _detect_hip_arch_linux()


# ---------------------------------------------------------------------------
# Configure / build / install (deps resolved by cmake/deps.cmake)
# ---------------------------------------------------------------------------


def generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock):
    step(f"Configure (HIP_ARCHITECTURES={hip_arch or 'mock'})")
    cmd = ["cmake", "-S", str(REPO), "-B", str(build_dir), "-G", args.cmake_generator]
    # The Visual Studio (multi-config) generator needs the target platform via -A
    # (Ninja/Makefiles take the build type via CMAKE_BUILD_TYPE only).
    if args.cmake_generator.startswith("Visual Studio"):
        cmd += ["-A", "x64"]
    cmd += [
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DCMAKE_INSTALL_PREFIX={args.install_dir}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    if IS_WINDOWS and not mock and not args.skip_wheel:
        cmd.append("-DBUILD_PYTHON_WHEEL=ON")
    if prefix_paths:
        # CMAKE_PREFIX_PATH always uses ';' as the list separator.
        cmd.append("-DCMAKE_PREFIX_PATH=" + ";".join(prefix_paths))
    if mock:
        cmd.append("-DBUILD_MOCK_RUNTIME=ON")
    else:
        cmd.append("-DBUILD_MOCK_RUNTIME=OFF")
        cmd.append(f"-DHIP_ARCHITECTURES={hip_arch}")
        if args.therock_dist:
            cmd.append(f"-DTHEROCK_DIST={args.therock_dist}")
    if args.enable_rocmlirtriton:
        # The first configure populates rocmlirTriton and resolves TheRock.
        # build_rocmlirtriton() then installs rockCompiler and this project is
        # configured a second time with the package available.
        cmd.append(
            "-DHIPDNN_EP_ENABLE_ROCMLIRTRITON="
            + ("ON" if args.rocmlirtriton_ready else "OFF")
        )
    if args.shared_llvm:
        # rocmlirTriton has already been cloned and built, so hip-ep neither
        # fetches it again nor builds a second copy of the same LLVM tree.
        cmd += [
            "-DHIPDNN_EP_FETCH_ROCMLIRTRITON=OFF",
            f"-DHIPDNN_ROCMLIRTRITON_SOURCE_DIR={args.rocmlirtriton_source_dir}",
            "-DHIPDNN_EP_EMBED_LLVM=OFF",
        ]
        cmd += [f"-D{name}={path}" for name, path in args.shared_llvm_dirs.items()]
    if IS_WINDOWS:
        cmd.append("-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded")
    if have_tool("sccache"):
        cmd += [
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ]
    cmd.append(f"-DPython3_EXECUTABLE={sys.executable}")
    # Arbitrary -D escape hatch (ORT-style --cmake_extra_defines KEY=VALUE).
    for define in args.cmake_extra_defines:
        cmd.append(f"-D{define}")
    run_subprocess(cmd)


def _dep_entry(name):
    """Return the (url, revision) pinned for one entry of cmake/deps.txt."""
    deps = REPO / "cmake" / "deps.txt"
    for line in deps.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) >= 3 and fields[0] == name:
            return fields[1], fields[2]
    raise BuildError(f"{name} is not pinned in {deps}")


def fetch_rocmlirtriton(build_dir):
    """Clone the pinned rocmlirTriton ahead of any CMake configure.

    The shared-LLVM flow builds rocmlirTriton before hip-ep is configured at
    all, so the checkout cannot come from hip-ep's own configure step the way
    it does in the default flow.
    """
    url, revision = _dep_entry("rocmlirTriton")
    source_dir = Path(build_dir) / "_deps" / "rocmlirtriton-src"
    llvm_cmakelists = (
        source_dir / "external" / "llvm-project" / "llvm" / "CMakeLists.txt"
    )

    if not (source_dir / ".git").exists():
        step(f"Clone rocmlirTriton {revision}")
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run_subprocess(["git", "clone", url, str(source_dir)])
    if _git_head(source_dir) != revision:
        step(f"Check out rocmlirTriton {revision}")
        run_subprocess(["git", "-C", str(source_dir), "fetch", "--tags", "origin"])
        run_subprocess(["git", "-C", str(source_dir), "checkout", "--force", revision])

    if not llvm_cmakelists.exists():
        raise BuildError(
            f"rocmlirTriton checkout at {source_dir} is missing "
            "external/llvm-project; the vendored subtree did not come down "
            "with the clone"
        )
    return source_dir


def _git_head(source_dir):
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def shared_llvm_package_dirs(rock_build):
    """CMake package dirs for the LLVM/MLIR/LLD that rocmlirTriton just built.

    cmake/triton.cmake emits MLIR's and LLD's build-tree configs under the
    top-level binary dir and LLVM's under external/llvm-project, so the three
    do not share a parent.
    """
    rock_build = Path(rock_build)
    dirs = {
        "LLVM_DIR": rock_build
        / "external"
        / "llvm-project"
        / "llvm"
        / "lib"
        / "cmake"
        / "llvm",
        "MLIR_DIR": rock_build / "lib" / "cmake" / "mlir",
        "LLD_DIR": rock_build / "lib" / "cmake" / "lld",
    }
    for name, path in dirs.items():
        if not path.is_dir():
            raise BuildError(
                f"the rocmlirTriton build did not produce {name} at {path}"
            )
    return dirs


def _cmake_cache_value(build_dir, name):
    """Read one typed entry from CMakeCache.txt."""
    cache = Path(build_dir) / "CMakeCache.txt"
    if not cache.exists():
        raise BuildError(f"CMake cache not found: {cache}")
    prefix = f"{name}:"
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1].strip()
    raise BuildError(f"{name} not found in {cache}")


def _missing_rockcompiler_lib_targets(rock_build):
    """CMake targets for .libs the fat package installs but has not built yet."""
    install_script = (
        Path(rock_build) / "mlir" / "tools" / "rocmlir-lib" / "cmake_install.cmake"
    )
    if not install_script.exists():
        return []
    missing = []
    for match in re.findall(
        r'"([^"]+\.lib)"', install_script.read_text(errors="ignore")
    ):
        lib_path = Path(match)
        if lib_path.exists():
            continue
        missing.append(lib_path.stem)
    seen = set()
    unique = []
    for name in missing:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _ninja_target_names(build_dir):
    """Return the Ninja target name set, or empty if Ninja is unavailable."""
    if not have_tool("ninja"):
        return set()
    result = subprocess.run(
        ["ninja", "-C", str(build_dir), "-t", "targets"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        name = line.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return names


def build_rocmlirtriton(args, build_dir, source_dir=None, rocm_path=None):
    """Build and install rocMLIR::rockCompiler for the hip-ep link.

    source_dir and rocm_path normally come from hip-ep's CMake cache. The
    shared-LLVM flow runs before hip-ep is ever configured, so there is no
    cache yet and both are supplied by the caller.
    """
    if not have_tool("ninja"):
        raise BuildError(
            "--enable_rocmlirtriton requires Ninja for the isolated "
            "rocmlirTriton compiler build"
        )

    if source_dir is None:
        source_dir = Path(
            _cmake_cache_value(build_dir, "HIPDNN_ROCMLIRTRITON_SOURCE_DIR")
        )
    if rocm_path is None:
        rocm_path = Path(_cmake_cache_value(build_dir, "THEROCK_DIST"))
    if not (source_dir / "CMakeLists.txt").exists():
        raise BuildError(f"rocmlirTriton checkout is incomplete: {source_dir}")
    if not (rocm_path / "bin").exists():
        raise BuildError(f"TheRock SDK is incomplete: {rocm_path}")

    rock_root = Path(build_dir) / "_rocmlirtriton"
    rock_build = rock_root / "build"
    rock_install = rock_root / "install"
    build_env = _msvc_build_environment()
    compiler_dir = rocm_path / "lib" / "llvm" / "bin"
    if not compiler_dir.exists():
        compiler_dir = rocm_path / "bin"
    clang_cl = compiler_dir / "clang-cl.exe"
    lld_link = compiler_dir / "lld-link.exe"
    if IS_WINDOWS and not clang_cl.exists():
        # Some TheRock Windows bundles ship clang.exe without the clang-cl.exe
        # alias. Clang selects its CL-compatible driver mode from argv[0], so a
        # build-local hard link provides the expected spelling without
        # modifying the SDK.
        clang = compiler_dir / "clang.exe"
        if not clang.exists():
            raise BuildError(
                "rocmlirTriton requires clang.exe/clang-cl.exe from "
                f"TheRock; checked {compiler_dir}"
            )
        toolchain_dir = rock_root / "toolchain"
        toolchain_dir.mkdir(parents=True, exist_ok=True)
        clang_cl = toolchain_dir / "clang-cl.exe"
        if not clang_cl.exists():
            try:
                os.link(clang, clang_cl)
            except OSError:
                shutil.copy2(clang, clang_cl)
    if IS_WINDOWS and not lld_link.exists():
        raise BuildError(f"rocmlirTriton requires lld-link.exe; checked {compiler_dir}")

    step("Configure rocmlirTriton static rockCompiler")
    cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(rock_build),
        "-G",
        "Ninja",
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DCMAKE_INSTALL_PREFIX={rock_install}",
        f"-DROCM_PATH={rocm_path}",
        "-DBUILD_FAT_LIBROCKCOMPILER=ON",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DMLIR_INCLUDE_TESTS=OFF",
        "-DMLIR_ENABLE_ROCM_RUNNER=OFF",
        # Fat-package install still ships LLVMX86*.lib even when the ROCm
        # runner is off; rocmlirTriton otherwise builds AMDGPU only.
        "-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU",
        "-DROCMLIR_DRIVER_E2E_TEST_ENABLED=OFF",
        "-DROCK_E2E_TEST_ENABLED=OFF",
        "-DTRITON_BUILD_BINARY=OFF",
        # rocmlirTriton forces -Werror plus warnings that GCC reports
        # differently than the Clang upstream validates with. Demote them.
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="
        + (
            Path(__file__).resolve().parent / "cmake" / "rocmlir_relax_werror.cmake"
        ).as_posix(),
    ]
    if args.shared_llvm and not IS_WINDOWS:
        # On ELF, hip-ep's libraries are compiled with RTTI and derive from MLIR
        # bases, so their vtables reference typeinfo an -fno-rtti LLVM never
        # emits. MSVC does not have that coupling and the hip-ep libraries that
        # consume MLIR build with /GR-, so leave rocmlirTriton's own setting
        # alone there rather than force a full rebuild of the shared tree.
        cmd.append("-DLLVM_ENABLE_RTTI=ON")
    if IS_WINDOWS:
        cmd += [
            f"-DCMAKE_C_COMPILER={clang_cl}",
            f"-DCMAKE_CXX_COMPILER={clang_cl}",
            f"-DCMAKE_LINKER={lld_link}",
            "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
        ]
    if have_tool("sccache"):
        cmd += [
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ]
    run_subprocess(cmd, env=build_env)

    step("Build rocmlirTriton static rockCompiler")
    run_subprocess(
        [
            "cmake",
            "--build",
            str(rock_build),
            "--config",
            args.config,
            "--parallel",
            str(args.parallel),
            "--target",
            "librockCompiler",
        ],
        env=build_env,
    )

    if args.shared_llvm:
        # hip-ep consumes these out of the shared build tree and nothing in
        # librockCompiler's dependency graph reaches them: lldCOFF backs
        # DLLLinker.cpp's in-process COFF link (rocmlirTriton only ever links
        # ELF code objects), llvm-link merges lib/Runtime's per-module bitcode
        # into runtime.bc, and the tablegen/LIT utilities drive hip-ep's own
        # dialects and tests.
        extras = [
            "lldCOFF",
            "lldCommon",
            "llvm-link",
            "mlir-tblgen",
            "mlir-pdll",
            "FileCheck",
            "not",
            "count",
            "split-file",
            # MLIR/LLVM libraries hip-ep links that rockCompiler does not use:
            # the shape dialect and its lowerings (hipsr lowers shape to
            # extents), the bufferization pipelines, the pass-plugin and
            # debug/observer support hip-mlir-opt registers, and the ORC JIT
            # stack behind the in-process execution engine.
            "LLVMExecutionEngine",
            "LLVMJITLink",
            "LLVMOrcJIT",
            "LLVMOrcShared",
            "LLVMOrcTargetProcess",
            "LLVMRuntimeDyld",
            "MLIRBufferizationPipelines",
            "MLIRBufferizationToMemRef",
            "MLIRDebug",
            "MLIRIRDL",
            "MLIRObservers",
            "MLIROptLib",
            "MLIRPluginsLib",
            "MLIRRemarkStreamer",
            "MLIRShapeDialect",
            "MLIRShapeOpsTransforms",
            "MLIRShapeToStandard",
        ]
        known = _ninja_target_names(rock_build)
        if known:
            extras = [
                target
                for target in extras
                if known.intersection({target, f"{target}.lib", f"{target}.exe"})
            ]
        if extras:
            step("Build shared-LLVM tools and libraries for hip-ep")
            run_subprocess(
                [
                    "cmake",
                    "--build",
                    str(rock_build),
                    "--config",
                    args.config,
                    "--parallel",
                    str(args.parallel),
                    "--target",
                    *extras,
                ],
                env=build_env,
            )
    # Windows fat-package install copies a fixed LLVM/MLIR .lib list. The
    # rockCompiler INTERFACE target does not depend on every member (notably
    # LLVMX86*), so build any missing archives before cmake --install.
    missing_targets = _missing_rockcompiler_lib_targets(rock_build)
    known = _ninja_target_names(rock_build)
    if known:
        missing_targets = [t for t in missing_targets if t in known]
    if missing_targets:
        step("Build remaining rockCompiler package libraries")
        run_subprocess(
            [
                "cmake",
                "--build",
                str(rock_build),
                "--config",
                args.config,
                "--parallel",
                str(args.parallel),
                "--target",
                *missing_targets,
            ],
            env=build_env,
        )
    step("Install rocmlirTriton static package")
    run_subprocess(
        ["cmake", "--install", str(rock_build), "--config", args.config],
        env=build_env,
    )

    configs = list(rock_install.rglob("rocmlir-config.cmake"))
    if not configs:
        configs = list(rock_install.rglob("rocMLIR-config.cmake"))
    if not configs:
        configs = list(rock_install.rglob("rocMLIRConfig.cmake"))
    if not configs:
        raise BuildError(
            "rocmlirTriton installed without a rocMLIR package config under "
            f"{rock_install}"
        )
    return rock_install


def _llvm_built_from_source(build_dir):
    """True when cmake/deps.cmake built LLVM in-tree (embedded sub-build)."""
    cache = Path(build_dir) / "CMakeCache.txt"
    if not cache.exists():
        return False
    for line in cache.read_text(errors="ignore").splitlines():
        if line.startswith("HIPDNN_LLVM_EMBEDDED:") and line.rstrip().endswith("ON"):
            return True
    return False


def build_targets(args, build_dir):
    step("Build")
    run_subprocess(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            args.config,
            "--parallel",
            str(args.parallel),
        ]
    )
    # The embedded LLVM sub-build marks lld EXCLUDE_FROM_ALL, so the main build
    # never produces it. The runtime per-model link (`clang++ -fuse-ld=lld`)
    # needs ld.lld next to the in-tree clang++, so build it explicitly by name.
    # (We cannot pull lld into the CMake graph via add_dependencies -- that
    # corrupts LLVM's install(EXPORT); see cmake/deps.cmake.) clang itself is
    # already built as a dependency of lib/Runtime's bitcode step.
    if _llvm_built_from_source(build_dir) and not IS_WINDOWS:
        step("Build in-tree lld (runtime device-link toolchain)")
        run_subprocess(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                args.config,
                "--parallel",
                str(args.parallel),
                "--target",
                "lld",
            ]
        )
    step("Install")
    run_subprocess(["cmake", "--install", str(build_dir), "--config", args.config])


def run_tests(args, build_dir):
    """Run the GPU-free test suites (no device needed, so they run on the build
    machine in every CI job): the MLIR LIT pass-verification suite plus the
    compiler-plugin registrar and output-allocator ctest unit tests."""
    step("Test (check-hip-mlir-lit)")
    run_subprocess(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            args.config,
            "--target",
            "check-hip-mlir-lit",
        ]
    )

    # GPU-free ctest unit suites (built as part of the default target above).
    # GPU / hip-test e2e suites are not invoked here.
    step("Test (compiler/runtime GPU-free unit tests)")
    run_subprocess(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            args.config,
            "-R",
            "StaticPlugins|OutputAllocator",
            "--output-on-failure",
        ]
    )


# ---------------------------------------------------------------------------
# Arguments + main
# ---------------------------------------------------------------------------


def parse_arguments():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default="Release",
        choices=["Release", "RelWithDebInfo", "Debug"],
        help="build configuration (default: Release)",
    )
    p.add_argument("--build_dir", default=str(WORKSPACE / "build" / PROJECT_NAME))
    p.add_argument("--install_dir", default=str(WORKSPACE / "install"))
    p.add_argument(
        "--cmake_generator",
        default=None,
        help="cmake generator (default: 'Visual Studio 17 2022' on Windows, Ninja elsewhere)",
    )
    p.add_argument(
        "--cmake_extra_defines",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra -D<KEY=VALUE> forwarded to cmake configure (repeatable)",
    )
    p.add_argument(
        "--cmake_prefix_path",
        default="",
        help="extra prefixes (';'-separated) forwarded to CMAKE_PREFIX_PATH, "
        "e.g. CI-built/cached llvm-install;ort-install",
    )
    p.add_argument(
        "--therock_dist",
        default="",
        help="path to a TheRock ROCm SDK (else cmake/deps.cmake auto-downloads)",
    )
    p.add_argument(
        "--hip_arch",
        default="",
        help="GPU arch (e.g. gfx1151); auto-detected on Linux if unset",
    )
    p.add_argument(
        "--mock", action="store_true", help="mock runtime (no GPU/HIP/TheRock)"
    )
    p.add_argument(
        "--skip_submodule_sync",
        action="store_true",
        help="do not sync/update git submodules",
    )
    p.add_argument(
        "--clean", action="store_true", help="remove build/ and install/ then exit"
    )
    p.add_argument(
        "--skip_build", action="store_true", help="configure only; do not build/install"
    )
    p.add_argument(
        "--skip_wheel",
        action="store_true",
        help="do not build the Python wheel (built by default on real Windows builds)",
    )
    p.add_argument(
        "--skip_tests",
        action="store_true",
        help="do not run the LIT tests after install",
    )
    p.add_argument(
        "--enable_rocmlirtriton",
        action="store_true",
        help="automatically build static rocMLIR::rockCompiler and link it "
        "into hipgpu (large, long-running build)",
    )
    p.add_argument(
        "--shared_llvm",
        action="store_true",
        help="build the pinned LLVM once, inside rocmlirTriton, and have hip-ep "
        "consume it instead of compiling the same tree a second time "
        "(implies --enable_rocmlirtriton, requires --therock_dist)",
    )
    p.add_argument(
        "--rocmlirtriton_only",
        action="store_true",
        help="build and install rocmlirTriton (the shared LLVM and "
        "rocMLIR::rockCompiler) then exit without configuring hip-ep. CI uses "
        "this to populate the dependency cache in a job that has no ONNX "
        "Runtime prefix yet (implies --shared_llvm)",
    )
    p.add_argument("--allow_running_as_root", action="store_true")
    p.add_argument("-j", "--parallel", type=int, default=os.cpu_count() or 4)
    return p.parse_args()


def main():
    log.debug("argv: %s", " ".join(shlex.quote(a) for a in sys.argv[1:]))
    _check_python_version()
    args = parse_arguments()

    build_dir = Path(args.build_dir)
    install_dir = Path(args.install_dir)
    prefix_paths = [
        s for s in args.cmake_prefix_path.replace(os.pathsep, ";").split(";") if s
    ]

    if args.clean:
        for d in (build_dir, install_dir):
            if d.exists():
                log.info(f"removing {d}")
                shutil.rmtree(d)
        log.info("clean complete.")
        return

    if not IS_WINDOWS and os.geteuid() == 0 and not args.allow_running_as_root:
        raise BuildError(
            "Running as root is not allowed. Pass --allow_running_as_root to override."
        )

    log.info(f"REPO={REPO}")
    log.info(f"WORKSPACE={WORKSPACE}")
    log.info(f"platform={'windows' if IS_WINDOWS else 'linux'}  jobs={args.parallel}")

    args.cmake_generator = args.cmake_generator or default_generator()
    log.info(f"cmake generator: {args.cmake_generator}")

    if not args.skip_submodule_sync:
        update_submodules()
    check_toolchain(args.cmake_generator)

    mock = args.mock
    hip_arch = args.hip_arch.strip()
    if not mock and not hip_arch:
        hip_arch = detect_hip_arch() or ""
        if hip_arch:
            log.info(f"auto-detected HIP_ARCHITECTURES={hip_arch}")
        else:
            log.info("no AMD GPU detected and no --hip_arch; falling back to --mock")
            mock = True

    if args.rocmlirtriton_only:
        args.shared_llvm = True

    if args.shared_llvm:
        args.enable_rocmlirtriton = True
        if not args.therock_dist:
            raise BuildError(
                "--shared_llvm requires --therock_dist: rocmlirTriton is built "
                "before hip-ep is configured, so TheRock cannot be read from "
                "hip-ep's CMake cache yet"
            )

    if args.enable_rocmlirtriton and mock:
        raise BuildError(
            "--enable_rocmlirtriton requires a real build and TheRock SDK; "
            "it cannot be combined with --mock"
        )

    args.rocmlirtriton_ready = False
    args.rocmlirtriton_source_dir = None
    args.shared_llvm_dirs = {}

    if args.shared_llvm:
        # One LLVM for both projects: rocmlirTriton builds the pinned tree, then
        # hip-ep consumes that build instead of compiling its own copy. That
        # inverts the default order -- rocmlirTriton has to be cloned and built
        # before hip-ep is configured, so hip-ep's first configure can already
        # resolve find_package(MLIR) against it.
        args.rocmlirtriton_source_dir = fetch_rocmlirtriton(build_dir)
        rock_install = build_rocmlirtriton(
            args,
            build_dir,
            source_dir=args.rocmlirtriton_source_dir,
            rocm_path=Path(args.therock_dist),
        )
        prefix_paths.append(str(rock_install))
        args.shared_llvm_dirs = shared_llvm_package_dirs(
            Path(build_dir) / "_rocmlirtriton" / "build"
        )
        log.info(f"sharing rocmlirTriton's LLVM: {args.shared_llvm_dirs['LLVM_DIR']}")
        args.rocmlirtriton_ready = True
        if args.rocmlirtriton_only:
            step("DONE")
            log.info(f"rocmlirTriton install: {rock_install}")
            for name, path in args.shared_llvm_dirs.items():
                log.info(f"{name}={path}")
            return
        generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock)
    else:
        generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock)
        if args.enable_rocmlirtriton:
            rock_install = build_rocmlirtriton(args, build_dir)
            prefix_paths.append(str(rock_install))
            args.rocmlirtriton_ready = True
            generate_build_tree(args, build_dir, prefix_paths, hip_arch, mock)
    if args.skip_build:
        log.info("configure done; skipping build (--skip_build).")
        return
    build_targets(args, build_dir)

    if not args.skip_tests:
        run_tests(args, build_dir)

    step("DONE")
    log.info(f"install tree: {install_dir}")


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        log.error(str(exc))
        sys.exit(1)
