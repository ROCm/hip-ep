#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Compiler: orchestrates hip-compiler.exe to compile MLIR text to GPU DLLs.

Handles MSVC environment setup, TheRock SDK paths, and subprocess management.
"""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Path Discovery
# ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent


def _find_hip_compiler() -> Path:
    """Find hip-compiler.exe in the build directory."""
    candidates = [
        _PROJECT_ROOT.parent / "build" / "onnx-hipdnn-ep" / "bin" / "hip-compiler.exe",
        _PROJECT_ROOT / "build" / "bin" / "hip-compiler.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Check PATH
    from shutil import which

    found = which("hip-compiler.exe") or which("hip-compiler")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "hip-compiler.exe not found. Build the project first.\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


def _find_therock() -> Path:
    """Find TheRock ROCm SDK."""
    env = os.environ.get("THEROCK_DIST", "")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        _PROJECT_ROOT.parent / "therock",
        Path("C:/Users")
        / os.environ.get("USERNAME", "user")
        / "Documents/code/therock",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "TheRock SDK not found. Set THEROCK_DIST environment variable."
    )


def _find_vcvars() -> Path:
    """Find Visual Studio vcvarsall.bat."""
    candidates = [
        Path(
            r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
        ),
        Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        ),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Visual Studio vcvarsall.bat not found.")


# ──────────────────────────────────────────────────────────────────────
# Compiler
# ──────────────────────────────────────────────────────────────────────


class Compiler:
    """Compiles MLIR text to GPU DLL via hip-compiler.exe."""

    def __init__(
        self,
        hip_compiler: Optional[Path] = None,
        therock: Optional[Path] = None,
        vcvars: Optional[Path] = None,
    ):
        self.hip_compiler = hip_compiler or _find_hip_compiler()
        self.therock = therock or _find_therock()
        self.vcvars = vcvars or _find_vcvars()

    def compile(
        self,
        mlir_text: str,
        output_dir: Optional[Path] = None,
        timeout: int = 120,
    ) -> Path:
        """Compile MLIR text to a GPU DLL.

        Args:
            mlir_text: Torch dialect MLIR text
            output_dir: Directory for output files (default: temp dir)
            timeout: Compilation timeout in seconds

        Returns:
            Path to the compiled DLL

        Raises:
            CompilationError: If hip-compiler fails
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="hip_compile_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        mlir_path = output_dir / "model.mlir"
        dll_path = output_dir / "model.dll"

        # Write MLIR
        mlir_path.write_text(mlir_text, encoding="utf-8")

        # Create compilation script
        cmd_path = output_dir / "_compile.cmd"
        cmd_path.write_text(
            "@echo off\n"
            f'call "{self.vcvars}" x64 >nul 2>&1\n'
            f"set THEROCK_DIST={self.therock}\n"
            f"set PATH={self.therock}\\bin;%PATH%\n"
            f'"{self.hip_compiler}" "{mlir_path}" -o "{dll_path}"\n',
            encoding="utf-8",
        )

        # Run
        t0 = time.perf_counter()
        result = subprocess.run(
            ["cmd", "/c", str(cmd_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(output_dir),
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0 or not dll_path.exists():
            error_msg = result.stderr[-500:] if result.stderr else "unknown error"
            raise CompilationError(f"hip-compiler failed ({elapsed:.1f}s): {error_msg}")

        log.info(
            f"Compiled MLIR to DLL in {elapsed:.1f}s "
            f"({dll_path.stat().st_size / 1024:.0f} KB)"
        )
        return dll_path


class CompilationError(Exception):
    """Raised when hip-compiler fails."""

    pass
