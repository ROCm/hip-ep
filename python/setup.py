#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Setuptools shim.

Metadata + packaging live in the generated pyproject.toml (the native artifacts
ship as package-data of the ``onnxruntime.capi`` namespace package). This file
only forces a platform-specific, Python-ABI-agnostic wheel tag (py3-none-<plat>),
since the wheel ships prebuilt native libraries, not a Python extension.
"""

import sysconfig

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools without the vendored command
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    # Keep the package content at the wheel root (Root-Is-Purelib: true) so it
    # ships as onnxruntime/capi/... (like ONNX Runtime) rather than relocated
    # under .data/purelib/. The bundled DLLs don't link libpython, so the tag is
    # py3-none-<platform>: ABI-agnostic but platform-specific. We compute the
    # platform tag ourselves because a "pure" wheel would otherwise report
    # plat == "any".
    def get_tag(self):
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": bdist_wheel})
