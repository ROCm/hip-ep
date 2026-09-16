#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sysconfig

from setuptools import setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools without the vendored command
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": bdist_wheel})
