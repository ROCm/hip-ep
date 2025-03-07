#
# Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
import importlib.abc
import importlib.machinery
import sys


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in sys.builtin_module_names:
            return importlib.machinery.ModuleSpec(
                fullname,
                importlib.machinery.BuiltinImporter,
            )


sys.meta_path.append(Finder())

import os
import types
import zipfile
import sys
import io


class embed_zipimporter(object):
    def __init__(self, zip_file):
        self._zipfile = zip_file
        self._paths = [x.filename for x in self._zipfile.filelist]

    def _mod_to_paths(self, fullname):
        py_filename = fullname.replace(".", "/") + ".py"
        py_package = fullname.replace(".", "/") + "/__init__.py"

        if py_filename in self._paths:
            return py_filename
        elif py_package in self._paths:
            return py_package
        else:
            return None

    def find_module(self, fullname, path):
        if self._mod_to_paths(fullname) is not None:
            return self
        return None

    def load_module(self, fullname):
        filename = self._mod_to_paths(fullname)
        if not filename in self._paths:
            raise ImportError(fullname)
        new_module = types.ModuleType(fullname)
        new_module.__name__ = fullname

        new_module.__file__ = filename
        new_module.__loader__ = self
        if filename.endswith("__init__.py"):
            new_module.__path__ = []
            new_module.__package__ = fullname
        else:
            new_module.__package__ = fullname.rpartition(".")[0]
        sys.modules[fullname] = new_module

        exec(self._zipfile.open(filename, "r").read(), new_module.__dict__)
        return new_module


module_zip = zipfile.ZipFile(io.BytesIO(vaip_lib), "r")
sys.meta_path.append(embed_zipimporter(module_zip))
