#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Self-contained tests for the ROCm 7.2.1 output validator.

Assertions are adapted from AMD rocprofiler-sdk's MIT-licensed rocprofv3
validation tests.

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rocprofv3_schema import (
    AGENT_INFO_HEADER,
    HIP_API_HEADER,
    KERNEL_HEADER,
    MEMORY_ALLOCATION_HEADER,
    MEMORY_COPY_HEADER,
    OutputPaths,
    ValidationError,
    validate,
)


TEST_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = TEST_DIR / "fixtures" / "rocprofv3_7_2_1"


class Rocprofv3SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = OutputPaths.from_directory(FIXTURE_DIR, "golden")

    def copy_fixtures(self, destination: Path) -> OutputPaths:
        copied = destination / "outputs"
        shutil.copytree(FIXTURE_DIR, copied)
        return OutputPaths.from_directory(copied, "golden")

    def test_official_serializer_headers_are_frozen(self) -> None:
        self.assertEqual(len(AGENT_INFO_HEADER), 53)
        self.assertEqual(len(HIP_API_HEADER), 7)
        self.assertEqual(len(KERNEL_HEADER), 22)
        self.assertEqual(len(MEMORY_COPY_HEADER), 8)
        self.assertEqual(len(MEMORY_ALLOCATION_HEADER), 8)

    def test_synthetic_golden_outputs_validate(self) -> None:
        validate(self.paths)

    def test_empty_domain_may_omit_its_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            data = json.loads(paths.json.read_text(encoding="utf-8"))
            data["rocprofiler-sdk-tool"][0]["buffer_records"]["memory_copy"] = []
            paths.json.write_text(json.dumps(data), encoding="utf-8")
            paths.memory_copy.unlink()
            validate(paths)

    def test_cli_accepts_directory_prefix_and_fixture_modes(self) -> None:
        script = TEST_DIR / "rocprofv3_schema.py"
        commands = (
            [sys.executable, str(script), "--fixtures"],
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(FIXTURE_DIR),
                "--prefix",
                "golden",
            ],
            [
                sys.executable,
                str(script),
                "--agent-info",
                str(self.paths.agent_info),
                "--hip-api",
                str(self.paths.hip_api),
                "--kernel",
                str(self.paths.kernel_dispatch),
                "--memory-copy",
                str(self.paths.memory_copy),
                "--memory-allocation",
                str(self.paths.memory_allocation),
                "--json",
                str(self.paths.json),
            ],
        )
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    command, check=False, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("PASS", result.stdout)

    def test_rejects_unquoted_csv_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            text = paths.hip_api.read_text(encoding="utf-8")
            paths.hip_api.write_text(
                text.replace('"HIP_RUNTIME_API_EXT"', "HIP_RUNTIME_API_EXT", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "quoting"):
                validate(paths)

    def test_rejects_non_integer_nanosecond_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            data = json.loads(paths.json.read_text(encoding="utf-8"))
            process = data["rocprofiler-sdk-tool"][0]
            process["buffer_records"]["kernel_dispatch"][0]["start_timestamp"] = 1.5
            paths.json.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "unsigned JSON integer"):
                validate(paths)

    def test_rejects_invalid_rocm_enum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            data = json.loads(paths.json.read_text(encoding="utf-8"))
            process = data["rocprofiler-sdk-tool"][0]
            process["buffer_records"]["memory_copy"][0]["operation"] = 5
            paths.json.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "memory-copy enum"):
                validate(paths)

    def test_rejects_broken_string_table_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            data = json.loads(paths.json.read_text(encoding="utf-8"))
            process = data["rocprofiler-sdk-tool"][0]
            process["strings"]["buffer_records"][10]["operations"][2] = (
                "WRONG_DIRECTION"
            )
            paths.json.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "incorrect memory-copy enum"):
                validate(paths)

    def test_rejects_missing_kernel_symbol_serializer_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            data = json.loads(paths.json.read_text(encoding="utf-8"))
            process = data["rocprofiler-sdk-tool"][0]
            del process["kernel_symbols"][1]["accum_vgpr_count"]
            paths.json.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "field mismatch"):
                validate(paths)

    def test_rejects_csv_json_count_or_correlation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            lines = paths.memory_copy.read_text(encoding="utf-8").splitlines()
            paths.memory_copy.write_text(
                "\n".join([lines[0], lines[1].replace(",3,", ",99,")]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "count mismatch"):
                validate(paths)

        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_fixtures(Path(temp))
            text = paths.kernel_dispatch.read_text(encoding="utf-8")
            paths.kernel_dispatch.write_text(
                text.replace(",2,1000000400,", ",99,1000000400,"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValidationError, "correlation/timestamp"):
                validate(paths)


if __name__ == "__main__":
    unittest.main()
