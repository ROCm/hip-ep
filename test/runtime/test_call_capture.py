#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""CPU checks for native diagnostic JSON transport; no GPU is executed."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class CallCaptureTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.executable = Path(cls.temporary.name) / "call-capture-test"
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        command = compiler + [
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(cls.root / "test/runtime/test_call_capture.cpp"),
            str(cls.root / "lib/Runtime/chrome_trace.cpp"),
            "-o",
            str(cls.executable),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    def run_fixture(self, mode=None):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "trace.json"
            command = [str(self.executable), str(base)]
            if mode:
                command.append(mode)
            subprocess.run(command, check=True, capture_output=True, text=True)
            traces = list(base.parent.glob("trace.*.json"))
            self.assertEqual(len(traces), 1)
            return json.loads(traces[0].read_text(encoding="utf-8"))

    def test_descriptors_deduplicate_without_losing_call_identity(self):
        value = self.run_fixture()
        self.assertEqual(value["schema"], "hip-ep-call-trace-1")
        self.assertTrue(value["example_only"])
        for field in ("cases", "graphs", "windows"):
            self.assertEqual(len(value[field]), 1)
        self.assertEqual(value["capture_identity_conflicts"], [])
        gpu = [event for event in value["traceEvents"] if event.get("cat") == "gpu"]
        self.assertEqual(
            [event["args"]["call_id"] for event in gpu[:2]],
            ["synthetic-call-1", "synthetic-call-2"],
        )
        self.assertEqual(gpu[0]["args"]["case_id"], gpu[1]["args"]["case_id"])

    def test_escaping_optional_slots_and_actual_types(self):
        case = self.run_fixture()["cases"][0]
        self.assertEqual(case["inputs"][0]["role"], 'input\n"quoted"')
        self.assertEqual(case["inputs"][0]["dtype"], "float16")
        self.assertEqual(case["inputs"][0]["shape"], [1, 8])
        self.assertEqual(case["inputs"][1], {"slot": 1, "present": False})
        self.assertEqual(case["parameters"]["scalar"], 0.125)
        self.assertEqual(case["parameters"]["length"], 8)

    def test_missing_and_nonfinite_values_do_not_become_finite_operands(self):
        parameters = self.run_fixture()["cases"][0]["parameters"]
        self.assertIsNone(parameters["missing"])
        self.assertIsNone(parameters["invalid"])

    def test_original_gpu_and_cpu_timing_fields_are_preserved(self):
        events = self.run_fixture()["traceEvents"]
        gpu = [event for event in events if event.get("cat") == "gpu"]
        cpu = [event for event in events if event.get("cat") == "cpu"]
        self.assertEqual([event["ts"] for event in gpu], [101, 102, 103])
        self.assertEqual([event["dur"] for event in gpu], [0.125, 0.250, 0.375])
        self.assertEqual([event["ts"] for event in cpu], [100, 101, 102])
        self.assertEqual([event["dur"] for event in cpu], [1, 1, 1])

    def test_unannotated_legacy_record_remains_visible(self):
        gpu = [
            event
            for event in self.run_fixture()["traceEvents"]
            if event.get("cat") == "gpu"
        ]
        self.assertEqual(gpu[2]["name"], "legacy")
        self.assertNotIn("record_kind", gpu[2]["args"])
        self.assertNotIn("case_id", gpu[2]["args"])

    def test_conflicting_case_identity_is_reported_without_changing_execution(self):
        value = self.run_fixture("conflict")
        self.assertIn("synthetic-case", value["capture_identity_conflicts"])
        gpu = [event for event in value["traceEvents"] if event.get("cat") == "gpu"]
        self.assertEqual(len(gpu), 4)
        self.assertEqual(value["cases"][0]["op_type"], "SyntheticUnknownOp")

    def test_mixed_example_provenance_is_reported(self):
        value = self.run_fixture("mixed")
        self.assertTrue(value["example_only"])
        self.assertIn(
            "example_provenance_conflict", value["capture_identity_conflicts"]
        )

    def test_reloaded_capture_survives_reuse_of_the_same_address(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "trace.json"
            subprocess.run(
                [str(self.executable), str(base), "reuse-address"], check=True
            )
            traces = list(base.parent.glob("trace.*.json"))
            self.assertEqual(len(traces), 2)
            documents = [json.loads(path.read_text()) for path in traces]
            self.assertEqual(
                {value["session_capture"]["session_id"] for value in documents},
                {"synthetic-session-0", "synthetic-session-1"},
            )
            for value in documents:
                gpu = [
                    event for event in value["traceEvents"] if event.get("cat") == "gpu"
                ]
                self.assertEqual(len(gpu), 1)
                self.assertEqual(gpu[0]["name"], value["session_capture"]["session_id"])


if __name__ == "__main__":
    unittest.main()
