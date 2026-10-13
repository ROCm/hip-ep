#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Parity tests for the independent Python and C++ post-processing paths."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON_CLI = ROOT / "python" / "ktrace_cli.py"
NATIVE = Path(
    os.environ.get("KTRACE_POSTPROCESS", ROOT / "bin" / "ktrace_postprocess.exe")
)


def _write_spool(directory: Path, dynamic_lds: int = 0) -> None:
    base = directory / ".probe_4242_ops.csv"
    base.write_text(
        "Id,Op,Kind,Kernel_Name,Bytes,Device,Queue,Corr,Begin_us,End_us,Duration_us\n"
        "1,dispatch,KernelExecution,_Z16synthetic_kernelPfi,0,0,7,2,1000000.400,1000000.900,0.500\n"
        "2,copy,CopyHostToDevice,,4096,0,7,3,1000001.000,1000001.200,0.200\n",
        encoding="utf-8",
    )
    Path(str(base) + ".api.csv").write_text(
        "Corr,Api,Api_Id,Thread,Enter_us,Exit_us,Host_us,Args\n"
        "1,hipMalloc,140,4242,1000000.000,1000000.100,0.100,ptr=0x12345000 size=4096\n"
        "2,hipLaunchKernel,171,4242,1000000.200,1000000.280,0.080,"
        f"grid=16x1x1 block=64x1x1 shmem={dynamic_lds} stream=0x10\n"
        "3,hipMemcpyAsync,215,4242,1000000.950,1000000.980,0.030,"
        "size=4096 kind=1 stream=0x10 dst=0x12345000 src=0x1000\n"
        "4,hipFree,69,4242,1000001.300,1000001.320,0.020,ptr=0x12345000\n",
        encoding="utf-8",
    )
    Path(str(base) + ".kernels.csv").write_text(
        "Kernel_Name,Func_Addr,VGPR,Scratch_Bytes,Static_LDS,Max_Threads,"
        "Binary_Version,Occ_Block_Size,Occ_Dyn_LDS,Theoretical_Blocks_Per_CU,"
        "Theoretical_Occupancy_Pct\n"
        "_Z16synthetic_kernelPfi,0x2000,24,32,256,1024,1,64,0,8,50.0\n",
        encoding="utf-8",
    )
    Path(str(base) + ".vram.csv").write_text(
        "Time_us,Free_Bytes,Total_Bytes,Used_Bytes\n"
        "1000000.000,3221225472,4294967296,1073741824\n"
        "1000001.500,3221221376,4294967296,1073745920\n",
        encoding="utf-8",
    )
    Path(str(base) + ".launch.csv").write_text(
        "Process_Id,Resume_Timestamp_ns\n4242,999900000\n",
        encoding="utf-8",
    )
    Path(str(base) + ".summary.txt").write_text(
        "pid              : 4242\n\n"
        "device           : Synthetic GPU\n"
        "  CUs            : 10\n"
        "  warp size      : 32\n"
        "  threads/CU     : 2048\n"
        "  regs/block     : 65536\n"
        "  LDS/CU         : 65536\n"
        "  clock / mem    : 2400000 kHz / 1000000 kHz, bus 256 bit\n",
        encoding="utf-8",
    )


def _add_roctx_spool(directory: Path) -> None:
    base = directory / ".probe_4242_ops.csv"
    Path(str(base) + ".roctx.csv").write_text(
        "Op,Op_Id,Process,Thread,Range_Id,Timestamp_us,Message\n"
        'MARK,0,4242,101,0,1000000.0500,"startup"\n'
        'RANGE_PUSH,1,4242,101,0,1000000.1000,"outer"\n'
        'RANGE_PUSH,1,4242,101,0,1000000.2000,"inner"\n'
        'RANGE_POP,2,4242,101,0,1000000.7000,""\n'
        'RANGE_POP,2,4242,101,0,1000000.9000,""\n'
        'RANGE_START,3,4242,101,7,1000001.0000,"cross"\n'
        'RANGE_STOP,4,4242,202,7,1000001.4000,""\n',
        encoding="utf-8",
    )


def _add_cpu_sidecars(directory: Path) -> None:
    (directory / "probe_cpu_sched.csv").write_text(
        "Timestamp_ns,Cpu,Old_Thread,New_Thread,Old_State,Wait_Reason\n"
        "999950000,0,0,4242,0,0\n"
        "1000050000,0,4242,0,2,5\n",
        encoding="utf-8",
    )
    (directory / "probe_cpu_samples.csv").write_text(
        "Timestamp_ns,Cpu,Process_Id,Thread_Id,Instruction_Pointer,Stack\n"
        '1000000000,0,4242,4242,0x1234,"0x1234;0x5678"\n',
        encoding="utf-8",
    )


def _add_runtime_sidecar(directory: Path) -> None:
    (directory / "probe_hip_runtime_info.csv").write_text(
        '"Process_Id","Hip_Runtime_Path","File_Version"\n'
        '4242,"C:\\Windows\\System32\\amdhip64_7.dll","10.0.1.2"\n',
        encoding="utf-8",
    )


def _write_phase_capture(directory: Path) -> Path:
    path = directory / "phase_kernel_trace.csv"
    header = (
        '"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id",'
        '"Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp",'
        '"LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count",'
        '"Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",'
        '"Grid_Size_X","Grid_Size_Y","Grid_Size_Z"\n'
    )
    records = [
        ("prefill_anchor_a", 1_000_000, 1_001_000),
        ("prefill_anchor_b", 1_100_000, 1_102_000),
        ("generic_prefill_work", 1_125_000, 1_145_000),
        ("generic_decode_work", 1_165_000, 1_183_000),
        ("decode_anchor_a", 1_200_000, 1_203_000),
        ("decode_anchor_b", 1_300_000, 1_304_000),
    ]
    lines = [header]
    for dispatch, (name, begin_us, end_us) in enumerate(records, 1):
        lines.append(
            f'"KERNEL_DISPATCH","Agent 1",1,1,1,{dispatch},{dispatch},'
            f'"{name}",{dispatch},{begin_us * 1000},{end_us * 1000},'
            "0,0,8,0,0,64,1,1,64,1,1\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_out_of_order_capture(directory: Path) -> Path:
    path = directory / "unordered_kernel_trace.csv"
    header = (
        '"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id",'
        '"Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp",'
        '"LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count",'
        '"Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",'
        '"Grid_Size_X","Grid_Size_Y","Grid_Size_Z"\n'
    )
    records = [
        ("kernel_a", 1_000_000, 1_010_000),
        ("kernel_c", 1_200_000, 1_210_000),
        ("kernel_b", 1_100_000, 1_110_000),
    ]
    lines = [header]
    for dispatch, (name, begin_us, end_us) in enumerate(records, 1):
        lines.append(
            f'"KERNEL_DISPATCH","Agent 1",4,0,1,{dispatch},{dispatch},'
            f'"{name}",{dispatch},{begin_us * 1000},{end_us * 1000},'
            "0,0,8,0,0,64,1,1,64,1,1\n"
        )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    )


def _scrub(value):
    dynamic = {
        "output_path",
        "tmp_directory",
        "raw_output_path",
        "raw_tmp_directory",
        "source_spool",
        "hostname",
        "release",
    }
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items() if key not in dynamic}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


@unittest.skipUnless(NATIVE.is_file(), "native post-processor has not been built")
class NativeParityTests(unittest.TestCase):
    def test_perfetto_lanes_handle_out_of_order_rocprof_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = _write_out_of_order_capture(root)
            py_trace = root / "python_trace.json"
            cpp_trace = root / "cpp_trace.json"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "chrome",
                str(capture),
                str(py_trace),
                "--all",
            )
            _run(
                str(NATIVE),
                "chrome",
                str(capture),
                str(cpp_trace),
                "--all",
            )
            py_doc = json.loads(py_trace.read_text(encoding="utf-8"))
            cpp_doc = json.loads(cpp_trace.read_text(encoding="utf-8"))
            self.assertEqual(py_doc["traceEvents"], cpp_doc["traceEvents"])
            queue_labels = [
                event["args"]["name"]
                for event in cpp_doc["traceEvents"]
                if event.get("name") == "thread_name" and event.get("pid") == 0
            ]
            self.assertEqual(queue_labels, ["queue Agent 1/4"])

    def test_phase_assignment_includes_untagged_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = _write_phase_capture(root)
            py_report = root / "python_report.txt"
            cpp_report = root / "cpp_report.txt"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "analyze",
                str(capture),
                "-o",
                str(py_report),
            )
            _run(
                str(NATIVE),
                "analyze",
                str(capture),
                "-o",
                str(cpp_report),
            )
            python_lines = py_report.read_text(encoding="utf-8").splitlines()
            cpp_lines = cpp_report.read_text(encoding="utf-8").splitlines()
            self.assertEqual(python_lines, cpp_lines)
            report = "\n".join(cpp_lines)
            self.assertIn("prefill is mostly: generic_prefill_work", report)
            self.assertIn("decode is mostly: generic_decode_work", report)

    def test_native_all_runs_complete_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "all"
            output.mkdir()
            _write_spool(output)
            result = _run(str(NATIVE), "all", str(output))
            self.assertIn("[all] complete: 1 process capture(s)", result.stdout)
            for name in (
                "probe_results.json",
                "probe_kernel_trace.csv",
                "probe_kernel_trace_readable.csv",
                "probe_ktrace_extensions.json",
                "probe_ktrace_launch_info.csv",
                "ktrace_report.txt",
                "ktrace_perfetto.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            _run(
                sys.executable,
                str(ROOT / "tests" / "rocprofv3_schema.py"),
                "--output-dir",
                str(output),
                "--prefix",
                "probe",
            )

    def test_native_all_output_format_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pftrace_only = Path(temp) / "pftrace"
            pftrace_only.mkdir()
            _write_spool(pftrace_only)
            _run(
                str(NATIVE),
                "all",
                str(pftrace_only),
                "--chrome-pftrace",
                "--flows",
            )
            self.assertTrue((pftrace_only / "ktrace_perfetto.pftrace").is_file())
            self.assertFalse((pftrace_only / "ktrace_perfetto.json").exists())

            both = Path(temp) / "both"
            both.mkdir()
            _write_spool(both)
            _run(
                str(NATIVE),
                "all",
                str(both),
                "--chrome-json",
                "--chrome-pftrace",
                "--flows",
            )
            self.assertTrue((both / "ktrace_perfetto.json").is_file())
            self.assertTrue((both / "ktrace_perfetto.pftrace").is_file())

    def test_native_finalize_dynamic_lds_passes_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "native"
            output.mkdir()
            _write_spool(output, dynamic_lds=33)
            _run(str(NATIVE), "finalize", str(output))
            _run(
                sys.executable,
                str(ROOT / "tests" / "rocprofv3_schema.py"),
                "--output-dir",
                str(output),
                "--prefix",
                "probe",
            )

    def test_finalize_analyze_and_chrome_match_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            python_out = root / "python"
            native_out = root / "native"
            source.mkdir()
            _write_spool(source)
            shutil.copytree(source, python_out)
            shutil.copytree(source, native_out)

            _run(sys.executable, str(PYTHON_CLI), "finalize", str(python_out))
            _run(str(NATIVE), "finalize", str(native_out))
            _add_cpu_sidecars(python_out)
            _add_cpu_sidecars(native_out)
            _add_runtime_sidecar(python_out)
            _add_runtime_sidecar(native_out)

            csv_names = (
                "probe_agent_info.csv",
                "probe_hip_api_trace.csv",
                "probe_kernel_trace.csv",
                "probe_kernel_trace_readable.csv",
                "probe_memory_copy_trace.csv",
                "probe_memory_allocation_trace.csv",
                "probe_ktrace_launch_info.csv",
            )
            for name in csv_names:
                left = python_out / name
                right = native_out / name
                self.assertEqual(left.exists(), right.exists(), name)
                if left.exists():
                    self.assertEqual(left.read_bytes(), right.read_bytes(), name)
            readable = (native_out / "probe_kernel_trace_readable.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("synthetic_kernel", readable)

            for name in ("probe_results.json", "probe_ktrace_extensions.json"):
                left = _scrub(
                    json.loads((python_out / name).read_text(encoding="utf-8"))
                )
                right = _scrub(
                    json.loads((native_out / name).read_text(encoding="utf-8"))
                )
                self.assertEqual(left, right, name)

            py_report = python_out / "report.txt"
            cpp_report = native_out / "report.txt"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "analyze",
                str(python_out / "probe_kernel_trace.csv"),
                "-o",
                str(py_report),
            )
            _run(
                str(NATIVE),
                "analyze",
                str(native_out / "probe_kernel_trace.csv"),
                "-o",
                str(cpp_report),
            )
            self.assertEqual(
                py_report.read_text(encoding="utf-8").splitlines(),
                cpp_report.read_text(encoding="utf-8").splitlines(),
            )
            self.assertIn(
                "  amdhip64 path : C:\\Windows\\System32\\amdhip64_7.dll",
                cpp_report.read_text(encoding="utf-8"),
            )

            py_trace = python_out / "trace.json"
            cpp_trace = native_out / "trace.json"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "chrome",
                str(python_out / "probe_kernel_trace.csv"),
                str(py_trace),
                "--all",
            )
            _run(
                str(NATIVE),
                "chrome",
                str(native_out / "probe_kernel_trace.csv"),
                str(cpp_trace),
                "--all",
            )
            py_doc = json.loads(py_trace.read_text(encoding="utf-8"))
            cpp_doc = json.loads(cpp_trace.read_text(encoding="utf-8"))
            self.assertEqual(py_doc["traceEvents"], cpp_doc["traceEvents"])
            self.assertEqual(py_doc["otherData"]["time_origin"], "ResumeThread QPC")
            self.assertEqual(cpp_doc["otherData"]["time_origin"], "ResumeThread QPC")
            self.assertTrue(
                any(event.get("cat") == "cpu_sched" for event in cpp_doc["traceEvents"])
            )
            self.assertTrue(
                any(
                    event.get("cat") == "cpu_sample" for event in cpp_doc["traceEvents"]
                )
            )
            hip_malloc = next(
                event
                for event in cpp_doc["traceEvents"]
                if event.get("ph") == "X" and event.get("name") == "hipMalloc"
            )
            self.assertEqual(hip_malloc["ts"], 100.0)

            py_flow_trace = python_out / "flow.json"
            cpp_flow_trace = native_out / "flow.json"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "chrome",
                str(python_out / "probe_kernel_trace.csv"),
                str(py_flow_trace),
                "--all",
                "--flows",
            )
            _run(
                str(NATIVE),
                "chrome",
                str(native_out / "probe_kernel_trace.csv"),
                str(cpp_flow_trace),
                "--all",
                "--flows",
            )
            py_flow_events = json.loads(py_flow_trace.read_text(encoding="utf-8"))[
                "traceEvents"
            ]
            cpp_flow_events = json.loads(cpp_flow_trace.read_text(encoding="utf-8"))[
                "traceEvents"
            ]
            self.assertEqual(py_flow_events, cpp_flow_events)
            self.assertTrue(
                any(event.get("name") == "hipLaunchKernel" for event in cpp_flow_events)
            )
            flow_starts = {
                event["id"] for event in cpp_flow_events if event.get("ph") == "s"
            }
            flow_finishes = {
                event["id"] for event in cpp_flow_events if event.get("ph") == "f"
            }
            self.assertTrue(flow_starts & flow_finishes)
            perfetto = python_out / "trace.pftrace"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "perfetto",
                str(python_out / "probe_kernel_trace.csv"),
                str(perfetto),
                "--all",
                "--flows",
            )
            payload = perfetto.read_bytes()
            self.assertGreater(len(payload), 100)
            self.assertEqual(payload[0], 0x0A)

            cpp_export_json = native_out / "export.json"
            cpp_export_pftrace = native_out / "export.pftrace"
            _run(
                str(NATIVE),
                "export",
                str(native_out / "probe_kernel_trace.csv"),
                "--chrome-json",
                str(cpp_export_json),
                "--chrome-pftrace",
                str(cpp_export_pftrace),
                "--all",
                "--flows",
            )
            self.assertEqual(
                json.loads(cpp_export_json.read_text(encoding="utf-8"))["traceEvents"],
                cpp_flow_events,
            )
            self.assertGreater(cpp_export_pftrace.stat().st_size, 100)
            self.assertEqual(cpp_export_pftrace.read_bytes()[0], 0x0A)

            try:
                from perfetto.trace_processor import TraceProcessor
            except ImportError:
                TraceProcessor = None
            if TraceProcessor is not None:

                def counts(path: Path) -> tuple[int, int, int, int]:
                    processor = TraceProcessor(file_path=str(path))
                    try:
                        queries = (
                            "select count(*) n from slice",
                            "select count(*) n from counter",
                            "select count(*) n from flow",
                            "select count(*) n from stack_profile_callsite",
                        )
                        return tuple(
                            list(processor.query(query))[0].n for query in queries
                        )
                    finally:
                        processor.close()

                self.assertEqual(
                    counts(cpp_export_pftrace),
                    counts(perfetto),
                )

    def test_roctx_finalize_and_chrome_match_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            python_out = root / "python"
            native_out = root / "native"
            source.mkdir()
            _write_spool(source)
            _add_roctx_spool(source)
            shutil.copytree(source, python_out)
            shutil.copytree(source, native_out)

            _run(sys.executable, str(PYTHON_CLI), "finalize", str(python_out))
            _run(str(NATIVE), "finalize", str(native_out))
            self.assertEqual(
                (python_out / "probe_roctx_trace.csv").read_bytes(),
                (native_out / "probe_roctx_trace.csv").read_bytes(),
            )

            py_trace = python_out / "roctx.json"
            cpp_trace = native_out / "roctx.json"
            _run(
                sys.executable,
                str(PYTHON_CLI),
                "chrome",
                str(python_out / "probe_kernel_trace.csv"),
                str(py_trace),
                "--all",
            )
            _run(
                str(NATIVE),
                "chrome",
                str(native_out / "probe_kernel_trace.csv"),
                str(cpp_trace),
                "--all",
            )
            py_doc = json.loads(py_trace.read_text(encoding="utf-8"))
            cpp_doc = json.loads(cpp_trace.read_text(encoding="utf-8"))
            py_events = [
                event for event in py_doc["traceEvents"] if event.get("cat") == "roctx"
            ]
            cpp_events = [
                event for event in cpp_doc["traceEvents"] if event.get("cat") == "roctx"
            ]
            self.assertEqual(py_events, cpp_events)
            self.assertTrue(cpp_events)
            self.assertTrue(all(event["pid"] == 2 for event in cpp_events))
            thread_names = {
                event["tid"]: event["args"]["name"]
                for event in cpp_doc["traceEvents"]
                if event.get("name") == "thread_name"
            }
            self.assertEqual(thread_names[101], "ROCTx thread 101")
            process_names = {
                event["pid"]: event["args"]["name"]
                for event in cpp_doc["traceEvents"]
                if event.get("name") == "process_name"
            }
            self.assertEqual(
                [process_names[pid] for pid in (0, 1, 2)],
                ["GPU (hardware queues)", "Host (HIP API)", "ROCTx annotations"],
            )


if __name__ == "__main__":
    unittest.main()
