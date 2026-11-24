#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""GenAI end-to-end accuracy gate for the Windows GPU workflow.

Runs the hipdnn-accuracy-test framework (run_accuracy.py) on a GenAI model
through the AMDGPU (HIP) EP from a gpu-test-package, and compares the result
with the latest main run. See test/genai_accuracy/README.md.

Subcommands:
  prepare  Clone the pinned framework, build or reuse the Python venv, install
           the package's ORT/OGA wheels, and download the pinned model.
  run      Mirror the model, generate its AMDGPU genai_config, run the test and
           write <results-dir>/result.json. Exits 1 when the test fails.
  report   Render a Markdown summary of result.json against a main baseline
           and list perplexity regressions beyond the tolerance.

Only the standard library is used here; the framework runs in the venv.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ACCURACY_REPO_URL = "https://github.com/AMDmoore/hipdnn-accuracy-test.git"
# Pinned so a perplexity change is attributable to the package under test, not
# to a framework update. Bump together with the main baseline.
ACCURACY_REPO_COMMIT = "6274c2b6b131f8d9cca9a443a24f7a6439e680b5"
VLM_TEMPLATE_CONFIG = "test_config_vlm.json"

# Mirrors one entry of the GenAI modelzoo used by the HIP EP GenAI CI.
MODELS = {
    "Qwen3.8-Flash-Next": {
        "hf_repo": "amd/Qwen3.8-Flash-Next_gpuep_1.0_gpu_128",
        "revision": "52f7f5d1c5184c47146cdb2cc92814c8e46c2ebe",
        "test": "PPL_VLM",
        "variant": "allgpu",
        # The modelzoo entry's ci_options. HIP_CHUNK_SIZE sets search.chunk_size
        # in the generated AMDGPU config; LLM_TOKENS_MITRIX only drives the
        # performance cases and is logged for parity with the Jenkins flow.
        "ci_options": {
            "LLM_TOKENS_MITRIX": "128|128,2048|128,16384|128",
            "HIP_CHUNK_SIZE": 512,
        },
        "hip_provider_options": {},
    },
}
DEFAULT_MODEL = "Qwen3.8-Flash-Next"

AMDGPU_GENAI_CONFIG = "genai_config_AMDGPU.json"
CPU_FALLBACK_KEY = "session.disable_cpu_ep_fallback"
AMDGPU_PROVIDER = {"AMDGPU": {"profile": "hip"}}

PINNED_PACKAGES = (
    "transformers==5.6.2",
    "tokenizers==0.22.2",
    "huggingface_hub==1.11.0",
    "typer==0.24.0",
    "click==8.2.1",
    "safetensors==0.7.0",
    "regex==2026.4.4",
    "numpy==2.4.4",
    "pandas==3.0.2",
    "psutil",
    "colorama==0.4.6",
    "tqdm==4.67.3",
    "thefuzz==0.22.1",
    "onnx==1.21.0",
    "torch==2.11.0+cpu",
    "datasets",
    "Pillow",
)
GENAI_WHEEL_PATTERNS = (
    "onnxruntime_directml-*.whl",
    "onnxruntime_genai_directml-*.whl",
)


def _log(message: str) -> None:
    print(message, flush=True)


def _run(command: list, **kwargs) -> subprocess.CompletedProcess:
    _log("[CMD] " + subprocess.list2cmdline([str(c) for c in command]))
    return subprocess.run([str(c) for c in command], check=True, **kwargs)


def _model_spec(name: str) -> dict:
    if name not in MODELS:
        raise SystemExit(f"Unknown model {name!r}; known: {', '.join(MODELS)}")
    return MODELS[name]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _paths(args) -> dict:
    cache_root = Path(args.cache_root).resolve()
    spec = _model_spec(args.model)
    return {
        "framework": cache_root / "hipdnn-accuracy-test",
        "venv": cache_root / "genai-accuracy-venv",
        "hf_home": cache_root / "hf-home",
        "model": cache_root / "genai-models" / spec["hf_repo"].replace("/", "--"),
    }


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _sync_framework(framework_dir: Path) -> None:
    if not (framework_dir / ".git").is_dir():
        if framework_dir.exists():
            shutil.rmtree(framework_dir)
        framework_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", framework_dir])
        _run(["git", "-C", framework_dir, "remote", "add", "origin", ACCURACY_REPO_URL])
    _run(
        [
            "git",
            "-C",
            framework_dir,
            "fetch",
            "-q",
            "--depth",
            "1",
            "origin",
            ACCURACY_REPO_COMMIT,
        ]
    )
    _run(["git", "-C", framework_dir, "checkout", "-q", "--force", "FETCH_HEAD"])
    # Drops EP DLLs and caches a previous run left in the checkout.
    _run(["git", "-C", framework_dir, "clean", "-q", "-fdx"])


def _venv_manifest() -> dict:
    return {
        "python": ".".join(map(str, sys.version_info[:2])),
        "base_executable": str(Path(sys.executable).resolve()),
        "packages": list(PINNED_PACKAGES),
    }


def _ensure_venv(venv_dir: Path) -> Path:
    manifest_path = venv_dir / "genai-accuracy-manifest.json"
    venv_python = _venv_python(venv_dir)
    desired = _venv_manifest()
    if manifest_path.is_file() and venv_python.is_file():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
        if current == desired:
            _log(f"Reusing GenAI accuracy venv: {venv_dir}")
            return venv_python

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    _run([sys.executable, "-m", "venv", venv_dir])
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            "pip",
        ]
    )
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            *PINNED_PACKAGES,
        ]
    )
    manifest_path.write_text(json.dumps(desired, indent=2) + "\n", encoding="utf-8")
    return venv_python


def _package_wheels(package_dir: Path) -> list:
    wheels = []
    for pattern in GENAI_WHEEL_PATTERNS:
        matches = sorted((package_dir / "wheels").glob(pattern))
        if not matches:
            raise SystemExit(f"Missing {pattern} under {package_dir / 'wheels'}")
        wheels.append(matches[-1])
    return wheels


def _install_package_wheels(venv_python: Path, package_dir: Path) -> None:
    wheels = _package_wheels(package_dir)
    # The wheel version does not change between builds, so force the reinstall
    # to pick up this package. Dependencies are reinstalled too, as the Jenkins
    # flow does for the dynamic modelzoo; that resolves numpy to the latest
    # release the ORT wheel accepts, overriding the pin above.
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            *wheels,
        ]
    )


def _download_model(
    venv_python: Path, spec: dict, model_dir: Path, hf_home: Path
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    # snapshot_download into local_dir is incremental: complete files are
    # verified from local metadata and skipped.
    script = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])\n"
    )
    env = dict(os.environ, HF_HOME=str(hf_home), HF_HUB_DISABLE_TELEMETRY="1")
    _run(
        [venv_python, "-c", script, spec["hf_repo"], spec["revision"], model_dir],
        env=env,
    )


def cmd_prepare(args) -> int:
    paths = _paths(args)
    spec = _model_spec(args.model)
    package_dir = Path(args.package_dir).resolve()
    _sync_framework(paths["framework"])
    venv_python = _ensure_venv(paths["venv"])
    _install_package_wheels(venv_python, package_dir)
    _download_model(venv_python, spec, paths["model"], paths["hf_home"])
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _mirror_model(source: Path, target: Path) -> None:
    """Hard-link the model into the work dir.

    run_accuracy.py switches configs by overwriting genai_config.json in the
    model dir, so it must not run on the shared download. Hard links keep the
    weights out of a second copy; metadata files are copied.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name.startswith(".") or not entry.is_file():
            continue
        destination = target / entry.name
        if entry.suffix == ".json":
            shutil.copy2(entry, destination)
            continue
        try:
            os.link(entry, destination)
        except OSError as exc:
            raise SystemExit(
                f"Cannot hard-link {entry} into {target} ({exc}); keep the work "
                "dir on the same volume as the model cache"
            ) from None


def _rewrite_sessions(node, provider_options: list) -> None:
    if isinstance(node, dict):
        session_options = node.get("session_options")
        if isinstance(session_options, dict):
            original = session_options.get("provider_options")
            keep_original = isinstance(original, list) and any(
                isinstance(option, dict) and "AMDGPU" in option for option in original
            )
            options = copy.deepcopy(original if keep_original else provider_options)
            preserved = {
                key: value
                for key, value in session_options.items()
                if key not in (CPU_FALLBACK_KEY, "provider_options")
            }
            session_options.clear()
            session_options.update(preserved)
            # Without this, a node the EP rejects silently runs on the CPU EP
            # and the score no longer reflects the GPU path.
            session_options[CPU_FALLBACK_KEY] = "1"
            session_options["provider_options"] = options
        for value in node.values():
            _rewrite_sessions(value, provider_options)
    elif isinstance(node, list):
        for value in node:
            _rewrite_sessions(value, provider_options)


def _write_amdgpu_genai_config(model_dir: Path, spec: dict) -> None:
    config = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
    provider = copy.deepcopy(AMDGPU_PROVIDER)
    provider["AMDGPU"].update(spec["hip_provider_options"])
    _rewrite_sessions(config.get("model", {}), [provider])
    chunk_size = spec["ci_options"].get("HIP_CHUNK_SIZE")
    if str(chunk_size).strip().lower() == "null":
        # Matches the Jenkins flow: "null" drops the template's chunk_size.
        config.get("search", {}).pop("chunk_size", None)
    elif chunk_size is not None:
        config.setdefault("search", {})["chunk_size"] = int(chunk_size)
    (model_dir / AMDGPU_GENAI_CONFIG).write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )


def _venv_site_packages(venv_python: Path) -> Path:
    result = _run(
        [
            venv_python,
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _install_dll_search_hook(venv_python: Path, package_bin: Path) -> None:
    """Put the package's onnxruntime.dll and HIP runtime first on the DLL path.

    The ORT wheel ships its own onnxruntime.dll; the AMDGPU EP must load next
    to the package build instead.
    """
    if os.name != "nt":
        return
    site_dir = _venv_site_packages(venv_python)
    dll_dirs = [str(site_dir / "onnxruntime_genai"), str(package_bin)]
    (site_dir / "_genai_dll_init.py").write_text(
        "import os\n"
        f"_DLL_DIRS = {json.dumps(dll_dirs)}\n"
        "_dirs = [d for d in _DLL_DIRS if os.path.isdir(d)]\n"
        "_norm = lambda p: os.path.normcase(os.path.normpath(p))\n"
        "_keep = [p for p in os.environ.get('PATH', '').split(os.pathsep)\n"
        "         if p and _norm(p) not in {_norm(d) for d in _dirs}]\n"
        "os.environ['PATH'] = os.pathsep.join(_dirs + _keep)\n"
        "for _d in _dirs:\n"
        "    os.add_dll_directory(_d)\n",
        encoding="utf-8",
    )
    (site_dir / "_genai_dll_init.pth").write_text(
        "import _genai_dll_init\n", encoding="utf-8"
    )


def _write_compat_sitecustomize(target_dir: Path) -> None:
    # perplexity_vlm.py has a literal '%' in an argparse help string, which
    # Python 3.14 rejects when the parser is built.
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "sitecustomize.py").write_text(
        "import argparse\n"
        "_check_help = getattr(argparse.ArgumentParser, '_check_help', None)\n"
        "def _lenient_check_help(self, action):\n"
        "    try:\n"
        "        return _check_help(self, action)\n"
        "    except ValueError as exc:\n"
        "        if str(exc) != 'badly formed help string':\n"
        "            raise\n"
        "if _check_help is not None:\n"
        "    argparse.ArgumentParser._check_help = _lenient_check_help\n",
        encoding="utf-8",
    )


def _test_params(framework_dir: Path, test: str) -> dict:
    template = json.loads(
        (framework_dir / VLM_TEMPLATE_CONFIG).read_text(encoding="utf-8")
    )
    return copy.deepcopy(template["tests"][test].get("params", {}))


def cmd_run(args) -> int:
    paths = _paths(args)
    spec = _model_spec(args.model)
    package_dir = Path(args.package_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    venv_python = _venv_python(paths["venv"])
    if (
        not venv_python.is_file()
        or not (paths["model"] / "genai_config.json").is_file()
    ):
        raise SystemExit("Run the prepare subcommand first")

    for directory in (work_dir, results_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    for key, value in spec["ci_options"].items():
        _log(f"[INFO] GenAI model {args.model}: applied ci option {key}={value}")
    _log(f"=== GenAI accuracy test: {args.model} ===")

    model_dir = work_dir / "model" / args.model
    _mirror_model(paths["model"], model_dir)
    _write_amdgpu_genai_config(model_dir, spec)
    _log(
        f"[INFO] generated {AMDGPU_GENAI_CONFIG} for GenAI model {args.model}: "
        f"{model_dir / AMDGPU_GENAI_CONFIG}"
    )

    test = spec["test"]
    variant = spec["variant"]
    params = _test_params(paths["framework"], test)
    case_dir = work_dir / "output"
    run_config = {
        "model_dir": str(model_dir),
        "output_dir": str(case_dir),
        "package_dir": str(package_dir),
        # gpu-test-package bundles the HIP runtime in bin/.
        "therock_dist": str(package_dir),
        "genai_configs": {variant: AMDGPU_GENAI_CONFIG},
        "tests": {test: {"seq_lengths": [variant], "params": params}},
    }
    run_config_path = work_dir / "run_accuracy_config.json"
    run_config_path.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    _install_dll_search_hook(venv_python, package_dir / "bin")
    ep_dll = package_dir / "bin" / "amdgpu-ep.dll"
    if ep_dll.is_file():
        shutil.copy2(ep_dll, paths["framework"] / ep_dll.name)
    compat_dir = work_dir / "python-compat"
    _write_compat_sitecustomize(compat_dir)

    env = dict(os.environ)
    env["HF_HOME"] = str(paths["hf_home"])
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(compat_dir), env.get("PYTHONPATH")])
    )

    command = [
        venv_python,
        paths["framework"] / "run_accuracy.py",
        "--config",
        run_config_path,
        "--tests",
        test,
        "--output-dir",
        case_dir,
        "--package-dir",
        package_dir,
    ]
    start = time.monotonic()
    return_code = subprocess.run(
        [str(c) for c in command], cwd=paths["framework"], env=env
    ).returncode
    elapsed = round(time.monotonic() - start, 1)

    records = []
    detail_path = case_dir / "results_detail.json"
    if detail_path.is_file():
        records = json.loads(detail_path.read_text(encoding="utf-8"))
    record = records[0] if records else {}
    success = return_code == 0 and bool(record.get("success"))
    error = record.get("error") or ""
    if return_code != 0 and not error:
        error = f"run_accuracy.py exited with code {return_code}"
    if not records and not error:
        error = "run_accuracy.py produced no result records"

    result = {
        "model": args.model,
        "hf_repo": spec["hf_repo"],
        "revision": spec["revision"],
        "framework_commit": ACCURACY_REPO_COMMIT,
        "test": test,
        "ep": "hip",
        "variant": variant,
        "params": params,
        "success": success,
        "error": error,
        "metrics": record.get("metrics", {}),
        "elapsed_seconds": elapsed,
    }
    (results_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for name in ("logs", "results_summary.csv", "results_detail.json"):
        source = case_dir / name
        if source.is_dir():
            shutil.copytree(source, results_dir / name)
        elif source.is_file():
            shutil.copy2(source, results_dir / name)
    shutil.copy2(model_dir / AMDGPU_GENAI_CONFIG, results_dir / AMDGPU_GENAI_CONFIG)

    _log(json.dumps(result, indent=2))
    return 0 if success else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

COMPARABLE_KEYS = ("model", "revision", "framework_commit", "test", "ep", "params")


def _load_result(path: str | None) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _perplexity(result: dict | None) -> float | None:
    if not result or not result.get("success"):
        return None
    value = result.get("metrics", {}).get("perplexity")
    return float(value) if value is not None else None


def cmd_report(args) -> int:
    current = _load_result(args.current)
    baseline = _load_result(args.baseline)
    tolerance = args.tolerance
    lines = ["## GenAI Accuracy Results (HIP EP)", ""]
    regressions = []

    if current is None:
        lines.append("No result: the accuracy run did not produce result.json.")
    else:
        cur = _perplexity(current)
        base = _perplexity(baseline)
        comparable = baseline is not None and all(
            current.get(key) == baseline.get(key) for key in COMPARABLE_KEYS
        )
        metrics = current.get("metrics", {})
        samples = (
            f"{metrics['samples_evaluated']}/{metrics['samples_total']}"
            if "samples_evaluated" in metrics and "samples_total" in metrics
            else "-"
        )
        if cur is None:
            status = f"FAIL: {current.get('error') or 'no perplexity'}"
            main_cell, delta_cell = (f"{base:.4f}" if base is not None else "-"), "-"
        elif base is None or not comparable:
            status = "PASS (no comparable main baseline)"
            main_cell = delta_cell = "-"
        else:
            delta = (cur - base) / base
            main_cell, delta_cell = f"{base:.4f}", f"{delta * 100:+.2f}%"
            if delta > tolerance:
                status = "REGRESSION"
                regressions.append(
                    f"{current['model']} {current['test']}: perplexity {cur:.4f} vs "
                    f"main {base:.4f} ({delta * 100:+.2f}% > {tolerance * 100:.1f}%)"
                )
            else:
                status = "PASS"
        lines += [
            "| Model | Test | EP | Perplexity | main | Delta | Samples | Time (s) | Status |",
            "|-------|------|----|-----------|------|---|---------|----------|--------|",
            f"| {current['model']} | {current['test']} | {current['ep']} | "
            f"{f'{cur:.4f}' if cur is not None else '-'} | {main_cell} | {delta_cell} | "
            f"{samples} | {current.get('elapsed_seconds', '-')} | {status} |",
            "",
            f"Model `{current['hf_repo']}@{current['revision'][:8]}` - framework "
            f"`{current['framework_commit'][:8]}` - lower perplexity is better; "
            f"PRs fail above +{tolerance * 100:.1f}% vs main.",
        ]
        if baseline is not None and not comparable:
            lines.append(
                "Main baseline used a different model revision, framework commit or "
                "test parameters, so it was not compared."
            )
    if args.baseline_note:
        lines.append(f"Main baseline: {args.baseline_note}")

    summary = "\n".join(lines) + "\n"
    Path(args.summary_out).write_text(summary, encoding="utf-8")
    if regressions:
        Path(args.regressions_out).write_text(
            "\n".join(regressions) + "\n", encoding="utf-8"
        )
    _log(summary)
    return 0


def main() -> int:
    # Windows runners default stdout to the ANSI code page (cp1252); never let a
    # non-encodable character in model output or the summary abort the run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
        p.add_argument("--package-dir", required=True, help="gpu-test-package root")
        p.add_argument(
            "--cache-root",
            required=True,
            help="Persistent dir for the framework, venv, HF cache and models",
        )

    p_prepare = sub.add_parser("prepare", help="Sync framework, venv and model")
    add_common(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run", help="Run the accuracy test")
    add_common(p_run)
    p_run.add_argument("--work-dir", required=True, help="Per-run scratch dir")
    p_run.add_argument("--results-dir", required=True, help="Per-run results dir")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="Compare against a main baseline")
    p_report.add_argument("--current", required=True, help="result.json of this run")
    p_report.add_argument("--baseline", help="result.json of the main baseline")
    p_report.add_argument(
        "--baseline-note", default="", help="Markdown baseline source"
    )
    p_report.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Allowed relative perplexity increase (default 0.02)",
    )
    p_report.add_argument("--summary-out", required=True)
    p_report.add_argument("--regressions-out", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
