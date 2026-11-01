#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Explicit, serial full-model comparison; never collected by pytest/CTest.

Runs no-Ring, Ring and TurboQuant from the SAME installed build. Each worker
has a fresh process/cache and retains one Generator across all prompt appends.
Diagnostic timings include logits readback and tracing; they are not benchmarks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from fixture import make_fixture, score_answer
from metrics import compare_logits, summarize

MODES = ("no-ring", "ring", "turboquant")


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def last_logits(gen):
    x = np.asarray(gen.get_logits())
    x = x.reshape(-1, x.shape[-1])[-1].astype(np.float32).copy()
    if not np.isfinite(x).all():
        raise AssertionError("Non-finite next-token logits")
    return x


def cache_hashes(gen, length, capacity, width):
    result = {}
    # Qwen3.8-27B has full-attention cache at layers 3,7,...,63.
    for layer in range(3, 64, 4):
        for kind in ("key", "value"):
            name = f"present.{layer}.{kind}"
            x = np.asarray(gen.get_output(name))
            if x.shape != (1, 4, capacity, width):
                raise AssertionError((name, x.shape, capacity, width))
            result[name] = hashlib.sha256(
                np.ascontiguousarray(x[:, :, :length]).view(np.uint8)
            ).hexdigest()
    return result


def prepare_model(args, directory, mode):
    directory.mkdir()
    # Keep large ONNX/weight/tokenizer data external. Hard links are local only;
    # require the output on the same volume instead of silently copying weights.
    for source in args.model.iterdir():
        if (
            source.is_file()
            and source.name != "genai_config.json"
            and source.suffix != ".part"
        ):
            os.link(source, directory / source.name)
    cfg = json.loads((args.model / "genai_config.json").read_text(encoding="utf-8"))
    capacity = args.turns * (args.prompt_tokens + args.max_new_tokens) + 256
    cfg["model"]["context_length"] = capacity
    cfg["search"].update(
        max_length=capacity, min_length=0, past_present_share_buffer=True
    )
    cfg["search"].pop("chunk_size", None)
    for name in ("decoder", "embedding"):
        session = cfg["model"][name]["session_options"]
        session["provider_options"] = [
            {
                "AMDGPU": {
                    "profile": "hip",
                    "config_file": str(args.install / "bin/morphizen_config.json"),
                }
            }
        ]
        session["session.disable_cpu_ep_fallback"] = "1"
    if mode == "turboquant":
        cfg["model"]["decoder"]["session_options"]["provider_options"][0][
            "AMDGPU"
        ].update(
            kvCacheQuantizationBits="4",
            kv_cache_codec="tq4_mse_qr_v1",
            kv_cache_codec_file=str(args.codec),
        )
    save(directory / "genai_config.json", cfg)
    return capacity


def worker(args):
    mode = "no-ring" if args.worker == "no-ring-repeat" else args.worker
    out = args.output / args.track / args.worker
    out.mkdir(parents=True, exist_ok=False)
    # Runtime flag is cached on first use. Each worker starts with its own env.
    assert os.environ["HIPDNN_EP_USE_RING_ATTENTION"] == (
        "0" if mode == "no-ring" else "1"
    )
    dirs = [args.install / "bin"] + args.dll_dir
    handles = [os.add_dll_directory(str(p)) for p in dirs] if os.name == "nt" else []
    import onnxruntime_genai as og

    capacity = prepare_model(args, out / "model", mode)
    report = {
        "status": "running",
        "mode": mode,
        "track": args.track,
        "capacity": capacity,
        "genai_version": og.__version__,
        "genai_module": og.__file__,
        "turns": [],
        "timing_scope": "diagnostic including tracing/readback",
    }
    save(out / "receipt.json", report)
    try:
        model = og.Model(str(out / "model"))
        tokenizer = og.Tokenizer(model)
        end_tokens = list(tokenizer.encode("<|im_end|>"))
        assert len(end_tokens) == 1, "Expected Qwen's single chat end token"
        end_token = int(end_tokens[0])
        eos = json.loads((args.model / "genai_config.json").read_text())["model"][
            "eos_token_id"
        ]
        eos = set(eos if isinstance(eos, list) else [eos])
        fixture_path = args.output / "fixture.json"
        if not fixture_path.exists():
            save(fixture_path, make_fixture(tokenizer, args.turns, args.prompt_tokens))
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert (
            fixture["prompt_tokens"] == args.prompt_tokens
            and len(fixture["turns"]) == args.turns
        )
        report["fixture_sha256"] = digest(fixture_path)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=capacity, min_length=0, do_sample=False)
        gen = og.Generator(model, params)
        width = 66 if mode == "turboquant" else 256
        canonical = args.output / "matched-history/no-ring"
        control = None
        if args.track == "matched-history" and args.worker != "no-ring":
            control = json.loads(
                (canonical / "receipt.json").read_text(encoding="utf-8")
            )
            assert control["status"] == "passed"
        for turn in fixture["turns"]:
            index = turn["turn"]
            past = len(gen.get_sequence(0)) if index > 1 else 0
            old = cache_hashes(gen, past, capacity, width) if past else {}
            tokens = np.array(turn["tokens"], np.int32)
            start = time.perf_counter()
            gen.append_tokens(tokens)
            logits = last_logits(gen)
            prefill_seconds = time.perf_counter() - start
            sequence = np.array(gen.get_sequence(0), copy=True)
            assert len(sequence) == past + len(tokens)
            np.testing.assert_array_equal(sequence[past:], tokens)
            if past:
                assert cache_hashes(gen, past, capacity, width) == old, (
                    "Old cache rows changed during append"
                )
            prefix_hash = hashlib.sha256(sequence.tobytes()).hexdigest()
            if control:
                assert prefix_hash == control["turns"][index - 1]["prefix_sha256"], (
                    "Histories differ"
                )
            generated, vectors = [], []
            continuation = (
                control["turns"][index - 1]["generated_tokens"] if control else None
            )
            steps = (
                len(continuation) if continuation is not None else args.max_new_tokens
            )
            start = time.perf_counter()
            for step in range(steps):
                vectors.append(logits)
                if continuation is not None:
                    token = int(continuation[step])
                    gen.append_tokens(np.array([token], np.int32))
                else:
                    # Use the same unprocessed greedy rule in both frameworks.
                    # Explicit tokens keep every committed position auditable
                    # and separate model numerics from search-state behavior.
                    token = int(np.argmax(logits))
                    gen.append_tokens(np.array([token], np.int32))
                generated.append(token)
                # Materialize the pending selected token before snapshotting KV.
                # This also makes matched/free step counts directly inspectable.
                logits = last_logits(gen)
                if continuation is None and token in eos:
                    break
            forced_end = False
            if continuation is None and generated[-1] != end_token:
                # A truncated answer still needs a chat boundary before the next
                # user turn. Record this forced token separately from sampling.
                vectors.append(logits)
                generated.append(end_token)
                gen.append_tokens(np.array([end_token], np.int32))
                logits = last_logits(gen)
                forced_end = True
            decode_seconds = time.perf_counter() - start
            assert len(gen.get_sequence(0)) == past + len(tokens) + len(generated), (
                len(gen.get_sequence(0)),
                past,
                len(tokens),
                len(generated),
            )
            answer = tokenizer.decode(generated)
            row = {
                "turn": index,
                "past": past,
                "prompt_tokens": len(tokens),
                "context_after": len(gen.get_sequence(0)),
                "prefix_sha256": prefix_hash,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "generated_tokens": generated,
                "answer": answer,
                "forced_chat_end": forced_end,
                "content_score": score_answer(answer, turn),
                "old_cache_rows_unchanged": True,
                "finite_logits": True,
                "kv_capacity_bytes": 16 * 2 * 4 * capacity * width * 2,
            }
            if args.track == "matched-history":
                np.save(out / f"turn-{index}-logits.npy", np.stack(vectors))
            report["turns"].append(row)
            save(out / "receipt.json", report)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("generated_tokens", "answer")
                    }
                ),
                flush=True,
            )
        report["status"] = "passed"
        save(out / "receipt.json", report)
        del gen, model
    except Exception as error:
        report.update(status="failed", error=str(error))
        save(out / "receipt.json", report)
        raise
    finally:
        for handle in handles:
            handle.close()


def verify_dispatch(path, mode, prompt_tokens, turns):
    text = path.read_text(encoding="utf-8", errors="replace")
    ring = re.findall(r"\[REAL\] Ring GQA q=(\d+) past=(\d+)", text)
    tq = re.findall(r"\[REAL\] TurboQuant KV q=(\d+) past=(\d+)", text)
    if mode == "no-ring":
        assert not ring and not tq, "Unexpected experimental dispatch"
        assert "[REAL] GQA fused prefill" in text and "[REAL] flash GQA decode" in text
        return {"dense_prefill_decode": True, "ring": 0, "turboquant": 0}
    rows = tq if mode == "turboquant" else ring
    assert rows and any(int(q) == 1 for q, _ in rows), "Missing decode dispatch"
    assert any(int(q) == prompt_tokens and int(p) == 0 for q, p in rows), (
        "Missing initial prefill"
    )
    if turns > 1:
        assert any(int(q) == prompt_tokens and int(p) > 0 for q, p in rows), (
            "Missing append prefill"
        )
    if mode == "ring":
        assert not tq
    return {"ring": len(ring), "turboquant": len(tq), "prefill_decode_append": True}


def analyze(args):
    root = args.output / "matched-history"
    pairs = [("pytorch-bf16", m) for m in MODES] + [
        ("no-ring", "ring"),
        ("no-ring", "turboquant"),
        ("ring", "turboquant"),
    ]
    if not args.skip_repeat:
        pairs.append(("no-ring", "no-ring-repeat"))
    results = []
    for ref, actual in pairs:
        if (
            not (root / ref / "receipt.json").exists()
            or not (root / actual / "receipt.json").exists()
        ):
            continue
        a = json.loads((root / ref / "receipt.json").read_text())
        b = json.loads((root / actual / "receipt.json").read_text())
        assert a["status"] == b["status"] == "passed"
        for turn in range(1, args.turns + 1):
            ar, br = a["turns"][turn - 1], b["turns"][turn - 1]
            assert ar["prefix_sha256"] == br["prefix_sha256"]
            assert ar["generated_tokens"] == br["generated_tokens"]
            av, bv = (
                np.load(root / name / f"turn-{turn}-logits.npy")
                for name in (ref, actual)
            )
            assert av.shape == bv.shape and len(av) == len(ar["generated_tokens"])
            rows = [
                compare_logits(x, y, t)
                for x, y, t in zip(av, bv, ar["generated_tokens"], strict=True)
            ]
            results.append(
                {
                    "reference": ref,
                    "actual": actual,
                    "turn": turn,
                    "context_after": ar["context_after"],
                    "prefill": rows[0],
                    "continuation": summarize(rows[1:]) if len(rows) > 1 else None,
                    "all_positions": summarize(rows),
                    "positions": rows,
                }
            )
    save(args.output / "comparisons.json", results)
    lines = [
        "# Multi-turn numerical characterization",
        "",
        "Identical canonical history; quality characterization, not a pass threshold.",
        "",
        "| Reference â†’ path | Turn | Context | Min cosine | Max rel L2 | Mean JS | Top-1 agreement | Mean NLL Î” |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        m = row["all_positions"]
        lines.append(
            f"| {row['reference']} â†’ {row['actual']} | {row['turn']} | {row['context_after']} | "
            f"{m['min_cosine']:.6f} | {m['max_relative_l2']:.6f} | {m['mean_js']:.6g} | "
            f"{m['top1_agreement']:.3f} | {m['mean_nll_delta']:.6g} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--tracks", default="matched-history,free-running")
    parser.add_argument(
        "--torch-model", type=Path, help="Optional matching BF16 Hugging Face model"
    )
    parser.add_argument(
        "--torch-python",
        type=Path,
        help="Python with PyTorch and state-aware Transformers",
    )
    parser.add_argument("--skip-repeat", action="store_true")
    parser.add_argument(
        "--worker", choices=(*MODES, "no-ring-repeat"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--track", choices=("matched-history", "free-running"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    for key in ("model", "install", "codec", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.dll_dir = [p.resolve() for p in args.dll_dir]
    if not 1 <= args.turns <= 8 or args.max_new_tokens < 2:
        parser.error("Use 1..8 turns and at least two continuation tokens")
    if bool(args.torch_model) != bool(args.torch_python):
        parser.error("Specify both --torch-model and --torch-python")
    if args.torch_model:
        args.torch_model, args.torch_python = (
            args.torch_model.resolve(),
            args.torch_python.resolve(),
        )
    if args.worker:
        return worker(args)
    modes, tracks = args.modes.split(","), args.tracks.split(",")
    if not set(modes) <= set(MODES) or not set(tracks) <= {
        "matched-history",
        "free-running",
    }:
        parser.error("Unknown mode/track")
    if "matched-history" in tracks and modes[0] != "no-ring":
        parser.error("Matched history must generate the no-ring canonical run first")
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "args": {
            k: str(v)
            if isinstance(v, Path)
            else [str(p) for p in v]
            if isinstance(v, list)
            else v
            for k, v in vars(args).items()
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
        ).hexdigest(),
        "harness_sha256": digest(Path(__file__)),
        "codec_sha256": digest(args.codec),
        "model_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in args.model.iterdir()
            if p.is_file() and p.suffix != ".part"
        },
        "install_libraries": {
            p.name: digest(p) for p in (args.install / "bin").glob("*.dll")
        },
        "status": "running",
        "workers": [],
    }
    save(args.output / "manifest.json", manifest)
    try:
        for track in tracks:
            runs = modes + (
                ["no-ring-repeat"]
                if track == "matched-history" and not args.skip_repeat
                else []
            )
            for mode in runs:
                effective = "no-ring" if mode == "no-ring-repeat" else mode
                env = dict(
                    os.environ,
                    HIPDNN_EP_STRICT="1",
                    HIPDNN_EP_DEBUG="1",
                    HIPDNN_EP_PERF="0",
                    HIPDNN_EP_USE_RING_ATTENTION="0" if effective == "no-ring" else "1",
                    AMDGPU_EP_PATH=str(args.install / "bin/hipgpu.dll"),
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                )
                env["PATH"] = (
                    os.pathsep.join(map(str, [args.install / "bin", *args.dll_dir]))
                    + os.pathsep
                    + env["PATH"]
                )
                work = args.output / f"work-{track}-{mode}"
                work.mkdir()
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model",
                    str(args.model),
                    "--install",
                    str(args.install),
                    "--codec",
                    str(args.codec),
                    "--output",
                    str(args.output),
                    "--turns",
                    str(args.turns),
                    "--prompt-tokens",
                    str(args.prompt_tokens),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--worker",
                    mode,
                    "--track",
                    track,
                ]
                for directory in args.dll_dir:
                    cmd += ["--dll-dir", str(directory)]
                log = args.output / f"{track}-{mode}.log"
                with log.open("w", encoding="utf-8") as f:
                    subprocess.run(
                        cmd,
                        env=env,
                        cwd=work,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                evidence = verify_dispatch(
                    log, effective, args.prompt_tokens, args.turns
                )
                manifest["workers"].append(
                    {"track": track, "mode": mode, "dispatch": evidence}
                )
                save(args.output / "manifest.json", manifest)
        if args.torch_model:
            for track in tracks:
                cmd = [
                    str(args.torch_python),
                    str(Path(__file__).with_name("pytorch_reference.py")),
                    "--model",
                    str(args.torch_model),
                    "--oga-model",
                    str(args.model),
                    "--results",
                    str(args.output),
                    "--track",
                    track,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ]
                # Keep the reference isolated from OGA's patched package/DLL path.
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                work = args.output / f"work-{track}-pytorch-bf16"
                work.mkdir()
                with (args.output / f"{track}-pytorch-bf16.log").open(
                    "w", encoding="utf-8"
                ) as f:
                    subprocess.run(
                        cmd,
                        cwd=work,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                manifest["workers"].append(
                    {"track": track, "mode": "pytorch-bf16", "gpu_only": True}
                )
                save(args.output / "manifest.json", manifest)
        if "matched-history" in tracks:
            analyze(args)
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
