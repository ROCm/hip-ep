"""Profile the mixed PyTorch + OGA/HIP EP DFlash pipeline.

The default run measures the same ten GSM8K prompts at block sizes 2, 4, 8,
and 16, alongside a clean OGA baseline and the adapter's block-size-1
baseline. Use one block size and one sample when collecting native traces.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack, nullcontext
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from benchmark_oga import (
    DEFAULT_DRAFT,
    DEFAULT_HF_HOME,
    DEFAULT_TOKENIZER,
    _apply_chat_template,
    load_and_process_dataset,
)
from oga_target_native import (
    DEFAULT_HIPEP_ROOT,
    DEFAULT_MODEL_DIR,
    DEFAULT_ROCM_ROOT,
    OgaDFlashTarget,
    dflash_generate_oga,
    ensure_runtime_environment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StageEvent:
    name: str
    elapsed_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    index: int
    context_length: int
    block_size: int
    draft_cache_length: int
    start_ns: int
    elapsed_ns: int = 0
    accepted_drafts: int = 0
    committed_count: int = 0
    partial: bool = False
    replay_rows: int = 0
    correction_mismatch: bool = False
    stages: list[StageEvent] = field(default_factory=list)


@dataclass
class GenerationRecord:
    mode: str
    block_size: int
    input_tokens: int
    max_new_tokens: int
    sample_index: int
    dataset_index: int
    start_ns: int
    decode_start_ns: int = 0
    elapsed_ns: int = 0
    decode_elapsed_ns: int = 0
    output_tokens: int = 0
    iterations_count: int = 0
    partial_rollbacks: int = 0
    correction_mismatches: int = 0
    stages: list[StageEvent] = field(default_factory=list)
    iterations: list[IterationRecord] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    boxed_answer: str | None = None


class _Roctx:
    def __init__(self, rocm_root: Path) -> None:
        candidates = sorted(
            (rocm_root / "lib").glob("librocprofiler-sdk-roctx.so*")
        )
        if not candidates:
            candidates = sorted((rocm_root / "lib").glob("libroctx64.so*"))
        if not candidates:
            raise RuntimeError(f"Could not find an ROCTx library under {rocm_root}")
        self.library = ctypes.CDLL(str(candidates[0]))
        self.library.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self.library.roctxRangePushA.restype = ctypes.c_int
        self.library.roctxRangePop.argtypes = []
        self.library.roctxRangePop.restype = ctypes.c_int

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        self.library.roctxRangePushA(name.encode("utf-8"))
        try:
            yield
        finally:
            self.library.roctxRangePop()


class ProfileRecorder:
    """Collect nested stage timings without importing a profiling framework."""

    def __init__(
        self,
        *,
        torch_ranges: bool = False,
        roctx_ranges: bool = False,
        rocm_root: Path = DEFAULT_ROCM_ROOT,
    ) -> None:
        self.torch_ranges = torch_ranges
        self.roctx = _Roctx(rocm_root) if roctx_ranges else None
        self.generations: list[GenerationRecord] = []
        self.context: dict[str, int] = {"sample_index": -1, "dataset_index": -1}
        self.current_generation: GenerationRecord | None = None
        self.current_iteration: IterationRecord | None = None

    def set_context(self, *, sample_index: int, dataset_index: int) -> None:
        self.context = {
            "sample_index": int(sample_index),
            "dataset_index": int(dataset_index),
        }

    def begin_generation(
        self,
        *,
        mode: str,
        block_size: int,
        input_tokens: int,
        max_new_tokens: int,
    ) -> None:
        if self.current_generation is not None:
            raise RuntimeError("A profiled generation is already active")
        self.current_generation = GenerationRecord(
            mode=mode,
            block_size=int(block_size),
            input_tokens=int(input_tokens),
            max_new_tokens=int(max_new_tokens),
            sample_index=self.context["sample_index"],
            dataset_index=self.context["dataset_index"],
            start_ns=time.perf_counter_ns(),
        )

    def mark_decode_start(self) -> None:
        if self.current_generation is None:
            raise RuntimeError("No profiled generation is active")
        self.current_generation.decode_start_ns = time.perf_counter_ns()

    def begin_iteration(
        self,
        *,
        index: int,
        context_length: int,
        block_size: int,
        draft_cache_length: int,
    ) -> None:
        if self.current_generation is None:
            raise RuntimeError("No profiled generation is active")
        if self.current_iteration is not None:
            raise RuntimeError("A profiled iteration is already active")
        self.current_iteration = IterationRecord(
            index=int(index),
            context_length=int(context_length),
            block_size=int(block_size),
            draft_cache_length=int(draft_cache_length),
            start_ns=time.perf_counter_ns(),
        )

    def end_iteration(
        self,
        *,
        accepted_drafts: int,
        committed_count: int,
        partial: bool,
        replay_rows: int,
        correction_mismatch: bool,
    ) -> None:
        if self.current_generation is None or self.current_iteration is None:
            raise RuntimeError("No profiled iteration is active")
        current = self.current_iteration
        current.elapsed_ns = time.perf_counter_ns() - current.start_ns
        current.accepted_drafts = int(accepted_drafts)
        current.committed_count = int(committed_count)
        current.partial = bool(partial)
        current.replay_rows = int(replay_rows)
        current.correction_mismatch = bool(correction_mismatch)
        self.current_generation.iterations.append(current)
        self.current_iteration = None

    def end_generation(
        self,
        *,
        output_tokens: int,
        iterations: int,
        partial_rollbacks: int,
        correction_mismatches: int,
        generation_correction_mismatches: int,
    ) -> None:
        del correction_mismatches  # The target-level value is cumulative across runs.
        if self.current_generation is None:
            raise RuntimeError("No profiled generation is active")
        if self.current_iteration is not None:
            raise RuntimeError("Cannot finish a generation with an active iteration")
        current = self.current_generation
        end_ns = time.perf_counter_ns()
        current.elapsed_ns = end_ns - current.start_ns
        current.decode_elapsed_ns = (
            end_ns - current.decode_start_ns if current.decode_start_ns else 0
        )
        current.output_tokens = int(output_tokens)
        current.iterations_count = int(iterations)
        current.partial_rollbacks = int(partial_rollbacks)
        current.correction_mismatches = int(generation_correction_mismatches)
        self.generations.append(current)
        self.current_generation = None

    def record_stage(
        self,
        name: str,
        elapsed_ns: int,
        metadata: dict[str, Any],
    ) -> None:
        event = StageEvent(
            name=name,
            elapsed_ns=int(elapsed_ns),
            metadata=dict(metadata),
        )
        if self.current_iteration is not None:
            self.current_iteration.stages.append(event)
        elif self.current_generation is not None:
            self.current_generation.stages.append(event)
        else:
            raise RuntimeError(f"Stage {name!r} was recorded outside a generation")

    def trace_context(
        self,
        name: str,
        metadata: dict[str, Any],
    ) -> Any:
        del metadata
        stack = ExitStack()
        if self.torch_ranges:
            import torch

            stack.enter_context(torch.profiler.record_function(name))
        if self.roctx is not None:
            stack.enter_context(self.roctx.range(name))
        return stack


@contextmanager
def _timed_stage(
    recorder: ProfileRecorder | None,
    name: str,
    **metadata: Any,
) -> Iterator[dict[str, Any]]:
    dynamic: dict[str, Any] = {}
    if recorder is None:
        yield dynamic
        return
    start_ns = time.perf_counter_ns()
    with recorder.trace_context(name, metadata):
        yield dynamic
    recorder.record_stage(
        name,
        time.perf_counter_ns() - start_ns,
        {**metadata, **dynamic},
    )


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(text[content_start:index].split()).replace(",", "")
    return None


def _clean_oga_generate(
    target: OgaDFlashTarget,
    input_ids: np.ndarray,
    *,
    max_new_tokens: int,
    stop_token_ids: list[int],
    recorder: ProfileRecorder | None,
) -> SimpleNamespace:
    tokens = np.ascontiguousarray(input_ids.reshape(-1), dtype=np.int32)
    if recorder is not None:
        recorder.begin_generation(
            mode="clean_oga",
            block_size=1,
            input_tokens=int(tokens.size),
            max_new_tokens=int(max_new_tokens),
        )

    with _timed_stage(recorder, "clean_oga.generator_create"):
        params = target._og.GeneratorParams(target.model)
        params.set_search_options(
            max_length=int(tokens.size) + int(max_new_tokens),
            do_sample=False,
        )
        generator = target._og.Generator(target.model, params)

    prefill_start_ns = time.perf_counter_ns()
    with _timed_stage(
        recorder,
        "clean_oga.prefill",
        rows=int(tokens.size),
        input_bytes=int(tokens.nbytes),
    ):
        generator.append_tokens(tokens)
    prefill_elapsed_ns = time.perf_counter_ns() - prefill_start_ns

    if recorder is not None:
        recorder.mark_decode_start()
    decode_start_ns = time.perf_counter_ns()
    generated: list[int] = []
    while len(generated) < max_new_tokens and not generator.is_done():
        if recorder is not None:
            recorder.begin_iteration(
                index=len(generated),
                context_length=int(tokens.size) + len(generated),
                block_size=1,
                draft_cache_length=0,
            )
        with _timed_stage(recorder, "clean_oga.generate_next_token"):
            generator.generate_next_token()
        with _timed_stage(recorder, "clean_oga.get_next_token") as profile:
            next_tokens = np.asarray(generator.get_next_tokens())
            profile["output_bytes"] = int(next_tokens.nbytes)
            token = int(next_tokens[0])
        generated.append(token)
        if recorder is not None:
            recorder.end_iteration(
                accepted_drafts=0,
                committed_count=1,
                partial=False,
                replay_rows=0,
                correction_mismatch=False,
            )
        if token in stop_token_ids:
            break

    decode_elapsed_ns = time.perf_counter_ns() - decode_start_ns
    if recorder is not None:
        recorder.end_generation(
            output_tokens=len(generated),
            iterations=len(generated),
            partial_rollbacks=0,
            correction_mismatches=0,
            generation_correction_mismatches=0,
        )
    return SimpleNamespace(
        output_token_ids=generated,
        num_output_tokens=len(generated),
        time_to_first_token=prefill_elapsed_ns / 1e9,
        time_per_output_token=(
            decode_elapsed_ns / 1e9 / len(generated) if generated else 0.0
        ),
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _generation_key(record: GenerationRecord) -> tuple[str, int]:
    return record.mode, record.block_size


def _aggregate_group(records: list[GenerationRecord]) -> dict[str, Any]:
    output_tokens = sum(record.output_tokens for record in records)
    decode_ns = sum(record.decode_elapsed_ns for record in records)
    iterations = [
        iteration
        for record in records
        for iteration in record.iterations
    ]
    iteration_ms = [iteration.elapsed_ns / 1e6 for iteration in iterations]
    committed = [iteration.committed_count for iteration in iterations]
    partial = [iteration for iteration in iterations if iteration.partial]

    stage_events: dict[str, list[StageEvent]] = defaultdict(list)
    for iteration in iterations:
        for event in iteration.stages:
            stage_events[event.name].append(event)

    stages = {}
    for name, events in sorted(stage_events.items()):
        durations_ms = [event.elapsed_ns / 1e6 for event in events]
        output_bytes = sum(
            int(event.metadata.get("output_bytes", 0)) for event in events
        )
        input_bytes = sum(
            int(event.metadata.get("input_bytes", 0)) for event in events
        )
        stages[name] = {
            "calls": len(events),
            "total_ms": sum(durations_ms),
            "mean_ms": statistics.fmean(durations_ms),
            "p50_ms": statistics.median(durations_ms),
            "p95_ms": _percentile(durations_ms, 0.95),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        }

    return {
        "samples": len(records),
        "output_tokens": output_tokens,
        "decode_seconds": decode_ns / 1e9,
        "throughput_tps": output_tokens / (decode_ns / 1e9) if decode_ns else 0.0,
        "iterations": len(iterations),
        "mean_iteration_ms": (
            statistics.fmean(iteration_ms) if iteration_ms else 0.0
        ),
        "p50_iteration_ms": statistics.median(iteration_ms) if iteration_ms else 0.0,
        "p95_iteration_ms": _percentile(iteration_ms, 0.95),
        "mean_committed_count": (
            statistics.fmean(committed) if committed else 0.0
        ),
        "mean_progress": output_tokens / len(iterations) if iterations else 0.0,
        "full_accept_rate": (
            sum(not iteration.partial for iteration in iterations) / len(iterations)
            if iterations
            else 0.0
        ),
        "partial_accept_rate": len(partial) / len(iterations) if iterations else 0.0,
        "mean_partial_replay_rows": (
            statistics.fmean(iteration.replay_rows for iteration in partial)
            if partial
            else 0.0
        ),
        "correction_mismatches": sum(
            record.correction_mismatches for record in records
        ),
        "stages": stages,
    }


def _add_parity(
    summary: dict[str, Any],
    records: list[GenerationRecord],
) -> None:
    adapter = {
        record.dataset_index: record
        for record in records
        if record.mode == "adapter_baseline"
    }
    clean = {
        record.dataset_index: record
        for record in records
        if record.mode == "clean_oga"
    }
    for (mode, block_size), group in summary["groups"].items():
        if mode == "clean_oga":
            continue
        reference = clean if mode == "adapter_baseline" else adapter
        comparable = [
            record for record in records
            if _generation_key(record) == (mode, block_size)
            and record.dataset_index in reference
        ]
        token_exact = sum(
            record.output_token_ids
            == reference[record.dataset_index].output_token_ids
            for record in comparable
        )
        boxed_comparable = [
            record for record in comparable
            if reference[record.dataset_index].boxed_answer is not None
            and record.boxed_answer is not None
        ]
        boxed_equal = sum(
            record.boxed_answer == reference[record.dataset_index].boxed_answer
            for record in boxed_comparable
        )
        group["parity"] = {
            "reference": "clean_oga" if mode == "adapter_baseline" else "adapter_baseline",
            "token_exact": token_exact,
            "token_comparable": len(comparable),
            "boxed_equal": boxed_equal,
            "boxed_comparable": len(boxed_comparable),
        }


def _summarize(records: list[GenerationRecord]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[GenerationRecord]] = defaultdict(list)
    for record in records:
        grouped[_generation_key(record)].append(record)
    summary: dict[str, Any] = {
        "groups": {
            key: _aggregate_group(group)
            for key, group in sorted(grouped.items())
        }
    }
    _add_parity(summary, records)

    clean = summary["groups"].get(("clean_oga", 1))
    if clean is not None and clean["throughput_tps"]:
        clean_tpot_ms = 1000.0 / clean["throughput_tps"]
        for (mode, _), group in summary["groups"].items():
            if mode != "dflash":
                continue
            group["break_even"] = {
                "clean_oga_tpot_ms": clean_tpot_ms,
                "budget_ms_per_iteration": clean_tpot_ms * group["mean_progress"],
                "actual_ms_per_iteration": group["mean_iteration_ms"],
                "iteration_budget_ratio": (
                    group["mean_iteration_ms"]
                    / (clean_tpot_ms * group["mean_progress"])
                    if group["mean_progress"]
                    else 0.0
                ),
            }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== OGA/HIP EP DFlash profile summary ===")
    for (mode, block_size), group in summary["groups"].items():
        print(
            f"\n[{mode} block={block_size}] "
            f"{group['throughput_tps']:.2f} tok/s, "
            f"{group['mean_iteration_ms']:.2f} ms/iteration, "
            f"progress={group['mean_progress']:.2f}, "
            f"full={group['full_accept_rate'] * 100:.1f}%, "
            f"partial={group['partial_accept_rate'] * 100:.1f}%"
        )
        if break_even := group.get("break_even"):
            print(
                "  break-even: "
                f"budget={break_even['budget_ms_per_iteration']:.2f} ms, "
                f"actual={break_even['actual_ms_per_iteration']:.2f} ms, "
                f"ratio={break_even['iteration_budget_ratio']:.2f}x"
            )
        if parity := group.get("parity"):
            print(
                f"  parity vs {parity['reference']}: "
                f"tokens={parity['token_exact']}/{parity['token_comparable']}, "
                f"boxed={parity['boxed_equal']}/{parity['boxed_comparable']}"
            )

        decode_ms = group["decode_seconds"] * 1000.0
        leaf_stages = [
            (name, stats)
            for name, stats in group["stages"].items()
            if not name.endswith(".total")
        ]
        leaf_stages.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        print("  top steady-state stages:")
        for name, stats in leaf_stages[:16]:
            percent = 100.0 * stats["total_ms"] / decode_ms if decode_ms else 0.0
            print(
                f"    {name:<43} "
                f"{stats['total_ms']:>10.2f} ms "
                f"{percent:>6.1f}% "
                f"{stats['mean_ms']:>8.3f} ms/call "
                f"p95={stats['p95_ms']:.3f}"
            )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            (
                f"{key[0]}:block={key[1]}"
                if isinstance(key, tuple)
                else str(key)
            ): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _aggregate_events(events: list[StageEvent]) -> dict[str, Any]:
    totals_ns: dict[str, int] = defaultdict(int)
    input_bytes: dict[str, int] = defaultdict(int)
    output_bytes: dict[str, int] = defaultdict(int)
    for event in events:
        totals_ns[event.name] += event.elapsed_ns
        input_bytes[event.name] += int(event.metadata.get("input_bytes", 0))
        output_bytes[event.name] += int(event.metadata.get("output_bytes", 0))
    return {
        name: {
            "elapsed_ms": elapsed_ns / 1e6,
            "input_bytes": input_bytes[name],
            "output_bytes": output_bytes[name],
        }
        for name, elapsed_ns in sorted(totals_ns.items())
    }


def _compact_generation(record: GenerationRecord) -> dict[str, Any]:
    return {
        "mode": record.mode,
        "block_size": record.block_size,
        "input_tokens": record.input_tokens,
        "max_new_tokens": record.max_new_tokens,
        "sample_index": record.sample_index,
        "dataset_index": record.dataset_index,
        "elapsed_ms": record.elapsed_ns / 1e6,
        "decode_elapsed_ms": record.decode_elapsed_ns / 1e6,
        "output_tokens": record.output_tokens,
        "iterations_count": record.iterations_count,
        "partial_rollbacks": record.partial_rollbacks,
        "correction_mismatches": record.correction_mismatches,
        "prefill_stages": _aggregate_events(record.stages),
        "iterations": [
            {
                "index": iteration.index,
                "context_length": iteration.context_length,
                "block_size": iteration.block_size,
                "draft_cache_length": iteration.draft_cache_length,
                "elapsed_ms": iteration.elapsed_ns / 1e6,
                "accepted_drafts": iteration.accepted_drafts,
                "committed_count": iteration.committed_count,
                "partial": iteration.partial,
                "replay_rows": iteration.replay_rows,
                "correction_mismatch": iteration.correction_mismatch,
                "stages": _aggregate_events(iteration.stages),
            }
            for iteration in record.iterations
        ],
        "output_token_ids": record.output_token_ids,
        "output_text": record.output_text,
        "boxed_answer": record.boxed_answer,
    }


def _prepare_samples(
    tokenizer: Any,
    *,
    max_samples: int,
    sample_indices: list[int] | None,
    enable_thinking: bool,
) -> list[dict[str, Any]]:
    dataset = load_and_process_dataset("gsm8k")
    random.seed(0)
    random.shuffle(dataset)
    selected_indices = (
        sample_indices
        if sample_indices is not None
        else list(range(min(max_samples, len(dataset))))
    )
    samples = []
    for sample_index, dataset_index in enumerate(selected_indices):
        instance = dataset[dataset_index]
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = _apply_chat_template(
            tokenizer,
            messages,
            enable_thinking,
        )
        samples.append(
            {
                "sample_index": sample_index,
                "dataset_index": dataset_index,
                "input_text": input_text,
            }
        )
    return samples


def _load_draft_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from dflash.model import DFlashDraftModel

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    draft_config = AutoConfig.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
    )
    if not hasattr(draft_config, "block_size"):
        draft_config.block_size = draft_config.dflash_config["block_size"]
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.draft_device).eval()
    return torch, tokenizer, draft


def _run_mode(
    *,
    mode: str,
    block_size: int,
    samples: list[dict[str, Any]],
    target: OgaDFlashTarget,
    draft: Any,
    tokenizer: Any,
    recorder: ProfileRecorder | None,
    max_new_tokens: int,
    draft_device: str,
) -> None:
    stop_ids = (
        [int(tokenizer.eos_token_id)]
        if tokenizer.eos_token_id is not None
        else []
    )
    for sample in samples:
        input_ids = tokenizer.encode(
            sample["input_text"],
            return_tensors="pt",
        ).to(draft_device)
        if recorder is not None:
            recorder.set_context(
                sample_index=sample["sample_index"],
                dataset_index=sample["dataset_index"],
            )
        if mode == "clean_oga":
            target.profile_sink = None
            response = _clean_oga_generate(
                target,
                input_ids.detach().cpu().numpy(),
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
                recorder=recorder,
            )
            generated_ids = response.output_token_ids
        else:
            target.profile_sink = recorder
            response = dflash_generate_oga(
                draft,
                target,
                input_ids,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids or None,
                temperature=0.0,
                block_size=block_size,
                return_stats=True,
            )
            generated_ids = (
                response.output_ids[0, response.num_input_tokens:]
                .detach()
                .cpu()
                .tolist()
            )

        if recorder is not None:
            record = recorder.generations[-1]
            record.output_token_ids = [int(token) for token in generated_ids]
            record.output_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            record.boxed_answer = _extract_last_boxed(record.output_text)


def _mode_specs(args: argparse.Namespace) -> list[tuple[str, int]]:
    modes = []
    if not args.skip_clean_baseline:
        modes.append(("clean_oga", 1))
    if not args.skip_adapter_baseline:
        modes.append(("adapter_baseline", 1))
    modes.extend(("dflash", int(block_size)) for block_size in args.block_sizes)
    return modes


def _warmup(
    *,
    modes: list[tuple[str, int]],
    sample: dict[str, Any],
    target: OgaDFlashTarget,
    draft: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> None:
    if args.warmup_new_tokens <= 0:
        return
    print(f"[warmup] {args.warmup_new_tokens} tokens per mode", flush=True)
    for mode, block_size in modes:
        _run_mode(
            mode=mode,
            block_size=block_size,
            samples=[sample],
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            recorder=None,
            max_new_tokens=min(args.warmup_new_tokens, args.max_new_tokens),
            draft_device=args.draft_device,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--draft-model", "--draft", default=DEFAULT_DRAFT)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
    )
    parser.add_argument("--warmup-new-tokens", type=int, default=32)
    parser.add_argument("--skip-clean-baseline", action="store_true")
    parser.add_argument("--skip-adapter-baseline", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--draft-device", default="cpu")
    parser.add_argument("--hipep-root", type=Path, default=DEFAULT_HIPEP_ROOT)
    parser.add_argument("--rocm-root", type=Path, default=DEFAULT_ROCM_ROOT)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hipep_study/profile/oga_dflash_profile.json"),
    )
    parser.add_argument(
        "--torch-trace",
        type=Path,
        default=None,
        help="Write a CPU PyTorch Chrome trace. Use one sample and one block size.",
    )
    parser.add_argument(
        "--roctx-ranges",
        action="store_true",
        help="Emit stage ranges for an external rocprofv3 native trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if any(block_size < 2 for block_size in args.block_sizes):
        raise SystemExit("--block-sizes values must all be at least 2")
    if args.max_samples < 1 or args.max_new_tokens < 1:
        raise SystemExit("--max-samples and --max-new-tokens must be positive")
    selected_sample_count = (
        len(args.sample_indices)
        if args.sample_indices is not None
        else args.max_samples
    )
    if (args.torch_trace is not None or args.roctx_ranges) and (
        selected_sample_count != 1
        or len(args.block_sizes) != 1
        or not args.skip_clean_baseline
        or not args.skip_adapter_baseline
    ):
        raise SystemExit(
            "Trace collection requires one sample, one block size, and both "
            "--skip-*-baseline flags to keep the trace bounded"
        )

    ensure_runtime_environment(args.hipep_root, args.rocm_root)
    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.hf_home))

    target = OgaDFlashTarget(args.target, args.hipep_root)
    torch, tokenizer, draft = _load_draft_and_tokenizer(args)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    samples = _prepare_samples(
        tokenizer,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
        enable_thinking=args.enable_thinking,
    )
    modes = _mode_specs(args)
    print(
        f"[profile] samples={len(samples)} max_new_tokens={args.max_new_tokens} "
        f"modes={modes}",
        flush=True,
    )
    print(f"[profile] providers={target.providers}", flush=True)

    _warmup(
        modes=modes,
        sample=samples[0],
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        args=args,
    )

    recorder = ProfileRecorder(
        torch_ranges=args.torch_trace is not None,
        roctx_ranges=args.roctx_ranges,
        rocm_root=args.rocm_root,
    )
    torch_profile: Any = nullcontext()
    if args.torch_trace is not None:
        args.torch_trace.parent.mkdir(parents=True, exist_ok=True)
        torch_profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        )

    with torch_profile as active_torch_profile:
        for mode, block_size in modes:
            print(f"[profile] running {mode} block={block_size}", flush=True)
            _run_mode(
                mode=mode,
                block_size=block_size,
                samples=samples,
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                recorder=recorder,
                max_new_tokens=args.max_new_tokens,
                draft_device=args.draft_device,
            )

    if args.torch_trace is not None:
        active_torch_profile.export_chrome_trace(str(args.torch_trace))
        print(f"[profile] PyTorch trace: {args.torch_trace}", flush=True)

    summary = _summarize(recorder.generations)
    _print_summary(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "target": str(Path(args.target).resolve()),
            "draft_model": args.draft_model,
            "tokenizer": args.tokenizer,
            "max_samples": len(samples),
            "dataset_indices": [sample["dataset_index"] for sample in samples],
            "max_new_tokens": args.max_new_tokens,
            "block_sizes": args.block_sizes,
            "warmup_new_tokens": args.warmup_new_tokens,
            "draft_device": args.draft_device,
            "providers": target.providers,
            "roctx_ranges": args.roctx_ranges,
        },
        "summary": _json_ready(summary),
        "generations": [
            _compact_generation(record)
            for record in recorder.generations
        ],
    }
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[profile] report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
