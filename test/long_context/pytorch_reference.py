#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Optional BF16 reference worker; invoked explicitly in a separate environment.

Uses the same token IDs and canonical histories as the OGA characterization.
BF16-versus-OGA differences include weight quantization and framework differences.
The PyTorch reference is not an isolated measurement of KV quantization error.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from fixture import score_answer
from metrics import compare_logits
from run_turboquant_multiturn import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oga-model", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--track", choices=("matched-history", "free-running"), required=True
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    out = args.results / args.track / "pytorch-bf16"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "mode": "pytorch-bf16",
        "track": args.track,
        "torch": torch.__version__,
        "hip_runtime": torch.version.hip,
        "numpy": np.__version__,
        "numpy_module": np.__file__,
        "transformers": transformers.__version__,
        "model": str(args.model.resolve()),
        "turns": [],
        "attention": "sdpa",
        "timing_scope": "diagnostic including logits readback",
    }
    save(out / "receipt.json", report)
    try:
        # Earlier Transformers versions silently reset recurrent state on append.
        # Pin a known state-aware implementation rather than patching the model.
        source = inspect.getsource(Qwen3_5GatedDeltaNet.forward)
        if "initial_state=recurrent_state" not in source:
            raise RuntimeError(
                "Use state-aware Qwen3.5 append (validated with Transformers 5.22.1)"
            )
        report["gated_delta_forward_sha256"] = hashlib.sha256(
            source.encode()
        ).hexdigest()
        fixture = json.loads(
            (args.results / "fixture.json").read_text(encoding="utf-8")
        )
        report["fixture_sha256"] = digest(args.results / "fixture.json")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        oga_tokenizer = Tokenizer.from_file(str(args.oga_model / "tokenizer.json"))
        assert tokenizer.get_vocab() == oga_tokenizer.get_vocab(), (
            "Vocabulary-to-ID mappings differ"
        )
        for turn in fixture["turns"]:
            assert (
                tokenizer.decode(turn["tokens"], skip_special_tokens=True)
                == turn["text"]
            )
        report["tokenizer_mapping_equal"] = True
        report["model_files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in args.model.iterdir()
            if p.suffix in (".json", ".safetensors")
        }
        torch.manual_seed(0)
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        assert not loading.get("missing_keys") and not loading.get("mismatched_keys"), (
            loading
        )
        assert not loading.get("unexpected_keys"), loading
        model.eval()
        assert all(
            p.device.type == "cuda" and p.dtype == torch.bfloat16
            for p in model.parameters()
        )
        report.update(
            device=torch.cuda.get_device_name(),
            parameter_bytes=sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            loading={
                k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()
            },
            gpu_only=True,
        )
        save(out / "receipt.json", report)
        cache = None

        @torch.inference_mode()
        def forward(tokens, past):
            result = model(
                input_ids=torch.tensor([tokens], device="cuda", dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = result.logits[0, -1].float().cpu().numpy().copy()
            assert np.isfinite(logits).all()
            return logits, result.past_key_values

        def hashes(past, length):
            values = {}
            for index in range(3, 64, 4):
                layer = past.layers[index]
                for kind in ("keys", "values"):
                    tensor = getattr(layer, kind)[:, :, :length].contiguous()
                    assert (
                        tensor.dtype == torch.bfloat16 and tensor.device.type == "cuda"
                    )
                    values[f"{index}.{kind}"] = hashlib.sha256(
                        tensor.view(torch.uint8).cpu().numpy()
                    ).hexdigest()
            return values

        # Independent replay sanity check before the actual conversation. Keep
        # measured BF16 rounding differences; do not invent a passing tolerance.
        probe = fixture["turns"][0]["tokens"][:64]
        full_logits, full_cache = forward(probe, None)
        del full_cache
        _, probe_cache = forward(probe[:32], None)
        append_logits, probe_cache = forward(probe[32:], probe_cache)
        assert probe_cache.get_seq_length() == len(probe)
        report["append_replay_sanity"] = compare_logits(
            full_logits, append_logits, int(np.argmax(full_logits))
        )
        del probe_cache
        torch.cuda.empty_cache()
        control = None
        if args.track == "matched-history":
            control = json.loads(
                (args.results / "matched-history/no-ring/receipt.json").read_text()
            )
            assert control["status"] == "passed"
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        sequence = []
        for turn in fixture["turns"]:
            index, tokens = turn["turn"], turn["tokens"]
            old = hashes(cache, len(sequence)) if sequence else {}
            past = len(sequence)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            logits, cache = forward(tokens, cache)
            prefill_seconds = time.perf_counter() - start
            sequence.extend(tokens)
            assert cache.get_seq_length() == len(sequence)
            assert not past or hashes(cache, past) == old, "Old BF16 cache rows changed"
            prefix_hash = hashlib.sha256(
                np.array(sequence, np.int32).tobytes()
            ).hexdigest()
            continuation = (
                control["turns"][index - 1]["generated_tokens"] if control else None
            )
            if control:
                assert prefix_hash == control["turns"][index - 1]["prefix_sha256"]
            vectors, generated = [], []
            steps = (
                len(continuation) if continuation is not None else args.max_new_tokens
            )
            start = time.perf_counter()
            for step in range(steps):
                vectors.append(logits)
                token = (
                    continuation[step]
                    if continuation is not None
                    else int(np.argmax(logits))
                )
                generated.append(token)
                logits, cache = forward([token], cache)
                if continuation is None and token in eos:
                    break
            forced_end = continuation is None and generated[-1] != end_token
            if forced_end:
                vectors.append(logits)
                generated.append(end_token)
                logits, cache = forward([end_token], cache)
            decode_seconds = time.perf_counter() - start
            sequence.extend(generated)
            assert cache.get_seq_length() == len(sequence)
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            row = {
                "turn": index,
                "past": past,
                "prompt_tokens": len(tokens),
                "context_after": len(sequence),
                "prefix_sha256": prefix_hash,
                "generated_tokens": generated,
                "answer": answer,
                "forced_chat_end": forced_end,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "content_score": score_answer(answer, turn),
                "finite_logits": True,
                "old_cache_rows_unchanged": True,
                "kv_live_bytes": sum(
                    getattr(cache.layers[i], k).numel() * 2
                    for i in range(3, 64, 4)
                    for k in ("keys", "values")
                ),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
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
                        if k not in ("answer", "generated_tokens")
                    }
                ),
                flush=True,
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save(out / "receipt.json", report)


if __name__ == "__main__":
    main()
