#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check the complete Qwen runtime operator, including Lt projection, against CPU."""
# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import native_inference
from runtime import INSTALL, binding, run, session
from selector_reference import token_support
from verify_handoff import SOURCE, WORK, sha

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from safetensors.torch import load_file


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WORK / "native-gqa")
    parser.add_argument(
        "--kernel-receipt",
        type=Path,
        default=WORK / "native-gqa/kernel-validation.json",
    )
    args = parser.parse_args()
    assert native_inference.WORK == WORK
    torch.set_num_threads(2)
    task = args.output_dir
    task.mkdir(parents=True, exist_ok=True)
    control = json.loads(args.kernel_receipt.read_text())
    assert control["status"] == "passed"
    assert sha(INSTALL / "bin/custom_kernels_gfx1151.dll") == control["kernel_sha256"]
    fixture_path = WORK / "selector-fixtures-v1/fixtures.safetensors"
    assert sha(fixture_path) == control["fixtures_sha256"]
    fixture = load_file(fixture_path)
    weights = load_file(
        SOURCE / "checkpoints/hip-ep-qwen/303042-step-0128/selector-bf16.safetensors"
    )
    initializers = []
    for kind in ["projection", "q_norm", "token_k_norm", "pool_k_norm"]:
        value = weights[f"selectors.27.{kind}.weight"].contiguous()
        tensor = TensorProto(name=kind, data_type=TensorProto.BFLOAT16)
        tensor.dims.extend(value.shape)
        tensor.raw_data = value.view(torch.uint8).numpy().tobytes()
        initializers.append(tensor)
    # Use the verified native FP32 RoPE cache; the CPU fixture includes its
    # first 2064 rows and already matches the source selector exactly.
    source = WORK / "default-rtn-w4a16-64/default.onnx"
    graph = onnx.load(source, load_external_data=False).graph
    for kind in ["cos", "sin"]:
        value = next(
            x for x in graph.initializer if x.name == f"model.rotary_emb.{kind}_cache"
        )
        descriptor = {x.key: x.value for x in value.external_data}
        mapped = np.memmap(
            source.parent / descriptor["location"],
            dtype=np.float32,
            mode="r",
            offset=int(descriptor["offset"]),
            shape=tuple(value.dims),
        )
        initializers.append(onnx.numpy_helper.from_array(np.asarray(mapped), name=kind))
    F16, I32, I64 = TensorProto.FLOAT16, TensorProto.INT32, TensorProto.INT64
    S, C = 2064, 2080
    specs = [
        ("q", F16, [1, "S", 6144]),
        ("k", F16, [1, "S", 1024]),
        ("v", F16, [1, "S", 1024]),
        ("past_key", F16, [1, 4, C, 256]),
        ("past_value", F16, [1, 4, C, 256]),
        ("length", I32, [1]),
        ("capacity", I32, []),
        ("hidden", F16, [1, "S", 5120]),
        ("positions", I64, [3, 1, "S"]),
    ]
    node = helper.make_node(
        "QwenQsaGroupQueryAttention",
        [
            "q",
            "k",
            "v",
            "past_key",
            "past_value",
            "length",
            "capacity",
            "hidden",
            "projection",
            "q_norm",
            "token_k_norm",
            "pool_k_norm",
            "cos",
            "sin",
            "positions",
        ],
        ["output", "present_key", "present_value"],
        domain="com.amd",
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "qwen_operator_control",
            [helper.make_tensor_value_info(*spec) for spec in specs],
            [
                helper.make_tensor_value_info("output", F16, [1, "S", 6144]),
                helper.make_tensor_value_info("present_key", F16, [1, 4, C, 256]),
                helper.make_tensor_value_info("present_value", F16, [1, 4, C, 256]),
            ],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.amd", 1)],
        ir_version=10,
    )
    path = task / "operator-control.onnx"
    onnx.save_model(model, path)
    sess = session(path)
    generator = torch.Generator(device="cpu").manual_seed(20261128)
    cpu_q = torch.randn(1, S, 24, 256, generator=generator).half()
    cpu_k = torch.randn(1, S, 4, 256, generator=generator).half()
    cpu_v = torch.randn(1, S, 4, 256, generator=generator).half()
    q, k, v, hidden = [x.cuda() for x in [cpu_q, cpu_k, cpu_v, fixture["hidden_fp16"]]]
    cache_k = torch.empty(1, 4, C, 256, device="cuda", dtype=torch.float16)
    cache_v = torch.empty_like(cache_k)
    rows = []
    result = {
        "status": "running",
        "rows": rows,
        "training_updates": 0,
        "graph_sha256": sha(path),
        "kernel_validation_sha256": sha(args.kernel_receipt),
        "install": str(INSTALL),
        "validator_sha256": sha(Path(__file__)),
    }

    def call(start, count, length_override=None, invalid_position=False):
        inputs = {
            name: value[:, start : start + count].reshape(1, count, -1).contiguous()
            for name, value in [("q", q), ("k", k), ("v", v), ("hidden", hidden)]
        }
        inputs.update(
            past_key=cache_k,
            past_value=cache_v,
            length=torch.tensor([start + count - 1], device="cuda", dtype=torch.int32),
            capacity=torch.tensor(C, device="cuda", dtype=torch.int32),
            positions=torch.arange(
                start, start + count, device="cuda", dtype=torch.int64
            )[None, None].repeat(3, 1, 1),
        )
        if length_override is not None:
            inputs["length"].fill_(length_override)
        if invalid_position:
            inputs["positions"][2, 0, 0] += 1
        output = torch.empty(1, count, 6144, device="cuda", dtype=torch.float16)
        run(
            sess,
            binding(
                sess,
                inputs,
                {"output": output, "present_key": cache_k, "present_value": cache_v},
            ),
        )
        return output.cpu().reshape(1, count, 24, 256)

    try:
        for label, start, options in [
            ("decode_without_prefill", 1, {}),
            ("int32_max_length", 0, {"length_override": 2**31 - 1}),
            ("invalid_negative_length", 0, {"length_override": -2}),
            ("noncanonical_position", 0, {"invalid_position": True}),
        ]:
            try:
                call(start, 1, **options)
            except RuntimeError as exc:
                assert "inference_compute" in str(exc), str(exc)
                rows.append({"check": label, "rejected": True, "error": str(exc)})
            else:
                raise AssertionError(f"Invalid request accepted: {label}")
        recovered = call(0, 17)
        assert recovered.isfinite().all() and call(17, 1).isfinite().all()
        assert torch.equal(recovered, call(0, 17))
        rows.append({"check": "error_flag_reset_and_prefill_replay", "exact": True})
        full = call(0, S)
        assert full.isfinite().all()
        assert torch.equal(cache_k[:, :, :S].cpu(), cpu_k.transpose(1, 2))
        assert torch.equal(cache_v[:, :, :S].cpu(), cpu_v.transpose(1, 2))
        outputs = {"full": full}
        prefix = 1927
        incremental = [call(0, prefix)[:, -1:]]
        for pos in range(prefix, S):
            incremental.append(call(pos, 1))
        outputs["cached"] = torch.cat(incremental, 1)
        positions = fixture["query_positions"].tolist()
        for pos in [2051, 2052, 2053, 2054, 2055, 2056, 2059, 2063]:
            selected = fixture["layer27.selected_groups"][positions.index(pos)]
            support = token_support(selected, pos)
            keys = cpu_k[0, support].float().repeat_interleave(6, dim=1).transpose(0, 1)
            values = (
                cpu_v[0, support].float().repeat_interleave(6, dim=1).transpose(0, 1)
            )
            logits = (cpu_q[0, pos].float()[:, None] * keys).sum(-1) * 0.0625
            expected = (logits.softmax(-1)[:, :, None] * values).sum(1)
            for kind, output in outputs.items():
                actual = output[0, pos if kind == "full" else pos - prefix + 1].float()
                error = float(
                    (actual - expected).square().mean().sqrt()
                    / expected.square().mean().sqrt()
                )
                assert error < 0.002, (kind, pos, error)
                rows.append({"path": kind, "position": pos, "relative_rms": error})
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        (task / "operator-validation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print("Complete native Qwen operator matches frozen CPU oracle", rows, flush=True)


if __name__ == "__main__":
    main()
