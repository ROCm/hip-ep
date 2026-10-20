"""Build a read-only-derived ONNX target for DFlash.

The source OGA model is never written.  This script creates a replacement
``text.onnx`` graph that exposes the DFlash residual streams and final normalized
hidden state, plus a standalone ``lm_head.onnx`` graph. 
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper


DEFAULT_SOURCE = Path(
    "/proj/xcohdstaff7/devledas/nobkup/models/"
    "Qwen3.6-35B-A3B_gpuep_1.0_gpu_128"
)
DEFAULT_OUTPUT = Path(__file__).parent / "models/Qwen3.6-35B-A3B-hidden"
TARGET_LAYER_IDS_BY_DRAFT = {
    "z-lab/Qwen3.6-35B-A3B-DFlash": (1, 6, 11, 16, 22, 27, 32, 37),
    "z-lab/Qwen3.6-27B-DFlash": (1, 16, 31, 46, 61),
}
DFLASH_LOGITS = "dflash_logits"
HEAD_NODE_NAME = "/lm_head/MatMul_Q4"
HEAD_INITIALIZERS = (
    "lm_head.MatMul.weight_Q4",
    "lm_head.MatMul.weight_scales",
)
COPIED_ASSETS = (
    "text.onnx.data",
    "embedding.onnx",
    "embedding.onnx.data",
    "vision.onnx",
    "vision.onnx.data",
)
LINKED_ASSETS = (
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)


@dataclass(frozen=True)
class TargetModelSpec:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    target_layer_ids: tuple[int, ...]
    final_hidden_output: str

    @property
    def hidden_output_names(self) -> list[str]:
        return [
            f"/model/layers.{layer_id + 1}/input_layernorm/output_3"
            for layer_id in self.target_layer_ids
        ]


def _read_source_spec(
    source: Path,
    target_layer_ids: tuple[int, ...],
    final_hidden_output: str | None,
) -> TargetModelSpec:
    with (source / "genai_config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    model = config["model"]
    decoder = model["decoder"]
    num_hidden_layers = int(decoder["num_hidden_layers"])
    return TargetModelSpec(
        hidden_size=int(decoder["hidden_size"]),
        vocab_size=int(model["vocab_size"]),
        num_hidden_layers=num_hidden_layers,
        target_layer_ids=target_layer_ids,
        final_hidden_output=(
            final_hidden_output
            or f"/model/layers.{num_hidden_layers}/final_norm_layernorm/output_0"
        ),
    )


def _parse_target_layer_ids(value: str) -> tuple[int, ...]:
    try:
        layer_ids = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--target-layer-ids must be a comma-separated list of integers"
        ) from exc
    if not layer_ids:
        raise argparse.ArgumentTypeError("--target-layer-ids cannot be empty")
    return layer_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(source: Path) -> dict[str, dict[str, int | str]]:
    snapshot: dict[str, dict[str, int | str]] = {}
    for name in ("text.onnx", "text.onnx.data"):
        path = source / name
        stat = path.stat()
        snapshot[name] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
        }
    return snapshot


def _assert_safe_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve(strict=True)
    output = output.resolve(strict=False)
    if output == source or output.is_relative_to(source):
        raise ValueError(f"Output must not be inside the source model directory: {output}")
    for name in (
        "text.onnx",
        "text.onnx.data",
        "embedding.onnx",
        "embedding.onnx.data",
        "vision.onnx",
        "vision.onnx.data",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    return source, output


def _replace_output_dir(output: Path, force: bool) -> None:
    if output.exists() or output.is_symlink():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to rebuild it")
        if output.is_symlink() or output.is_file():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.mkdir(parents=True)


def _copy_value_info(graph: onnx.GraphProto, name: str) -> onnx.ValueInfoProto:
    for value in (*graph.value_info, *graph.output):
        if value.name == name:
            copied = onnx.ValueInfoProto()
            copied.CopyFrom(value)
            return copied
    raise KeyError(f"Tensor has no value_info entry: {name}")


def _build_text_graph(source: Path, output: Path, spec: TargetModelSpec) -> list[str]:
    source_graph = source / "text.onnx"
    model = onnx.load(str(source_graph), load_external_data=False)
    graph = model.graph
    produced = {name for node in graph.node for name in node.output if name}

    # expose the eight target hidden-state taps required by the DFlash draft:
    tap_names = spec.hidden_output_names
    required = [*tap_names, spec.final_hidden_output] #also add the final hidden state output 
    missing = [name for name in required if name not in produced]
    if missing:
        raise RuntimeError("Required tensors are absent from source graph:\n  " + "\n  ".join(missing))

    existing = {value.name for value in graph.output}
    for name in required:
        if name not in existing:
            graph.output.append(_copy_value_info(graph, name)) 

    head_node = next((node for node in graph.node if node.name == HEAD_NODE_NAME), None)
    if head_node is None:
        raise RuntimeError(f"Could not find {HEAD_NODE_NAME}")
    dflash_head = onnx.NodeProto()
    dflash_head.CopyFrom(head_node)
    dflash_head.name = "/lm_head/DFlashMatMul_Q4"
    dflash_head.input[0] = spec.final_hidden_output #graph surgery to add the final hidden state output to the lm_head input
    dflash_head.output[0] = DFLASH_LOGITS #graph surgery to add the full block size logits output to the lm_head output
    graph.node.append(dflash_head)
    graph.output.append(
        helper.make_tensor_value_info(
            DFLASH_LOGITS,
            TensorProto.FLOAT16,
            ["batch_size", "sequence_length", spec.vocab_size],
        )
    )

    destination = output / "text.onnx"
    onnx.save_model(model, str(destination))

    check = onnx.load(str(destination), load_external_data=False)
    check_outputs = {value.name for value in check.graph.output}
    missing = [name for name in required if name not in check_outputs]
    if missing:
        raise RuntimeError(f"Saved graph lost requested outputs: {missing}")
    return tap_names


def _build_lm_head(source: Path, output: Path, spec: TargetModelSpec) -> None:
    model = onnx.load(str(source / "text.onnx"), load_external_data=False)
    head_node = next((node for node in model.graph.node if node.name == HEAD_NODE_NAME), None)
    if head_node is None:
        raise RuntimeError(f"Could not find {HEAD_NODE_NAME}")

    initializers_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    missing = [name for name in HEAD_INITIALIZERS if name not in initializers_by_name]
    if missing:
        raise RuntimeError(f"LM-head initializers are missing: {missing}")

    node = onnx.NodeProto()
    node.CopyFrom(head_node)
    del node.input[:]
    node.input.extend(("hidden_states", *HEAD_INITIALIZERS))
    del node.output[:]
    node.output.append("logits")

    initializers = []
    for name in HEAD_INITIALIZERS:
        tensor = onnx.TensorProto()
        tensor.CopyFrom(initializers_by_name[name])
        initializers.append(tensor)

    graph = helper.make_graph(
        [node],
        "qwen36_quantized_lm_head",
        [helper.make_tensor_value_info(
            "hidden_states", TensorProto.FLOAT16, ["batch_size", "sequence_length", spec.hidden_size]
        )],
        [helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT16, ["batch_size", "sequence_length", spec.vocab_size]
        )],
        initializer=initializers,
    )
    head_model = helper.make_model(
        graph,
        producer_name="dflash-hipep-study",
        opset_imports=[
            helper.make_opsetid(opset.domain, opset.version)
            for opset in model.opset_import
        ],
        ir_version=model.ir_version,
    )
    destination = output / "lm_head.onnx"
    onnx.save_model(head_model, str(destination))

    check = onnx.load(str(destination), load_external_data=False)
    if [item.name for item in check.graph.initializer] != list(HEAD_INITIALIZERS):
        raise RuntimeError("Saved LM-head graph has unexpected initializers")
    for tensor in check.graph.initializer:
        location = {item.key: item.value for item in tensor.external_data}.get("location")
        if location != "text.onnx.data":
            raise RuntimeError(
                f"{tensor.name} should reference text.onnx.data, but references {location!r}"
            )


def _materialize_assets(source: Path, output: Path) -> tuple[list[str], list[str]]:
    copied = []
    for name in COPIED_ASSETS:
        source_path = source / name
        shutil.copy2(source_path, output / name)
        copied.append(name)

    linked = []
    for name in LINKED_ASSETS:
        source_path = source / name
        if not source_path.exists():
            continue
        destination = output / name
        destination.symlink_to(source_path.resolve())
        linked.append(name)
    return copied, linked


def build(
    source: Path,
    output: Path,
    force: bool,
    target_layer_ids: tuple[int, ...] | None = None,
    final_hidden_output: str | None = None,
) -> Path:
    source, output = _assert_safe_paths(source, output)
    spec = _read_source_spec(
        source,
        target_layer_ids or next(iter(TARGET_LAYER_IDS_BY_DRAFT.values())),
        final_hidden_output,
    )
    before = _source_snapshot(source)
    _replace_output_dir(output, force)

    try:
        # Create the derived text.onnx: expose the eight DFlash hidden-state
        # taps, final normalized hidden state, and full-block target logits.

        # 1. Expose the eight target hidden-state taps required by the DFlash draft:
        # target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

        # 2. The original OGA graph exposes logits for only one position (the
        # next-token prediction). DFlash needs one target prediction per block
        # position so Python can verify all proposed draft tokens:
        # output = target(
        #     block_output_ids,
        #     position_ids=block_position_ids,
        #     past_key_values=past_key_values_target,
        #     use_cache=True,
        #     output_hidden_states=block_size > 1,
        # )

        # 3. Expose the final normalized target hidden state so the integration
        # test can verify that standalone lm_head.onnx matches the copied head
        # in text.onnx. The production DFlash loop does not consume this output.
        tap_names = _build_text_graph(source, output, spec)


        # Create the separate lm_head.onnx used to project draft hidden states
        # with the target model's original quantized LM-head weights.
        _build_lm_head(source, output, spec)
        # Copy the required ONNX graphs/external weights and symlink the
        # unchanged tokenizer and configuration files from the source model.
        copied, linked = _materialize_assets(source, output)
        # Re-hash the source decoder graph and weights to prove the build did
        # not modify the original model.
        after = _source_snapshot(source)
        if before != after:
            raise RuntimeError("Source model changed while derived graphs were being built")

        manifest = {
            "source": str(source),
            "source_snapshot": before,
            "hidden_size": spec.hidden_size,
            "vocab_size": spec.vocab_size,
            "num_hidden_layers": spec.num_hidden_layers,
            "target_layer_ids": list(spec.target_layer_ids),
            "hidden_state_outputs": tap_names,
            "final_hidden_output": spec.final_hidden_output,
            "full_sequence_logits_output": DFLASH_LOGITS,
            "copied_assets": copied,
            "linked_assets": linked,
            "notes": {
                "source_model_modified": False,
                "lm_head_weights": "Referenced by external-data offsets in the copied text.onnx.data",
                "external_data_copied": "ORT rejects symlinks that resolve outside the model directory",
            },
        }
        (output / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    print(f"Built derived target: {output}")
    print(f"Source unchanged: {source}")
    print(f"Hidden-state taps: {len(tap_names)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-layer-ids",
        type=_parse_target_layer_ids,
        default=None,
        help=(
            "Comma-separated target layer IDs consumed by the DFlash draft. "
            "Overrides --target-layer-spec when both are provided."
        ),
    )
    parser.add_argument(
        "--target-layer-spec",
        choices=tuple(TARGET_LAYER_IDS_BY_DRAFT),
        default=next(iter(TARGET_LAYER_IDS_BY_DRAFT)),
        help=(
            "Known DFlash draft config whose target_layer_ids should be used."
        ),
    )
    parser.add_argument(
        "--final-hidden-output",
        default=None,
        help=(
            "Override the final hidden tensor name. Defaults to "
            "/model/layers.<num_hidden_layers>/final_norm_layernorm/output_0."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target_layer_ids = (
        args.target_layer_ids
        or TARGET_LAYER_IDS_BY_DRAFT[args.target_layer_spec]
    )
    build(
        args.source,
        args.output,
        args.force,
        target_layer_ids,
        args.final_hidden_output,
    )


if __name__ == "__main__":
    main()
