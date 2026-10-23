"""DFlash target adapter for the patched upstream OGA runtime.

This standalone module requires an OGA build with:

* AMDGPU continuous decoding enabled; and
* MultiModalPipelineState snapshot/rewind forwarding to its inner decoder.

At every verification boundary, OGA snapshots its managed KV, convolution, and
recurrent state. A partial rejection rewinds to that boundary and appends only
the accepted part of the candidate block.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import numpy as np


DEFAULT_MODEL_DIR = Path(__file__).parent / "models/Qwen3.6-35B-A3B-hidden"
DEFAULT_HIPEP_ROOT = Path("/proj/xcohdstaff7/devledas/hipep_0.5.1_linux")
DEFAULT_ROCM_ROOT = Path("/opt/rocm/core-7.11")
DFLASH_LOGITS = "dflash_logits"
EP_NAME = "MorphiZenEP"


@contextmanager
def _profile_stage(
    sink: Any | None,
    name: str,
    **metadata: Any,
) -> Iterator[dict[str, Any]]:
    """Record one optional wall-time region without coupling to a profiler."""
    dynamic: dict[str, Any] = {}
    if sink is None:
        yield dynamic
        return

    start_ns = time.perf_counter_ns()
    trace_factory = getattr(sink, "trace_context", None)
    trace_context = (
        trace_factory(name, metadata)
        if trace_factory is not None
        else nullcontext()
    )
    try:
        with trace_context:
            yield dynamic
    finally:
        sink.record_stage(
            name,
            time.perf_counter_ns() - start_ns,
            {**metadata, **dynamic},
        )


def _profile_call(sink: Any | None, method: str, **metadata: Any) -> None:
    if sink is not None:
        callback = getattr(sink, method, None)
        if callback is not None:
            callback(**metadata)


def _prepend(env: dict[str, str], name: str, *paths: Path) -> None:
    values = [str(path) for path in paths]
    if current := env.get(name):
        values.append(current)
    env[name] = os.pathsep.join(values)


def ensure_runtime_environment(
    hipep_root: Path = DEFAULT_HIPEP_ROOT,
    rocm_root: Path = DEFAULT_ROCM_ROOT,
) -> None:
    """Re-exec once so OGA, HIP EP, and ROCm resolve from one process."""
    if os.environ.get("_DFLASH_OGA_HIPEP_BOOTSTRAPPED") == "1":
        return

    env = os.environ.copy()
    env["_DFLASH_OGA_HIPEP_BOOTSTRAPPED"] = "1"
    env["HIPEP_ROOT"] = str(hipep_root)
    env["THEROCK_DIST"] = str(rocm_root)
    env["AMDGPU_EP_PATH"] = str(hipep_root / "lib/libamdgpu-ep.so")
    env["MORPHIZEN_EP_LIB"] = str(hipep_root / "lib/libhipgpu.so")
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    oga_package_lib = (
        Path(sys.prefix)
        / "lib"
        / python_tag
        / "site-packages"
        / "onnxruntime_genai"
    )
    _prepend(env, "PATH", hipep_root / "bin")
    _prepend(
        env,
        "LD_LIBRARY_PATH",
        oga_package_lib,
        hipep_root / "lib",
        rocm_root / "lib",
        rocm_root / "lib/rocm_sysdeps/lib",
        Path(sys.prefix) / "lib",
    )
    _prepend(env, "LIBRARY_PATH", hipep_root / "lib", rocm_root / "lib")
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


@dataclass(frozen=True)
class TargetModelSpec:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    hidden_output_names: tuple[str, ...]
    final_hidden_output: str

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "TargetModelSpec":
        with (model_dir / "genai_config.json").open(encoding="utf-8") as stream:
            genai_config = json.load(stream)
        model_config = genai_config["model"]
        decoder_config = model_config["decoder"]
        hidden_size = int(decoder_config["hidden_size"])
        vocab_size = int(model_config["vocab_size"])
        num_hidden_layers = int(decoder_config["num_hidden_layers"])

        with (model_dir / "build_manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        hidden_output_names = tuple(
            str(name) for name in manifest["hidden_state_outputs"]
        )
        final_hidden_output = str(manifest["final_hidden_output"])
        return cls(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_output_names=hidden_output_names,
            final_hidden_output=final_hidden_output,
        )


@dataclass(frozen=True)
class OgaPrefillOutput:
    logits: np.ndarray
    target_hidden: np.ndarray


@dataclass(frozen=True)
class OgaVerificationOutput:
    block_ids: np.ndarray
    logits: np.ndarray
    target_hidden: np.ndarray
    bonus_token: int
    processed_length_before: int


class OgaDFlashTarget:
    """Stateful snapshot/rewind target plus stateless HIP helper graphs."""

    def __init__(
        self,
        model_dir: Path,
        hipep_root: Path = DEFAULT_HIPEP_ROOT,
        profile_sink: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.hipep_root = Path(hipep_root).resolve()
        self.profile_sink = profile_sink
        self._validate_files()
        self.spec = TargetModelSpec.from_model_dir(self.model_dir)
        self.hidden_size = self.spec.hidden_size
        self.vocab_size = self.spec.vocab_size
        self.hidden_output_names = self.spec.hidden_output_names
        self.final_hidden_output = self.spec.final_hidden_output

        import onnxruntime as ort
        import onnxruntime_genai as og

        self._og = og
        config = og.Config(str(self.model_dir))
        config.overlay(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "session_options": {
                                "disable_cpu_ep_fallback": "1",
                            }
                        }
                    }
                }
            )
        )
        config.clear_providers()
        config.append_provider("AMDGPU")
        config.set_provider_option("AMDGPU", "profile", "hip")
        self.model = og.Model(config)
        if str(self.model.device_type).lower() == "cpu":
            raise RuntimeError("OGA reported CPU despite the AMDGPU override")

        ep_library = self.hipep_root / "lib/libhipgpu.so"
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == EP_NAME
        ]
        if not devices:
            ort.register_execution_provider_library(EP_NAME, str(ep_library))
            devices = [
                device for device in ort.get_ep_devices() if device.ep_name == EP_NAME
            ]
        if not devices:
            raise RuntimeError(f"No {EP_NAME} device was registered from {ep_library}")

        self.embedding_session = self._create_session(
            ort, devices, "embedding.onnx"
        )
        self.head_session = self._create_session(ort, devices, "lm_head.onnx")
        self.generator: Any | None = None
        self.max_length = 0
        self.processed_length = 0
        self.partial_rollback_count = 0
        if not hasattr(self._og.Generator, "snapshot_state"):
            raise RuntimeError(
                "The installed OGA runtime has no Generator.snapshot_state(); "
                "install the patched upstream OGA runtime"
            )
        self._expected_next_token: int | None = None
        self.rollback_correction_mismatch_count = 0

    def _validate_files(self) -> None:
        required = (
            "genai_config.json",
            "text.onnx",
            "text.onnx.data",
            "embedding.onnx",
            "embedding.onnx.data",
            "vision.onnx",
            "vision.onnx.data",
            "lm_head.onnx",
            "tokenizer.json",
            "build_manifest.json",
        )
        missing = [
            str(self.model_dir / name)
            for name in required
            if not (self.model_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Derived OGA target is incomplete:\n  " + "\n  ".join(missing)
            )

    def _create_session(
        self,
        ort: Any,
        devices: list[Any],
        filename: str,
    ) -> Any:
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(devices, {})
        session = ort.InferenceSession(
            str(self.model_dir / filename),
            sess_options=options,
        )
        if session.get_providers()[0] != EP_NAME:
            raise RuntimeError(
                f"{filename} did not select {EP_NAME}: {session.get_providers()}"
            )
        return session

    @property
    def providers(self) -> dict[str, Any]:
        return {
            "decoder": str(self.model.device_type),
            "embedding": self.embedding_session.get_providers(),
            "lm_head": self.head_session.get_providers(),
        }

    def _new_generator(self) -> Any:
        params = self._og.GeneratorParams(self.model)
        params.set_search_options(max_length=self.max_length, do_sample=False)
        return self._og.Generator(self.model, params)

    @staticmethod
    def _as_token_array(input_ids: np.ndarray) -> np.ndarray:
        tokens = np.asarray(input_ids, dtype=np.int32)
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.ndim != 1:
            raise ValueError(f"Expected one token sequence, got shape {tokens.shape}")
        return np.ascontiguousarray(tokens)

    def _hidden_outputs(self, profile_prefix: str) -> np.ndarray:
        if self.generator is None:
            raise RuntimeError("Target generator has not been initialized")
        sink = getattr(self, "profile_sink", None)
        values = []
        for index, name in enumerate(self.hidden_output_names):
            with _profile_stage(
                sink,
                f"{profile_prefix}.hidden_read",
                tap_index=index,
                output_name=name,
            ) as profile:
                value = np.asarray(self.generator.get_output(name))
                profile["output_bytes"] = int(value.nbytes)
            values.append(value)
        if not values or any(value.size == 0 for value in values):
            raise RuntimeError("OGA returned an empty DFlash hidden-state output")
        sequence_shapes = {value.shape[:-1] for value in values}
        if len(sequence_shapes) != 1 or any(
            value.shape[-1] != self.hidden_size for value in values
        ):
            raise RuntimeError(
                f"Unexpected hidden-state shapes: {[value.shape for value in values]}"
            )
        with _profile_stage(
            sink,
            f"{profile_prefix}.hidden_concat",
            input_bytes=sum(int(value.nbytes) for value in values),
        ) as profile:
            hidden = np.concatenate(values, axis=-1)
            profile["output_bytes"] = int(hidden.nbytes)
        return hidden

    def _full_logits(self, profile_prefix: str) -> np.ndarray:
        if self.generator is None:
            raise RuntimeError("Target generator has not been initialized")
        with _profile_stage(
            getattr(self, "profile_sink", None),
            f"{profile_prefix}.logits_read",
            output_name=DFLASH_LOGITS,
        ) as profile:
            logits = np.asarray(self.generator.get_output(DFLASH_LOGITS))
            profile["output_bytes"] = int(logits.nbytes)
        if (
            logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[-1] != self.vocab_size
        ):
            raise RuntimeError(f"Unexpected {DFLASH_LOGITS} shape: {logits.shape}")
        return logits

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        sink = getattr(self, "profile_sink", None)
        with _profile_stage(sink, "helper.embedding.input_conversion") as profile:
            ids = np.asarray(input_ids, dtype=np.int64)
            profile["output_bytes"] = int(ids.nbytes)
        if ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {ids.shape}"
            )
        with _profile_stage(
            sink,
            "helper.embedding.session",
            input_bytes=int(ids.nbytes),
            rows=int(ids.shape[1]),
        ) as profile:
            result = self.embedding_session.run(
                ["inputs_embeds"],
                {
                    "input_ids": ids,
                    "image_features": np.empty(
                        (0, self.hidden_size),
                        dtype=np.float16,
                    ),
                },
            )[0]
            profile["output_bytes"] = int(result.nbytes)
        return result

    def lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        sink = getattr(self, "profile_sink", None)
        with _profile_stage(sink, "helper.lm_head.input_conversion") as profile:
            hidden = np.asarray(hidden_states, dtype=np.float16)
            profile["output_bytes"] = int(hidden.nbytes)
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [batch, sequence, {self.hidden_size}], "
                f"got {hidden.shape}"
            )
        with _profile_stage(
            sink,
            "helper.lm_head.session",
            input_bytes=int(hidden.nbytes),
            rows=int(hidden.shape[1]),
        ) as profile:
            result = self.head_session.run(
                ["logits"],
                {"hidden_states": hidden},
            )[0]
            profile["output_bytes"] = int(result.nbytes)
        return result

    def prefill(
        self,
        input_ids: np.ndarray,
        *,
        max_new_tokens: int,
        block_size: int,
    ) -> OgaPrefillOutput:
        """Process the prompt while keeping OGA state and sequence length aligned."""
        sink = getattr(self, "profile_sink", None)
        with _profile_stage(sink, "target.prefill.total"):
            with _profile_stage(sink, "target.prefill.input_conversion") as profile:
                tokens = self._as_token_array(input_ids)
                profile["output_bytes"] = int(tokens.nbytes)
            self.max_length = (
                int(tokens.size) + int(max_new_tokens) + int(block_size) + 2
            )
            with _profile_stage(sink, "target.prefill.generator_create"):
                self.generator = self._new_generator()
            with _profile_stage(
                sink,
                "target.prefill.append",
                rows=int(tokens.size),
                input_bytes=int(tokens.nbytes),
            ):
                self.generator.append_tokens(tokens)

            with _profile_stage(sink, "target.prefill.logits_read") as profile:
                logits = np.asarray(self.generator.get_logits()).copy()
                profile["output_bytes"] = int(logits.nbytes)
            if logits.ndim == 2:
                logits = logits[:, None, :]
            if logits.ndim != 3 or logits.shape[:2] != (1, 1):
                raise RuntimeError(f"Unexpected prefill logits shape: {logits.shape}")

            self.processed_length = int(tokens.size)
            self.partial_rollback_count = 0
            self._expected_next_token = int(logits[0, 0].argmax())
            hidden = self._hidden_outputs("target.prefill")
            return OgaPrefillOutput(
                logits=logits,
                target_hidden=hidden,
            )

    def verify(self, block_ids: np.ndarray) -> OgaVerificationOutput:
        """Snapshot the clean boundary and score the complete candidate block."""
        if self.generator is None:
            raise RuntimeError("Call prefill() before verify()")
        sink = getattr(self, "profile_sink", None)
        with _profile_stage(sink, "target.verify.total"):
            with _profile_stage(sink, "target.verify.input_conversion") as profile:
                tokens = self._as_token_array(block_ids)
                profile["output_bytes"] = int(tokens.nbytes)
            if tokens.size == 0:
                raise ValueError("Verification block cannot be empty")
            if self._expected_next_token is None:
                raise RuntimeError("Target has no expected verification anchor")
            if int(tokens[0]) != self._expected_next_token:
                raise RuntimeError(
                    f"Verification anchor {int(tokens[0])} does not match target "
                    f"token {self._expected_next_token}"
                )

            processed_before = self.processed_length
            # The snapshot position equals the number of tokens already processed:
            # no OGA-sampled token is kept pending in this implementation.
            with _profile_stage(
                sink,
                "target.verify.snapshot",
                position=processed_before,
            ):
                self.generator.snapshot_state()
            with _profile_stage(
                sink,
                "target.verify.append",
                rows=int(tokens.size),
                input_bytes=int(tokens.nbytes),
            ):
                self.generator.append_tokens(tokens)

            logits = self._full_logits("target.verify")
            hidden = self._hidden_outputs("target.verify")
            expected_rows = int(tokens.size)
            if logits.shape[1] != expected_rows or hidden.shape[1] != expected_rows:
                raise RuntimeError(
                    f"Verification block length {expected_rows} returned "
                    f"logits={logits.shape}, hidden={hidden.shape}"
                )

            return OgaVerificationOutput(
                block_ids=tokens[None, :].astype(np.int64, copy=False),
                logits=logits,
                target_hidden=hidden,
                bonus_token=int(logits[0, -1].argmax()),
                processed_length_before=processed_before,
            )

    def commit_verified_prefix(
        self,
        verification: OgaVerificationOutput,
        *,
        committed_count: int,
        expected_bonus: int,
        committed_prefix_ids: np.ndarray,
    ) -> int:
        """Commit accepted rows, restoring the native snapshot when necessary."""
        if self.generator is None:
            raise RuntimeError("Call prefill() before committing verification")

        sink = getattr(self, "profile_sink", None)
        commit_start_ns = time.perf_counter_ns() if sink is not None else 0
        block_size = int(verification.block_ids.shape[1])
        if not 1 <= committed_count <= block_size:
            raise ValueError(
                f"committed_count must be in [1, {block_size}], got {committed_count}"
            )
        if expected_bonus != int(
            verification.logits[0, committed_count - 1].argmax()
        ):
            raise RuntimeError("Expected bonus is inconsistent with verification logits")

        committed_prefix = self._as_token_array(committed_prefix_ids)
        accepted_length = verification.processed_length_before + committed_count
        if int(committed_prefix.size) != accepted_length:
            raise ValueError(
                f"Committed prefix has {committed_prefix.size} tokens, "
                f"expected {accepted_length}"
            )

        accepted_block = np.ascontiguousarray(
            verification.block_ids[0, :committed_count],
            dtype=np.int32,
        )
        if not np.array_equal(
            committed_prefix[-committed_count:],
            accepted_block,
        ):
            raise ValueError(
                "Committed prefix does not end with the accepted verification rows"
            )

        if committed_count < block_size:
            # Restore the exact pre-verification target state, then process only
            # the anchor and draft rows that Python actually accepted.
            with _profile_stage(
                sink,
                "target.commit.rewind",
                position=verification.processed_length_before,
            ):
                self.generator.rewind_to(verification.processed_length_before)
            with _profile_stage(
                sink,
                "target.commit.replay_append",
                rows=committed_count,
                input_bytes=int(accepted_block.nbytes),
            ):
                self.generator.append_tokens(accepted_block)
            restored_logits = self._full_logits("target.commit.replay")
            restored_hidden = self._hidden_outputs("target.commit.replay")
            if (
                restored_logits.shape[1] != committed_count
                or restored_hidden.shape[1] != committed_count
            ):
                raise RuntimeError(
                    f"Accepted rollback length {committed_count} returned "
                    f"logits={restored_logits.shape}, hidden={restored_hidden.shape}"
                )

            restored_bonus = int(restored_logits[0, -1].argmax())
            if restored_bonus != expected_bonus:
                # Keep a summary diagnostic without interrupting or cluttering
                # the benchmark, matching upstream DFlash's non-enforcing path.
                self.rollback_correction_mismatch_count += 1

            # dflash_generate_oga carries these accepted hidden rows into the
            # next draft call. Replace them with rows from the state-consistent
            # restored append, matching upstream OGA's MTP fallback behavior.
            verification.target_hidden[:, :committed_count, :] = restored_hidden
            self.partial_rollback_count += 1
        elif verification.bonus_token != expected_bonus:
            raise RuntimeError(
                f"OGA bonus {verification.bonus_token} != expected {expected_bonus}"
            )

        self.processed_length = accepted_length
        self._expected_next_token = expected_bonus
        if sink is not None:
            sink.record_stage(
                "target.commit.total",
                time.perf_counter_ns() - commit_start_ns,
                {
                    "block_size": block_size,
                    "committed_count": committed_count,
                    "partial": committed_count < block_size,
                    "replay_rows": (
                        committed_count if committed_count < block_size else 0
                    ),
                },
            )
        return expected_bonus


def dflash_generate_oga(
    model: "DFlashDraftModel",
    target: OgaDFlashTarget,
    input_ids: "torch.LongTensor",
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    """Run the PyTorch DFlash controller with an OGA/HIPEP target."""
    if temperature >= 1e-5:
        raise NotImplementedError(
            "The OGA DFlash integration currently supports greedy decoding"
        )

    import torch
    from dflash.model import sample
    from transformers.cache_utils import DynamicCache

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = (
        model.mask_token_id if mask_token_id is None else mask_token_id
    )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The OGA DFlash integration supports batch size 1")

    draft_parameter = next(model.parameters())
    draft_device = draft_parameter.device
    draft_dtype = draft_parameter.dtype
    input_ids = input_ids.to(draft_device)
    profile_sink = getattr(target, "profile_sink", None)
    mismatch_generation_start = target.rollback_correction_mismatch_count
    _profile_call(
        profile_sink,
        "begin_generation",
        mode="adapter_baseline" if block_size == 1 else "dflash",
        block_size=int(block_size),
        input_tokens=int(num_input_tokens),
        max_new_tokens=int(max_new_tokens),
    )

    with _profile_stage(profile_sink, "controller.setup"):
        output_ids = torch.full(
            (1, max_length + block_size),
            mask_token_id,
            dtype=torch.long,
            device=draft_device,
        )
        position_ids = torch.arange(
            output_ids.shape[1],
            device=draft_device,
        ).unsqueeze(0)
        past_key_values_draft = DynamicCache()

    prefill_start = time.perf_counter() if return_stats else None
    with _profile_stage(
        profile_sink,
        "controller.prefill.input_to_numpy",
    ) as profile:
        prefill_ids = input_ids.detach().cpu().numpy().astype(
            np.int64,
            copy=False,
        )
        profile["output_bytes"] = int(prefill_ids.nbytes)
    output = target.prefill( #target prefill call
        prefill_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
    )
    with _profile_stage(profile_sink, "controller.prefill.output_commit"):
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            torch.from_numpy(output.logits).to(draft_device),
            temperature,
        )
    with _profile_stage(
        profile_sink,
        "controller.prefill.hidden_to_torch",
        input_bytes=int(output.target_hidden.nbytes),
    ):
        target_hidden = torch.from_numpy(output.target_hidden).to(
            device=draft_device,
            dtype=draft_dtype,
        )
    time_to_first_token = (
        time.perf_counter() - prefill_start if return_stats else None
    )

    decode_start = time.perf_counter() if return_stats else None
    _profile_call(profile_sink, "mark_decode_start")
    acceptance_lengths: list[int] = []
    start = num_input_tokens
    draft_prefill = True
    iteration = 0
    while start < max_length:
        _profile_call(
            profile_sink,
            "begin_iteration",
            index=iteration,
            context_length=int(start),
            block_size=int(block_size),
            draft_cache_length=int(past_key_values_draft.get_seq_length()),
        )
        mismatch_before = target.rollback_correction_mismatch_count
        with _profile_stage(profile_sink, "controller.block_clone"):
            block_output_ids = output_ids[:, start : start + block_size].clone()

        if block_size > 1:
            with _profile_stage(
                profile_sink,
                "controller.embedding_ids_to_numpy",
            ) as profile:
                embedding_ids = block_output_ids.detach().cpu().numpy()
                profile["output_bytes"] = int(embedding_ids.nbytes)
            embedding_output = target.embed(embedding_ids) #target embed call
            with _profile_stage(
                profile_sink,
                "controller.embedding_to_torch",
                input_bytes=int(embedding_output.nbytes),
            ):
                noise_embedding = torch.from_numpy(embedding_output).to(
                    device=draft_device,
                    dtype=draft_dtype,
                )
            with _profile_stage(profile_sink, "controller.draft_forward"): #draft forward call
                draft_hidden = model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :,
                        past_key_values_draft.get_seq_length() : start + block_size,
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, 1 - block_size :, :]

            with _profile_stage(
                profile_sink,
                "controller.draft_hidden_to_numpy",
            ) as profile:
                draft_hidden_numpy = (
                    draft_hidden.detach().to(torch.float16).cpu().numpy()
                )
                profile["output_bytes"] = int(draft_hidden_numpy.nbytes)
            draft_logits_numpy = target.lm_head(draft_hidden_numpy) #target lm head call
            with _profile_stage(
                profile_sink,
                "controller.draft_logits_to_torch",
                input_bytes=int(draft_logits_numpy.nbytes),
            ):
                draft_logits = torch.from_numpy(draft_logits_numpy).to(draft_device)

            with _profile_stage(profile_sink, "controller.draft_cache_crop"):
                past_key_values_draft.crop(start)
            with _profile_stage(profile_sink, "controller.draft_sample"):
                block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = time.perf_counter()

        with _profile_stage(
            profile_sink,
            "controller.verify_ids_to_numpy",
        ) as profile:
            verification_ids = block_output_ids.detach().cpu().numpy()
            profile["output_bytes"] = int(verification_ids.nbytes)
        output = target.verify(verification_ids) #target verify call - temporarily advance through entire candidate and get logits and hidden states
        with _profile_stage(
            profile_sink,
            "controller.posterior_argmax",
            input_bytes=int(output.logits.nbytes),
        ):
            posterior = sample(
                torch.from_numpy(output.logits).to(draft_device),
                temperature,
            )

        with _profile_stage(profile_sink, "controller.acceptance_compare"):
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
        committed_count = acceptance_length + 1
        with _profile_stage(profile_sink, "controller.output_commit"):
            output_ids[:, start : start + committed_count] = block_output_ids[
                :, :committed_count
            ]
            output_ids[:, start + committed_count] = posterior[
                :, acceptance_length
            ]
            start += committed_count

        with _profile_stage(
            profile_sink,
            "controller.committed_prefix_to_numpy",
        ) as profile:
            committed_prefix = output_ids[:, :start].detach().cpu().numpy()
            profile["output_bytes"] = int(committed_prefix.nbytes)
        target.commit_verified_prefix( #target commit verified prefix call (partial acceptance logic/ rewind and replay)
            output,
            committed_count=committed_count,
            expected_bonus=int(posterior[0, acceptance_length]),
            committed_prefix_ids=committed_prefix,
        )
        acceptance_lengths.append(committed_count)

        if block_size > 1:
            accepted_hidden = output.target_hidden[:, :committed_count, :]
            with _profile_stage(
                profile_sink,
                "controller.next_hidden_to_torch",
                input_bytes=int(accepted_hidden.nbytes),
            ):
                target_hidden = torch.from_numpy(accepted_hidden).to(
                    device=draft_device,
                    dtype=draft_dtype,
                )

        _profile_call(
            profile_sink,
            "end_iteration",
            accepted_drafts=int(acceptance_length),
            committed_count=int(committed_count),
            partial=bool(committed_count < block_size),
            replay_rows=(
                int(committed_count) if committed_count < block_size else 0
            ),
            correction_mismatch=bool(
                target.rollback_correction_mismatch_count > mismatch_before
            ),
        )
        iteration += 1

        with _profile_stage(profile_sink, "controller.stop_check"):
            should_stop = stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            )
        if should_stop:
            break

    with _profile_stage(profile_sink, "controller.output_finalize"):
        output_ids = output_ids[:, : min(start + 1, max_length)]
        if stop_token_ids is not None:
            stop_token_tensor = torch.tensor(
                stop_token_ids,
                device=draft_device,
            )
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:],
                stop_token_tensor,
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    _profile_call(
        profile_sink,
        "end_generation",
        output_tokens=int(num_output_tokens),
        iterations=int(iteration),
        partial_rollbacks=int(target.partial_rollback_count),
        correction_mismatches=int(target.rollback_correction_mismatch_count),
        generation_correction_mismatches=int(
            target.rollback_correction_mismatch_count - mismatch_generation_start
        ),
    )

    if not return_stats:
        return output_ids

    total_decode_time = time.perf_counter() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        partial_rollback_count=target.partial_rollback_count,
    )


__all__ = [
    "DEFAULT_HIPEP_ROOT",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_ROCM_ROOT",
    "DFLASH_LOGITS",
    "OgaDFlashTarget",
    "OgaPrefillOutput",
    "OgaVerificationOutput",
    "dflash_generate_oga",
    "ensure_runtime_environment",
]
