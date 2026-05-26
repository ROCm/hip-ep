#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Orchestrate backend inference and reference comparison.

The ``ModelRunner`` is backend-agnostic: it delegates actual inference
to a pluggable :class:`~framework.backend.Backend` and resolves the
reference outputs from one of two sources -- a live ORT CPU run or a
per-test cache (default; backed by ORT CPU on miss). Sessions are
created sequentially (never at the same time) so peak memory stays low
when models carry large weight initializers.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np
import onnx

from .backend import Backend
from .comparator import print_output_summary, sizeof_fmt, tensor_desc
from .ort_cpu_backend import OrtCpuBackend
from .reference_cache import ReferenceCache, _contig_preserve_rank, sanitize_name

_TAG = "[ModelRunner]"

# Type aliases for the run_sample inputs/model parameters. Kept as plain
# unions rather than TypeAlias so they show up correctly in any Python
# >= 3.9.
ModelLike = Union[bytes, "onnx.ModelProto", Path, str]
InputLike = Union[np.ndarray, Path, str]
ReferenceMode = str  # "cpu" | "cache"


class ModelRunner:
    """Orchestrates backend inference and reference comparison.

    Memory strategy: only one inference session exists at a time.
      1. Persist model bytes to disk, release the Python buffer.
      2. Run the test backend, destroy its session.
      3. If reference="cpu" (or "cache" miss), run the CPU reference,
         destroy its session.
      4. Snapshot inputs + both output tensors into the work dir for
         postmortem on failure (deleted by the conftest fixture on
         pass; see ``_record_artifacts``).
    """

    def __init__(
        self,
        backend: Backend,
        work_dir: str | Path,
        cache: ReferenceCache,
        show_output: bool = False,
    ):
        self.backend = backend
        self._ref_backend = OrtCpuBackend()
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self._counter = 0
        self._current_test_name = ""
        self._last_subdir: Path | None = None
        self.show_output = show_output

        print(f"{_TAG} ======== ModelRunner initialising ========")
        print(f"{_TAG}   Backend   : {self.backend.name}")
        print(f"{_TAG}   Reference : {self._ref_backend.name}")
        print(f"{_TAG}   Work dir  : {self.work_dir}")
        print(f"{_TAG}   Cache dir : {self.cache.root}")
        print(
            f"{_TAG}   Cache mode: "
            f"{'OFF' if self.cache.no_cache else ('REFRESH' if self.cache.refresh else 'ON')}"
        )
        print(f"{_TAG} =========================================")

    # ------------------------------------------------------------------
    # Per-test plumbing
    # ------------------------------------------------------------------

    def set_test_name(self, name: str):
        """Set the current pytest node id (called by the per-test fixture)."""
        self._current_test_name = sanitize_name(name) if name else ""

    def _next_subdir(self, name: str = "") -> Path:
        self._counter += 1
        label = f"{self._counter:03d}"
        tag = name or self._current_test_name
        if tag:
            label += f"_{tag}"
        d = self.work_dir / label
        d.mkdir(parents=True, exist_ok=True)
        self._last_subdir = d
        return d

    def cleanup_last(self):
        """Remove the last test's work directory to reclaim disk space."""
        if self._last_subdir and self._last_subdir.exists():
            shutil.rmtree(self._last_subdir, ignore_errors=True)
            self._last_subdir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sample(
        self,
        model: ModelLike,
        inputs: Sequence[InputLike],
        *,
        name: str | None = None,
        reference: ReferenceMode = "cache",
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run *model* on the configured backend and return (actual, expected).

        Parameters
        ----------
        model:
            One of: serialised ONNX bytes, an ``onnx.ModelProto``, or a
            ``Path``/``str`` pointing at a ``.onnx`` file on disk.
        inputs:
            Sequence of input tensors, each either a numpy array or a
            ``Path``/``str`` pointing at a ``.npy`` file on disk.
        name:
            Optional override for the cache directory name and the
            work-directory tag. Defaults to the current pytest node id.
            Pass an explicit name when sharing one sample across multiple
            tests so they hit the same cache entry instead of fighting
            over the same directory.
        reference:
            * ``"cpu"`` -- always run ORT CPU as the reference.
            * ``"cache"`` (default) -- read from the on-disk cache; on
              miss, run CPU and write it back.
        """
        if reference not in ("cpu", "cache"):
            raise ValueError(f"reference must be one of cpu|cache, got {reference!r}")

        sample_name = name or self._current_test_name or "unnamed"
        sub = self._next_subdir(sample_name)
        model_path = sub / "model.onnx"

        print(f"{_TAG} -------- {sample_name} --------")

        model_bytes = _resolve_model_to_bytes(model)
        model_path.write_bytes(model_bytes)
        print(f"{_TAG} Model : {model_path}")
        print(f"{_TAG} Size  : {sizeof_fmt(len(model_bytes))}")

        np_inputs = _resolve_inputs(inputs)
        print(f"{_TAG} Inputs: {len(np_inputs)}")
        for i, arr in enumerate(np_inputs):
            print(f"{_TAG}   [{i}] {tensor_desc(arr)}")

        # 1. Backend inference --------------------------------------------
        print(f"{_TAG} Running {self.backend.name} ...")
        t0 = time.perf_counter()
        actual = self.backend.run(str(model_path), np_inputs)
        backend_elapsed = time.perf_counter() - t0
        print(f"{_TAG}   Done ({backend_elapsed:.3f}s), {len(actual)} output(s)")

        # 2. Reference resolution ----------------------------------------
        ref_elapsed = 0.0
        ref_source = reference
        if reference == "cache":
            cached = self.cache.load(sample_name, model_bytes, np_inputs)
            if cached is not None:
                expected = cached
                ref_source = "cache(hit)"
            else:
                expected, ref_elapsed = self._run_cpu_reference(
                    str(model_path), np_inputs
                )
                self.cache.store(sample_name, model_bytes, np_inputs, expected)
                ref_source = "cache(miss->cpu)"
        else:
            expected, ref_elapsed = self._run_cpu_reference(str(model_path), np_inputs)

        # 3. Snapshot per-test artefacts ---------------------------------
        # Always write inputs + both output tensors next to model.onnx in
        # the per-test work dir. On pass the whole subdir is deleted by
        # the conftest fixture, so this costs nothing in the success
        # case. On failure (or with --keep-artifacts) it gives a
        # self-contained postmortem snapshot in a single directory.
        self._record_artifacts(np_inputs, actual, expected)

        # 4. Summary ------------------------------------------------------
        print_output_summary(actual, expected, tag=_TAG)
        print(
            f"{_TAG} Timing: backend={backend_elapsed:.3f}s  "
            f"ref={ref_elapsed:.3f}s ({ref_source})  "
            f"total={backend_elapsed + ref_elapsed:.3f}s"
        )

        return actual, expected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_cpu_reference(
        self,
        model_path: str,
        inputs: list[np.ndarray],
    ) -> tuple[list[np.ndarray], float]:
        print(f"{_TAG} Running {self._ref_backend.name} ...")
        t0 = time.perf_counter()
        outputs = self._ref_backend.run(model_path, inputs)
        elapsed = time.perf_counter() - t0
        print(f"{_TAG}   Done ({elapsed:.3f}s), {len(outputs)} output(s)")
        return outputs, elapsed

    def _record_artifacts(
        self,
        inputs: list[np.ndarray],
        actual: list[np.ndarray],
        expected: list[np.ndarray],
    ) -> None:
        """Persist the per-test tensors next to ``model.onnx``.

        Writes three subdirectories into ``self._last_subdir``:
          * ``inputs/in_<i>.npy``           -- exactly what the backend received
          * ``outputs_actual/out_<i>.npy``  -- what the backend produced
          * ``outputs_expected/out_<i>.npy`` -- reference (cache / cpu)

        Called unconditionally at the end of ``run_sample``. Passing
        tests get cleaned up by the conftest fixture so this is free;
        failing tests retain the full snapshot for postmortem.
        """
        if self._last_subdir is None:
            return
        subdir = self._last_subdir
        sub_in = subdir / "inputs"
        sub_act = subdir / "outputs_actual"
        sub_exp = subdir / "outputs_expected"
        sub_in.mkdir(exist_ok=True)
        sub_act.mkdir(exist_ok=True)
        sub_exp.mkdir(exist_ok=True)
        for i, arr in enumerate(inputs):
            np.save(sub_in / f"in_{i}.npy", _contig_preserve_rank(arr))
        for i, arr in enumerate(actual):
            np.save(sub_act / f"out_{i}.npy", _contig_preserve_rank(arr))
        for i, arr in enumerate(expected):
            np.save(sub_exp / f"out_{i}.npy", _contig_preserve_rank(arr))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_to_bytes(model: ModelLike) -> bytes:
    """Normalise *model* to a ``bytes`` payload regardless of input form."""
    if isinstance(model, (bytes, bytearray)):
        return bytes(model)
    if isinstance(model, onnx.ModelProto):
        return model.SerializeToString()
    if isinstance(model, (str, Path)):
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"model path does not exist: {path}")
        return path.read_bytes()
    raise TypeError(
        f"model must be bytes | onnx.ModelProto | Path | str, got {type(model)}"
    )


def _resolve_inputs(inputs: Iterable[InputLike]) -> list[np.ndarray]:
    """Normalise *inputs* to a list of numpy arrays.

    Path / str entries are loaded as ``.npy`` files. Numpy arrays are
    forwarded by reference. The downstream consumers (ORT ``Run``, the
    cache serialiser, the artifact snapshotter) all treat their input
    as read-only -- ORT does not mutate session inputs and the
    serialisers always go through ``np.save(_contig_preserve_rank(...))``
    -- so no defensive copy is needed.
    """
    result: list[np.ndarray] = []
    for i, item in enumerate(inputs):
        if isinstance(item, np.ndarray):
            result.append(item)
        elif isinstance(item, (str, Path)):
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"input[{i}] path does not exist: {path}")
            result.append(np.load(path))
        else:
            raise TypeError(
                f"input[{i}] must be ndarray | Path | str, got {type(item)}"
            )
    return result
