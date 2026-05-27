#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ORT CPU reference backend (implicit -- not user-selectable)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .backend import Backend


class OrtCpuBackend(Backend):
    """Runs ONNX models with the ORT ``CPUExecutionProvider``.

    Used internally by :class:`~framework.model_runner.ModelRunner` as the
    reference backend. Not intended to be selected via ``--backend``.
    """

    @property
    def name(self) -> str:
        return "CPUExecutionProvider"

    def run(
        self,
        model_path: str,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        input_dict = {sess.get_inputs()[i].name: inp for i, inp in enumerate(inputs)}
        outputs = sess.run(None, input_dict)
        del sess
        return outputs
