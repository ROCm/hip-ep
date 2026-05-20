#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Abstract base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Interface that every inference backend must implement.

    A backend takes a saved ONNX model path and a list of input tensors,
    runs inference, and returns the output tensors.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in logs (e.g. provider or tool name)."""
        ...

    @abstractmethod
    def run(
        self,
        model_path: str,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference on *model_path* with the given *inputs*.

        Parameters
        ----------
        model_path : str
            Path to a saved ``.onnx`` model file on disk.
        inputs : list[np.ndarray]
            Input tensors in graph-input order.

        Returns
        -------
        list[np.ndarray]
            Output tensors in graph-output order.
        """
        ...
