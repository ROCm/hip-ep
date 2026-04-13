#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Module matching strategies for automatic model adaptation.

Each strategy knows how to:
  1. Match a specific nn.Module pattern (MLP, attention, etc.)
  2. Create a standalone proxy for torch.export
  3. Extract weight tensors for the compiled DLL
  4. Create a DLL-backed wrapper module
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class MatchResult:
    """Result of matching a strategy to a module."""

    strategy_name: str
    module_path: str
    hidden_size: int
    weight_names: List[str]


class Strategy(ABC):
    """Base class for module matching strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def matches(self, module: nn.Module) -> bool:
        """Check if this strategy can handle the given module."""
        ...

    @abstractmethod
    def create_proxy(self, module: nn.Module) -> nn.Module:
        """Create a standalone proxy module for torch.export."""
        ...

    @abstractmethod
    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        """Extract weight tensors (transposed for linear) for DLL input."""
        ...

    @abstractmethod
    def get_hidden_size(self, module: nn.Module) -> int:
        """Get the hidden dimension for shape computation."""
        ...

    def get_input_rank(self, module: nn.Module) -> int:
        """Get expected input tensor rank (3 for [B,S,H], 2 for [B*S,H])."""
        return 3  # default: [batch, seq, hidden]


class _MlpProxyModule(nn.Module):
    """Standalone MLP: gate_proj + act_fn + up_proj + mul + down_proj."""

    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _FfnProxyModule(nn.Module):
    """Standalone FFN: fc1 + act + fc2."""

    def __init__(self, fc1, fc2, act_fn):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.act_fn = act_fn

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# Concrete Strategies
# ──────────────────────────────────────────────────────────────────────


class MLPStrategy(Strategy):
    """Matches SwiGLU MLP blocks (Llama, Qwen, Mistral, Gemma, etc.)

    Pattern: gate_proj + up_proj + down_proj + act_fn (SiLU/GELU)
    """

    @property
    def name(self) -> str:
        return "mlp"

    def matches(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and hasattr(module, "act_fn")
        )

    def create_proxy(self, module: nn.Module) -> nn.Module:
        return (
            _MlpProxyModule(
                copy.deepcopy(module.gate_proj).cpu(),
                copy.deepcopy(module.up_proj).cpu(),
                copy.deepcopy(module.down_proj).cpu(),
                module.act_fn,
            )
            .eval()
            .half()
        )

    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        return [
            module.gate_proj.weight.data.t().contiguous().cpu(),
            module.up_proj.weight.data.t().contiguous().cpu(),
            module.down_proj.weight.data.t().contiguous().cpu(),
        ]

    def get_hidden_size(self, module: nn.Module) -> int:
        return module.gate_proj.in_features


class SharedExpertStrategy(Strategy):
    """Matches MoE shared expert (Qwen3.5, DeepSeek, etc.)

    Pattern: module.shared_expert with gate_proj + up_proj + down_proj
    Delegates to MLPStrategy for the actual shared_expert submodule.
    """

    def __init__(self):
        self._mlp = MLPStrategy()

    @property
    def name(self) -> str:
        return "shared_expert"

    def matches(self, module: nn.Module) -> bool:
        return hasattr(module, "shared_expert") and self._mlp.matches(
            module.shared_expert
        )

    def create_proxy(self, module: nn.Module) -> nn.Module:
        return self._mlp.create_proxy(module.shared_expert)

    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        return self._mlp.get_weights(module.shared_expert)

    def get_hidden_size(self, module: nn.Module) -> int:
        return self._mlp.get_hidden_size(module.shared_expert)

    def get_input_rank(self, module: nn.Module) -> int:
        return 2  # MoE reshapes to [batch*seq, hidden]


class FFNStrategy(Strategy):
    """Matches fc1/fc2 FFN blocks (BERT, GPT-2, T5, etc.)

    Pattern: fc1 + fc2 + activation
    """

    @property
    def name(self) -> str:
        return "ffn"

    def matches(self, module: nn.Module) -> bool:
        return (
            hasattr(module, "fc1")
            and hasattr(module, "fc2")
            and not hasattr(module, "gate_proj")
        )

    def create_proxy(self, module: nn.Module) -> nn.Module:
        act_fn = getattr(module, "act_fn", getattr(module, "activation_fn", nn.GELU()))
        return (
            _FfnProxyModule(
                copy.deepcopy(module.fc1).cpu(),
                copy.deepcopy(module.fc2).cpu(),
                act_fn,
            )
            .eval()
            .half()
        )

    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        return [
            module.fc1.weight.data.t().contiguous().cpu(),
            module.fc2.weight.data.t().contiguous().cpu(),
        ]

    def get_hidden_size(self, module: nn.Module) -> int:
        return module.fc1.in_features


# ──────────────────────────────────────────────────────────────────────
# Default Strategy List
# ──────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGIES: List[Strategy] = [
    SharedExpertStrategy(),  # must be before MLPStrategy (more specific)
    MLPStrategy(),
    FFNStrategy(),
]
