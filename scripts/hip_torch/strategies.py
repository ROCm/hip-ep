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


class NormMlpResidualStrategy(Strategy):
    """Matches RMSNorm + MLP + Residual add as a single compiled unit.

    Covers: rms_norm + gate_proj + silu + up_proj + mul + down_proj + add
    This is the full post-attention compute block in Llama/Qwen/Mistral.
    """

    @property
    def name(self) -> str:
        return "norm_mlp_residual"

    def matches(self, module: nn.Module) -> bool:
        # This strategy matches at the decoder layer level
        return (
            hasattr(module, "post_attention_layernorm")
            and hasattr(module, "mlp")
            and hasattr(module.mlp, "gate_proj")
        )

    def create_proxy(self, module: nn.Module) -> nn.Module:
        return (
            _NormMlpResidualProxy(
                copy.deepcopy(module.post_attention_layernorm).cpu(),
                copy.deepcopy(module.mlp).cpu(),
            )
            .eval()
            .half()
        )

    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        norm = module.post_attention_layernorm
        mlp = module.mlp
        return [
            norm.weight.data.cpu().contiguous(),
            mlp.gate_proj.weight.data.t().contiguous().cpu(),
            mlp.up_proj.weight.data.t().contiguous().cpu(),
            mlp.down_proj.weight.data.t().contiguous().cpu(),
        ]

    def get_hidden_size(self, module: nn.Module) -> int:
        return module.mlp.gate_proj.in_features


class _NormMlpResidualProxy(nn.Module):
    """RMSNorm + MLP + residual add."""

    def __init__(self, norm, mlp):
        super().__init__()
        self.norm_weight = nn.Parameter(norm.weight.data)
        self.eps = norm.variance_epsilon
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, x):
        normed = torch.nn.functional.rms_norm(
            x, (x.shape[-1],), self.norm_weight, self.eps
        )
        mlp_out = self.down_proj(
            self.act_fn(self.gate_proj(normed)) * self.up_proj(normed)
        )
        return x + mlp_out


class LinearProjectionStrategy(Strategy):
    """Matches a single nn.Linear for offloading (Q/K/V/O projections)."""

    def __init__(self, attr_name: str, label: str = ""):
        self._attr = attr_name
        self._label = label or attr_name

    @property
    def name(self) -> str:
        return self._label

    def matches(self, module: nn.Module) -> bool:
        sub = getattr(module, self._attr, None)
        return sub is not None and isinstance(sub, nn.Linear)

    def create_proxy(self, module: nn.Module) -> nn.Module:
        return (
            _LinearProxy(copy.deepcopy(getattr(module, self._attr)).cpu()).eval().half()
        )

    def get_weights(self, module: nn.Module) -> List[torch.Tensor]:
        linear = getattr(module, self._attr)
        return [linear.weight.data.t().contiguous().cpu()]

    def get_hidden_size(self, module: nn.Module) -> int:
        return getattr(module, self._attr).in_features


class _LinearProxy(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)


# ──────────────────────────────────────────────────────────────────────
# Default Strategy List
# ──────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGIES: List[Strategy] = [
    SharedExpertStrategy(),  # must be before MLPStrategy (more specific)
    MLPStrategy(),
    FFNStrategy(),
]

# Extended strategies for maximum offload
MAX_OFFLOAD_STRATEGIES: List[Strategy] = [
    SharedExpertStrategy(),
    NormMlpResidualStrategy(),
    MLPStrategy(),
    FFNStrategy(),
]
