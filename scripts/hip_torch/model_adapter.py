#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Generic model adapter: automatically find and offload compilable submodules.

Works with ANY HuggingFace model by walking the nn.Module tree and
matching against pluggable strategies (MLP, MoE, FFN, etc.).

Usage:
    from hip_torch.model_adapter import ModelAdapter
    adapter = ModelAdapter(model)
    report = adapter.compile_and_replace(prompt_len=10)
    # Model's MLP/FFN blocks are now GPU DLL-backed
    output = model.generate(...)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .compiler import Compiler, CompilationError
from .dll_cache import DllCache
from .dll_runner import HipDllRunner
from .fx_emitter import fx_graph_to_mlir
from .strategies import DEFAULT_STRATEGIES, MAX_OFFLOAD_STRATEGIES, Strategy
from .strategies import LinearProjectionStrategy

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# DLL-Backed Module Wrapper
# ──────────────────────────────────────────────────────────────────────


class DllBackedModule(nn.Module):
    """Generic wrapper that dispatches to compiled DLLs based on shape.

    Falls back to original PyTorch module for unseen shapes.
    """

    _total_dll_calls = 0
    _total_fallback_calls = 0

    def __init__(
        self,
        original: nn.Module,
        runners: Dict[str, HipDllRunner],
        weights: List[torch.Tensor],
    ):
        super().__init__()
        self._original = original
        self._runners = runners
        self._weights = [w.cpu().contiguous() for w in weights]
        self._shape_to_runner: Dict[Tuple[int, ...], HipDllRunner] = {}
        for label, runner in runners.items():
            shape = tuple(runner.input_metas[-1]["shape"])
            self._shape_to_runner[shape] = runner

    def forward(self, x):
        runner = self._shape_to_runner.get(tuple(x.shape))
        if runner is None:
            DllBackedModule._total_fallback_calls += 1
            return self._original(x)

        DllBackedModule._total_dll_calls += 1
        device = x.device
        dtype = x.dtype
        x_f16 = x.detach().cpu().to(torch.float16).contiguous()
        out = runner(*self._weights, x_f16)[0]
        return out.to(dtype).to(device)

    @classmethod
    def reset_counters(cls):
        cls._total_dll_calls = 0
        cls._total_fallback_calls = 0


# ──────────────────────────────────────────────────────────────────────
# Adapter Report
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AdapterReport:
    """Report of model adaptation results."""

    model_name: str = ""
    total_layers: int = 0
    replaced_count: int = 0
    strategy_used: str = ""
    compile_time: float = 0.0
    dll_shapes: Dict[str, List[int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return self.replaced_count

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Strategy: {self.strategy_used}",
            f"Replaced: {self.replaced_count}/{self.total_layers} layers",
            f"Compile time: {self.compile_time:.1f}s",
        ]
        for label, shape in self.dll_shapes.items():
            lines.append(f"  {label} DLL: {shape}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Model Adapter
# ──────────────────────────────────────────────────────────────────────


class ModelAdapter:
    """Automatically find and offload compilable submodules to GPU DLLs."""

    def __init__(
        self,
        model: nn.Module,
        strategies: Optional[List[Strategy]] = None,
        compiler: Optional[Compiler] = None,
        cache: Optional[DllCache] = None,
        max_offload: bool = False,
    ):
        self.model = model
        if max_offload:
            self.strategies = MAX_OFFLOAD_STRATEGIES
        else:
            self.strategies = strategies or DEFAULT_STRATEGIES
        self.compiler = compiler or Compiler()
        self.cache = cache or DllCache()
        self._max_offload = max_offload
        self._attn_proj_strategies = [
            LinearProjectionStrategy("q_proj", "q_proj"),
            LinearProjectionStrategy("k_proj", "k_proj"),
            LinearProjectionStrategy("v_proj", "v_proj"),
            LinearProjectionStrategy("o_proj", "o_proj"),
        ]

    def find_targets(self) -> List[Tuple[str, nn.Module, Strategy]]:
        """Walk model tree and find submodules matching any strategy."""
        targets = []
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            return targets

        for i, layer in enumerate(layers):
            for strategy in self.strategies:
                # For MoE: check layer.mlp (SharedExpertStrategy matches MoE block)
                mlp = getattr(layer, "mlp", None)
                if mlp and strategy.matches(mlp):
                    targets.append((f"model.layers.{i}.mlp", mlp, strategy))
                    break
        return targets

    def compile_and_replace(
        self,
        prompt_len: int = 1,
        shapes: Optional[Dict[str, int]] = None,
    ) -> AdapterReport:
        """Find compilable submodules, compile to DLLs, and replace.

        Args:
            prompt_len: Prompt length for prefill DLL compilation
            shapes: Custom shape labels → seq_len mapping
                    Default: {"decode": 1, "prefill": prompt_len}
        """
        report = AdapterReport()
        t0 = time.perf_counter()

        # Find targets
        targets = self.find_targets()
        if not targets:
            report.errors.append("No compilable submodules found")
            return report

        # All targets should use the same strategy (homogeneous layers)
        _, first_module, strategy = targets[0]
        report.strategy_used = strategy.name
        report.total_layers = len(targets)

        # Determine shapes to compile
        if shapes is None:
            shapes = {"decode": 1, "prefill": prompt_len}

        # Compile DLLs for each shape
        hidden = strategy.get_hidden_size(first_module)
        input_rank = strategy.get_input_rank(first_module)
        runners: Dict[str, HipDllRunner] = {}

        for label, seq_len in shapes.items():
            runner = self._compile_for_shape(
                strategy, first_module, hidden, seq_len, input_rank, label
            )
            if runner:
                runners[label] = runner
                report.dll_shapes[label] = list(runner.input_metas[-1]["shape"])

        if not runners:
            report.errors.append("All compilations failed")
            report.compile_time = time.perf_counter() - t0
            return report

        # Replace submodules
        DllBackedModule.reset_counters()
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            path, module, strat = targets[i] if i < len(targets) else (None, None, None)
            if module is None:
                continue

            weights = strat.get_weights(module)

            if strat.name == "shared_expert":
                original = layer.mlp.shared_expert
                layer.mlp.shared_expert = DllBackedModule(original, runners, weights)
            else:
                original = layer.mlp
                layer.mlp = DllBackedModule(original, runners, weights)
            report.replaced_count += 1

        # If max_offload, also offload attention projections
        if self._max_offload and report.replaced_count > 0:
            self._offload_attention_projections(shapes, report)

        report.compile_time = time.perf_counter() - t0
        return report

    def _offload_attention_projections(self, shapes, report):
        """Offload Q/K/V/O linear projections to GPU DLLs."""
        layers = self.model.model.layers
        attn_module = getattr(
            layers[0], "self_attn", getattr(layers[0], "linear_attn", None)
        )
        if attn_module is None:
            return

        for proj_strategy in self._attn_proj_strategies:
            if not proj_strategy.matches(attn_module):
                continue

            hidden = proj_strategy.get_hidden_size(attn_module)
            input_rank = proj_strategy.get_input_rank(attn_module)
            runners = {}

            for label, seq_len in shapes.items():
                runner = self._compile_for_shape(
                    proj_strategy,
                    attn_module,
                    hidden,
                    seq_len,
                    input_rank,
                    f"{proj_strategy.name}_{label}",
                )
                if runner:
                    runners[label] = runner

            if not runners:
                report.errors.append(f"{proj_strategy.name} compilation failed")
                continue

            # Replace in all layers
            for layer in layers:
                attn = getattr(layer, "self_attn", getattr(layer, "linear_attn", None))
                if attn is None:
                    continue
                weights = proj_strategy.get_weights(attn)
                original = getattr(attn, proj_strategy._attr)
                setattr(
                    attn,
                    proj_strategy._attr,
                    DllBackedModule(original, runners, weights),
                )

            report.dll_shapes[proj_strategy.name] = list(
                runners[list(runners.keys())[0]].input_metas[-1]["shape"]
            )
            log.info(f"Offloaded {proj_strategy.name} in all layers")

    def _compile_for_shape(
        self,
        strategy: Strategy,
        module: nn.Module,
        hidden: int,
        seq_len: int,
        input_rank: int,
        label: str,
    ) -> Optional[HipDllRunner]:
        """Compile a proxy module for a specific input shape."""
        proxy = strategy.create_proxy(module)

        if input_rank == 2:
            example = torch.randn(seq_len, hidden, dtype=torch.float16)
        else:
            example = torch.randn(1, seq_len, hidden, dtype=torch.float16)

        log.info(f"[{label}] Compiling for shape {list(example.shape)}")

        # Export
        try:
            ep = torch.export.export(proxy, (example,))
        except Exception as e:
            log.warning(f"[{label}] torch.export failed: {e}")
            return None

        # Generate MLIR
        try:
            mlir = fx_graph_to_mlir(ep, decompose=False)
        except Exception as e:
            log.warning(f"[{label}] MLIR generation failed: {e}")
            return None

        # Check cache
        cache_key = self.cache.compute_key(mlir)
        cached_dll = self.cache.get(cache_key)
        if cached_dll:
            log.info(f"[{label}] Cache hit: {cache_key}")
            try:
                return HipDllRunner(str(cached_dll), work_dir=str(cached_dll.parent))
            except Exception as e:
                log.warning(f"[{label}] Cached DLL load failed: {e}")

        # Compile
        try:
            dll_path = self.compiler.compile(mlir)
        except CompilationError as e:
            log.warning(f"[{label}] Compilation failed: {e}")
            return None

        # Cache
        cached = self.cache.put(
            cache_key,
            dll_path,
            metadata={"label": label, "shape": list(example.shape)},
        )

        # Load
        try:
            import os

            os.environ.setdefault("THEROCK_DIST", str(self.compiler.therock))
            runner = HipDllRunner(str(cached), work_dir=str(cached.parent))
            log.info(f"[{label}] Compiled: {runner}")
            return runner
        except Exception as e:
            log.warning(f"[{label}] DLL load failed: {e}")
            return None

    def get_execution_stats(self) -> Dict:
        """Get execution statistics after running the model."""
        dll = DllBackedModule._total_dll_calls
        fb = DllBackedModule._total_fallback_calls
        total = dll + fb
        return {
            "dll_calls": dll,
            "fallback_calls": fb,
            "total_calls": total,
            "dll_ratio": dll / max(total, 1),
        }
