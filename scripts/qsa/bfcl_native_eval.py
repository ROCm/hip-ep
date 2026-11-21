#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Use pinned official BFCL v4 prompts/backends/checkers with native hip-ep."""
# Imports follow isolated registry/environment setup; no third-party code is edited.
# ruff: noqa: E402

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "_local/qsa/native-gqa/bfcl-v4"
SOURCE = ROOT / "_local/qsa/benchmarks/gorilla"
REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
os.environ["BFCL_PROJECT_ROOT"] = str(TASK)
sys.path.insert(0, str(SOURCE / "berkeley-function-call-leaderboard"))

# The official default registry eagerly imports dozens of commercial SDKs.
# Supply only our two local model entries. Prompting, parsing, tool backends
# and scoring still come from the unmodified, pinned official source.
registry = types.ModuleType("bfcl_eval.constants.model_config")
registry.MODEL_CONFIG_MAPPING = {}
sys.modules[registry.__name__] = registry

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.category_mapping import (
    SINGLE_TURN_CATEGORY,
    MULTI_TURN_CATEGORY,
)
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.utils import (
    load_dataset_entry,
    load_ground_truth_entry,
    make_json_serializable,
)
from overrides import override


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class NativeHandler(OSSHandler):
    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, queue=None, **kw
    ):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kw)
        # BFCL keys simulated tool histories by this field. Both variants use
        # the same HF model ID, so give their tool histories separate keys.
        self.model_name_underline_replaced = registry_name.replace("-", "_")
        self.queue = queue
        self.variant = "Changed" if registry_name.endswith("Changed") else "Original"
        self.saved_responses = {}
        self.replayed_sources = []

    @override
    def _format_prompt(self, messages, function):
        # OLMo's pinned base tokenizer has no chat template. Use a fixed plain
        # transcript, without injecting answers, demonstrations or grammars.
        parts = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            name = ": " + str(message["name"]) if message.get("name") else ""
            parts.append(message["role"].capitalize() + name + ":\n" + content)
        return "\n\n".join(parts) + "\n\nAssistant:\n"

    @override
    def _query_prompting(self, inference_data):
        prompt = self._format_prompt(
            inference_data["message"], inference_data["function"]
        )
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        key = self.variant, hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key in self.saved_responses:
            response, source = self.saved_responses[key]
            self.replayed_sources.append(source)
            inference_data["inference_input_log"]["saved_native_response"] = source
            # This API operation is a cache lookup; original generation time
            # and token IDs remain in the referenced native response file.
            return response, 0.0
        assert self.queue is not None
        identifier = uuid.uuid4().hex
        request = {
            "request_id": identifier,
            "variant": self.variant,
            "prompt": prompt,
            "max_new_tokens": 4096,
            "temperature": 0,
            "stop": ["\n\nUser:", "\n\nSystem:", "\n\nTool:"],
        }
        write_json(self.queue / (identifier + ".request.json"), request)
        path = self.queue / (identifier + ".response.json")
        started = time.monotonic()
        while not path.exists():
            if time.monotonic() - started > 3500:
                raise TimeoutError(
                    "Native generation did not complete within the stage limit"
                )
            time.sleep(0.1)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["status"] == "complete", result
        assert result["variant"] == self.variant
        assert (
            result["prompt_sha256"]
            == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        )
        return result, time.monotonic() - started

    @override
    def _parse_query_response_prompting(self, api_response) -> dict:
        return {
            "model_responses": api_response["text"],
            "input_token": api_response["prompt_tokens"],
            "output_token": api_response["completion_tokens"],
        }


for variant in ["Original", "Changed"]:
    name = "HipEP-OLMo3-" + variant
    registry.MODEL_CONFIG_MAPPING[name] = types.SimpleNamespace(
        model_name="allenai/Olmo-3-1025-7B",
        display_name=name,
        url="https://huggingface.co/allenai/Olmo-3-1025-7B",
        org="local experiment",
        license="Apache-2.0",
        model_handler=NativeHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )

from bfcl_eval.eval_checker.eval_runner import (
    _evaluate_single_ast_entry,
    _evaluate_single_multi_turn_entry,
    _evaluate_single_relevance_entry,
)


def handler(variant, queue=None):
    name = "HipEP-OLMo3-" + variant
    return NativeHandler("allenai/Olmo-3-1025-7B", 0, name, False, queue=queue)


def score(model, entry, result, truth, category):
    entry = copy.deepcopy(entry)
    common = (model, entry["id"], result)
    if category.startswith("multi_turn_"):
        return _evaluate_single_multi_turn_entry(
            *common, truth, entry, model.registry_name, category
        )
    if "relevance" in category:
        return _evaluate_single_relevance_entry(
            *common, entry, model.registry_name, category
        )
    language, output = (
        (Language.JAVA, ReturnFormat.JAVA)
        if category == "simple_java"
        else (Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT)
        if category == "simple_javascript"
        else (Language.PYTHON, ReturnFormat.PYTHON)
    )
    return _evaluate_single_ast_entry(
        *common,
        truth,
        entry,
        model.registry_name,
        category,
        language=language,
        return_format=output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prepare-pilot", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--replay-queue", type=Path, action="append", default=[])
    parser.add_argument("--queue", type=Path)
    parser.add_argument(
        "--category",
        choices=SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY,
        default="simple_python",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    assert actual == REVISION
    assert not subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "diff", "--name-only"],
        cwd=SOURCE,
        text=True,
    ).strip()
    TASK.mkdir(parents=True, exist_ok=True)
    if args.prepare_pilot:
        rng = random.Random(42)
        cases = []
        for category in SINGLE_TURN_CATEGORY + MULTI_TURN_CATEGORY:
            entries = load_dataset_entry(category)
            count = 8 if category in SINGLE_TURN_CATEGORY else 6
            # The first simple case is the public checker/transport control.
            eligible = range(1 if category == "simple_python" else 0, len(entries))
            for index in sorted(rng.sample(eligible, count)):
                cases.append(
                    {
                        "category": category,
                        "index": index,
                        "id": entries[index]["id"],
                        "pair_index": len(cases),
                    }
                )
        assert len(cases) == 128
        manifest = {
            "bfcl_revision": REVISION,
            "seed": 42,
            "cases": cases,
            "scope": "stratified component pilot; 104 single-turn and 24 multi-turn cases",
            "excluded": ["web_search", "memory", "format_sensitivity"],
            "official_overall_score": False,
        }
        target = TASK / "pilot-manifest.json"
        if target.exists():
            assert json.loads(target.read_text()) == manifest
        write_json(target, manifest)
        print("Prepared 128 paired cases across 17 BFCL v4 components", flush=True)
        return
    if args.self_test:
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING,
        )

        simulator_names = [
            n
            for n in CLASS_FILE_PATH_MAPPING
            if not n.startswith("MemoryAPI") and n != "WebSearchAPI"
        ]
        for name in simulator_names:
            assert getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[name]), name)
        entries = load_dataset_entry("simple_python")
        truth = load_ground_truth_entry("simple_python")[0]["ground_truth"]
        model = handler("Original")
        assert score(
            model,
            entries[0],
            "[calculate_triangle_area(base=10, height=5)]",
            truth,
            "simple_python",
        )["valid"]
        assert not score(
            model,
            entries[0],
            "[calculate_triangle_area(base=999, height=5)]",
            truth,
            "simple_python",
        )["valid"]
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        config = load_dataset_entry("multi_turn_base")[0]["initial_config"]
        instances = []
        for variant in ["Original", "Changed", "Original"]:
            _, state = execute_multi_turn_func_call(
                [],
                config,
                ["GorillaFileSystem"],
                handler(variant).model_name_underline_replaced,
                "adapter-isolation-control",
            )
            instances.append(state["GorillaFileSystem"])
        assert instances[0] is instances[2] and instances[0] is not instances[1]
        write_json(
            TASK / "checker-self-test.json",
            {
                "status": "passed",
                "bfcl_revision": actual,
                "simulator_imports": simulator_names,
            },
        )
        print("Official BFCL correct/wrong AST controls passed", flush=True)
        return
    assert args.queue is not None and args.count > 0 and args.start >= 0
    assert (
        json.loads((TASK / "checker-self-test.json").read_text())["status"] == "passed"
    )
    assert (
        len(
            json.loads((TASK / "checker-self-test.json").read_text())[
                "simulator_imports"
            ]
        )
        == 8
    )
    audit = json.loads((TASK.parent / "final-audit.json").read_text())
    protocol = {
        "benchmark": "BFCL v4 single-turn and multi-turn components",
        "bfcl_revision": REVISION,
        "model": "allenai/Olmo-3-1025-7B",
        "model_mode": "base-model prompting; no instruction tuning or chat template",
        "variants": ["Original", "Changed"],
        "batch": 1,
        "capacity": 65536,
        "max_new_tokens": 4096,
        "decoding": "greedy; EOS or fixed transcript role stops; no constrained decoding",
        "stops": ["\n\nUser:", "\n\nSystem:", "\n\nTool:"],
        "selector_sha256": audit["selector_sha256"],
        "binary_sha256": audit["binary_sha256"],
        "training_updates": 0,
        "official_overall_score": "unavailable until all scoring categories complete",
    }
    protocol_path = TASK / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    write_json(protocol_path, protocol)
    models = {v: handler(v, args.queue) for v in ["Original", "Changed"]}
    if args.replay_queue:
        from native_response_cache import load_responses

        saved = load_responses(TASK, args.replay_queue)
        for model in models.values():
            model.saved_responses = saved
    if args.pilot:
        manifest = json.loads((TASK / "pilot-manifest.json").read_text())
        assert manifest["bfcl_revision"] == REVISION
        cases = manifest["cases"]
    else:
        entries = load_dataset_entry(args.category)
        cases = [
            {
                "category": args.category,
                "index": i,
                "id": entries[i]["id"],
                "pair_index": i,
            }
            for i in range(args.start, min(args.start + args.count, len(entries)))
        ]
    cached = {}
    for case in cases:
        category, index = case["category"], case["index"]
        if category not in cached:
            entries = load_dataset_entry(category)
            # Match official evaluate_task: Java/JS prompting normalizes types,
            # while scoring must retain their original language-specific types.
            scoring_entries = load_dataset_entry(
                category, include_prereq=False, include_language_specific_hint=False
            )
            truth = [] if "relevance" in category else load_ground_truth_entry(category)
            folder = TASK / category
            folder.mkdir(exist_ok=True)
            path = folder / "rows.jsonl"
            previous = (
                [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                if path.exists()
                else []
            )
            done = {(r["variant"], r["id"]) for r in previous}
            cached[category] = entries, scoring_entries, truth, path, done
        entries, scoring_entries, truth, path, done = cached[category]
        entry = entries[index]
        assert entry["id"] == case["id"]
        assert scoring_entries[index]["id"] == entry["id"]
        variants = (
            ["Original", "Changed"]
            if case["pair_index"] % 2 == 0
            else ["Changed", "Original"]
        )
        for variant in variants:
            if (variant, entry["id"]) in done:
                continue
            model = models[variant]
            model.replayed_sources = []
            result, metadata = model.inference(copy.deepcopy(entry), True, False)
            checked = score(
                model,
                scoring_entries[index],
                result,
                None if not truth else truth[index]["ground_truth"],
                category,
            )
            row = {
                "variant": variant,
                "id": entry["id"],
                "index": index,
                "category": category,
                "pair_index": case["pair_index"],
                "order": variants.index(variant),
                "result": result,
                "metadata": metadata,
                "correct": checked["valid"],
                "check": checked,
                "bfcl_revision": actual,
                "replayed_native_responses": model.replayed_sources,
            }
            row = make_json_serializable(row)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            done.add((variant, entry["id"]))
            model.write(
                {"id": entry["id"], "result": result, **metadata}, TASK / "result"
            )
            print(entry["id"], variant, "correct", checked["valid"], flush=True)
    write_json(args.queue / "stop.json", {"status": "complete"})


if __name__ == "__main__":
    main()
