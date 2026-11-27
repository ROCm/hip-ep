#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[2] / ".github/llvm/packages.py"
spec = importlib.util.spec_from_file_location("packages", MODULE)
packages = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, GITHUB_REPOSITORY="example/hip-ep", DEFAULT_BRANCH="main"):
    spec.loader.exec_module(packages)


class LinuxPackagesTest(unittest.TestCase):
    def test_only_default_branch_publisher_can_supply_shared_package(self):
        def entry(number, branch, expired=False):
            return {
                "id": number,
                "expired": expired,
                "workflow_run": {"id": number, "head_branch": branch},
            }

        responses = [
            {
                "artifacts": [
                    entry(1, "main", True),
                    entry(2, "feature"),
                    entry(3, "main"),
                    entry(4, "main"),
                    entry(5, "main"),
                ]
            },
            {"path": packages.PUBLISHER, "event": "pull_request"},
            {"path": ".github/workflows/other.yml", "event": "workflow_run"},
            {"path": packages.PUBLISHER, "event": "workflow_run", "id": 5},
        ]
        with patch.object(packages, "api", side_effect=responses):
            self.assertEqual(
                packages.artifact("package"), {"artifact_id": 5, "run_id": 5}
            )

    def test_concurrent_requests_use_the_same_publication_group(self):
        sha, recipe = "a" * 40, "b" * 40
        key = f"llvm-ubuntu24-x64-x86-amdgpu-{sha}-{recipe}"
        jobs = {"jobs": [{"name": f"llvm / await {key}", "status": "in_progress"}]}
        with (
            patch.object(packages, "recipe_hash", return_value=recipe),
            patch.object(packages, "api", return_value=jobs),
            patch.object(packages, "output") as out,
        ):
            for run in ("1", "2"):
                packages.request(run)
                out.assert_called_with(name=key, variant="x86-amdgpu", sha=sha)

    def test_changed_recipe_builds_only_a_local_preview(self):
        tree = {
            "truncated": False,
            "tree": [
                {"path": packages.RECIPE, "sha": "1" * 40},
                {"path": packages.PUBLISHER, "sha": "2" * 40},
            ],
        }
        with (
            patch.object(packages, "recipe_hash", return_value="3" * 40),
            patch.object(packages, "pinned_sha", return_value="4" * 40),
            patch.object(packages, "api", return_value=tree),
            patch.object(packages, "artifact") as lookup,
            patch.object(packages, "output") as out,
        ):
            packages.plan("x86")
            self.assertEqual(out.call_args.kwargs["preview"], "true")
            lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
