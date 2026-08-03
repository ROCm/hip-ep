#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Cross-check HIP DPS shape-contract metadata against the design inventory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ALLOWED_CONTRACTS = {
    "same_shape",
    "broadcast",
    "reduction",
    "semantic",
    "payload",
    "outs_authoritative",
}
CONTRACT_BASES = {
    "same_shape": "Hip_DpsOp_SameShape",
    "broadcast": "Hip_DpsOp_Broadcast",
    "reduction": "Hip_DpsOp_Reduction",
    "semantic": "Hip_DpsOp_Semantic",
    "payload": "Hip_DpsOp_Payload",
    "outs_authoritative": "Hip_DpsOp_OutsAuthoritative",
}
SEMANTIC_OUTS_REIFY_ALLOWLIST = {
    "gqa": "shared converter extent utility + payload fallback + partial static verifier",
    "size": "shared infer/converter/verifier",
}
REVIEWED_POLICY_STATUSES = {
    "payload": {
        "audited payload policy",
        "affine refinement complete",
        "complete; dynamic clamp deliberately capacity-authoritative",
        "refinement complete; shared constant rule + one bulk runtime readback",
    },
    "outs_authoritative": {"audited outs-authoritative"},
}
DEFAULT_OUTS_REIFY_MARKER = "::mlir::hip::HipDpsOp"
INVENTORY_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|[^|]*\| ([^|]+) \|$")


def _llvm_include_dir(tblgen: Path) -> Path:
    suffix = ".exe" if tblgen.suffix.lower() == ".exe" else ""
    llvm_config = tblgen.with_name(f"llvm-config{suffix}")
    if llvm_config.exists():
        result = subprocess.run(
            [str(llvm_config), "--includedir"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip())
    return tblgen.parent.parent / "include"


def _audit_tablegen(
    records: dict[str, object],
    *,
    require_complete_allowlist: bool = False,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    contracts: dict[str, str] = {}
    same_shape_mechanisms: dict[str, str] = {}
    outs_reify_exceptions: set[str] = set()
    for name, untyped_record in records.items():
        if not (name.startswith("Hip_") and name.endswith("Op")):
            continue
        if not isinstance(untyped_record, dict):
            continue
        record = untyped_record
        contract = record.get("hipShapeContract")
        if contract is None:
            continue
        op_name = record["opName"]
        if not record.get("hipHasExplicitShapeContract", 0):
            raise ValueError(
                f"{name} ({op_name}) must inherit an explicit HIP DPS "
                "shape-contract base"
            )
        if contract == "unclassified":
            raise ValueError(f"{name} ({op_name}) has no reviewed shape contract")
        if contract not in ALLOWED_CONTRACTS:
            raise ValueError(
                f"{name} ({op_name}) has unknown shape contract {contract!r}"
            )

        expected_base = CONTRACT_BASES[contract]
        superclasses = record.get("!superclasses", [])
        if expected_base not in superclasses:
            raise ValueError(
                f"{name} ({op_name}) contract {contract!r} must inherit {expected_base}"
            )

        same_shape_source = record.get("hipSameShapeSource", "")
        if contract == "same_shape":
            if not same_shape_source:
                raise ValueError(
                    f"{name} ({op_name}) uses Hip_DpsOp_SameShape without a "
                    "named source accessor"
                )
            same_shape_mechanisms[op_name] = "shared named-source base"
        elif same_shape_source:
            raise ValueError(
                f"{name} ({op_name}) has same-shape source metadata but "
                f"contract {contract!r}"
            )

        if contract == "semantic":
            if not record.get("hasVerifier", 0):
                raise ValueError(
                    f"{name} ({op_name}) semantic contract requires verifier wiring"
                )
            if DEFAULT_OUTS_REIFY_MARKER in record.get("extraClassDefinition", ""):
                if op_name not in SEMANTIC_OUTS_REIFY_ALLOWLIST:
                    raise ValueError(
                        f"{name} ({op_name}) semantic contract uses the default "
                        "outs-lift reifier without a reviewed exception"
                    )
                outs_reify_exceptions.add(op_name)

        if op_name in contracts:
            raise ValueError(f"duplicate HIP op name in TableGen: {op_name}")
        contracts[op_name] = contract
    if require_complete_allowlist:
        stale_exceptions = sorted(
            set(SEMANTIC_OUTS_REIFY_ALLOWLIST) - outs_reify_exceptions
        )
        if stale_exceptions:
            raise ValueError(
                "stale semantic outs-lift allowlist entries: "
                f"{', '.join(stale_exceptions)}"
            )
    return contracts, same_shape_mechanisms, outs_reify_exceptions


def _load_tablegen(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    command = [
        str(args.tblgen),
        "--dump-json",
        "-I",
        str(args.project_include),
        "-I",
        str(_llvm_include_dir(args.tblgen)),
        str(args.ops),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    records = json.loads(result.stdout)
    return _audit_tablegen(records, require_complete_allowlist=True)


def _load_inventory(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    contracts: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = INVENTORY_ROW.match(line)
        if not match:
            continue
        op_name, contract, status = match.groups()
        if op_name in contracts:
            raise ValueError(f"duplicate HIP op in inventory: {op_name}")
        contracts[op_name] = contract
        statuses[op_name] = status.strip()
    return contracts, statuses


def _validate_inventory_statuses(
    contracts: dict[str, str],
    statuses: dict[str, str],
    same_shape_mechanisms: dict[str, str],
    outs_reify_exceptions: set[str],
) -> None:
    for op_name, expected_status in same_shape_mechanisms.items():
        if statuses[op_name] != expected_status:
            raise ValueError(
                f"{op_name}: same-shape mechanism requires inventory status "
                f"{expected_status!r}, found {statuses[op_name]!r}"
            )
    for op_name in outs_reify_exceptions:
        expected_status = SEMANTIC_OUTS_REIFY_ALLOWLIST[op_name]
        if statuses[op_name] != expected_status:
            raise ValueError(
                f"{op_name}: semantic outs-lift exception requires inventory "
                f"status {expected_status!r}, found {statuses[op_name]!r}"
            )
    for op_name, contract in contracts.items():
        allowed_statuses = REVIEWED_POLICY_STATUSES.get(contract)
        if allowed_statuses is None:
            continue
        status = statuses[op_name]
        if status not in allowed_statuses:
            raise ValueError(
                f"{op_name}: {contract} contract requires a reviewed inventory "
                f"status, found {status!r}"
            )


def _fixture_record(
    contract: str,
    *,
    op_name: str = "fixture",
    explicit: bool = True,
    base: str | None = None,
    verifier: bool = True,
    outs_reify: bool = False,
) -> dict[str, object]:
    return {
        "opName": op_name,
        "hipShapeContract": contract,
        "hipHasExplicitShapeContract": int(explicit),
        "hipSameShapeSource": "Input" if contract == "same_shape" else "",
        "!superclasses": [base or CONTRACT_BASES[contract]],
        "hasVerifier": int(verifier),
        "extraClassDefinition": DEFAULT_OUTS_REIFY_MARKER if outs_reify else "",
    }


def _run_self_test() -> int:
    def expect_failure(record: dict[str, object], message: str) -> None:
        try:
            _audit_tablegen({"Hip_FixtureOp": record})
        except ValueError as exc:
            if message not in str(exc):
                raise AssertionError(f"expected {message!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"expected failure containing {message!r}")

    expect_failure(_fixture_record("semantic", explicit=False), "explicit HIP DPS")
    expect_failure(
        _fixture_record("broadcast", base="Hip_DpsOp_Semantic"),
        "must inherit Hip_DpsOp_Broadcast",
    )
    expect_failure(
        _fixture_record("semantic", verifier=False), "requires verifier wiring"
    )
    expect_failure(
        _fixture_record("semantic", outs_reify=True),
        "without a reviewed exception",
    )

    payload = _fixture_record("payload")
    contracts, same_shape, exceptions = _audit_tablegen({"Hip_FixtureOp": payload})
    try:
        _validate_inventory_statuses(
            contracts, {"fixture": "no-op stub"}, same_shape, exceptions
        )
    except ValueError as exc:
        if "reviewed inventory status" not in str(exc):
            raise
    else:
        raise AssertionError("payload no-op status unexpectedly passed")

    gqa = _fixture_record("semantic", op_name="gqa", outs_reify=True)
    contracts, same_shape, exceptions = _audit_tablegen({"Hip_GqaOp": gqa})
    _validate_inventory_statuses(
        contracts,
        {"gqa": SEMANTIC_OUTS_REIFY_ALLOWLIST["gqa"]},
        same_shape,
        exceptions,
    )
    print("verified shape-contract guard fixtures")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tblgen", type=Path)
    parser.add_argument("--project-include", type=Path)
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            return _run_self_test()
        required = ("tblgen", "project_include", "ops", "inventory")
        if missing := [
            f"--{name.replace('_', '-')}"
            for name in required
            if not getattr(args, name)
        ]:
            parser.error(f"the following arguments are required: {', '.join(missing)}")

        tablegen, same_shape_mechanisms, outs_reify_exceptions = _load_tablegen(args)
        inventory, inventory_statuses = _load_inventory(args.inventory)
        if tablegen != inventory:
            missing = sorted(set(tablegen) - set(inventory))
            stale = sorted(set(inventory) - set(tablegen))
            mismatched = sorted(
                op
                for op in set(tablegen) & set(inventory)
                if tablegen[op] != inventory[op]
            )
            if missing:
                print(f"missing inventory ops: {', '.join(missing)}", file=sys.stderr)
            if stale:
                print(f"stale inventory ops: {', '.join(stale)}", file=sys.stderr)
            for op in mismatched:
                print(
                    f"{op}: TableGen={tablegen[op]}, inventory={inventory[op]}",
                    file=sys.stderr,
                )
            return 1
        _validate_inventory_statuses(
            tablegen,
            inventory_statuses,
            same_shape_mechanisms,
            outs_reify_exceptions,
        )
        print(f"verified {len(tablegen)} HIP DPS shape contracts")
        return 0
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"shape-contract audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
