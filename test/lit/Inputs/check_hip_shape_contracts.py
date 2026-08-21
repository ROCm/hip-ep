#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Cross-check structural HIP DPS contracts against the reviewed inventory.

The checker consumes llvm-tblgen's JSON record graph. It does not parse C++:
generated verifier declarations, C++ compilation/linking, and focused MLIR
negative tests enforce method-level behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


CONTRACT_FAMILIES = {
    "same_shape": {
        "Hip_DpsOp_SameShape",
        "Hip_DpsOp_SameShapeManualVerify",
    },
    "broadcast": {"Hip_DpsOp_Broadcast"},
    "reduction": {"Hip_DpsOp_Reduction"},
    "semantic": {
        "Hip_DpsOp_Semantic",
        "Hip_DpsOp_SemanticNoInfer",
        "Hip_DpsOp_SemanticAutoReifyInfer",
        "Hip_DpsOp_SemanticAutoReifyManualVerify",
    },
    "payload": {
        "Hip_DpsOp_Payload",
        "Hip_DpsOp_PayloadManualVerify",
        "Hip_DpsOp_PayloadNoInfer",
        "Hip_DpsOp_PayloadAutoReify",
    },
    "outs_authoritative": {"Hip_DpsOp_OutsAuthoritative"},
}
VERIFIER_FAMILIES = {
    "Hip_DpsOp_SameShape",
    "Hip_DpsOp_SameShapeManualVerify",
    "Hip_DpsOp_Broadcast",
    "Hip_DpsOp_Reduction",
    "Hip_DpsOp_Semantic",
    "Hip_DpsOp_SemanticNoInfer",
    "Hip_DpsOp_SemanticAutoReifyInfer",
    "Hip_DpsOp_SemanticAutoReifyManualVerify",
    "Hip_DpsOp_PayloadManualVerify",
}
REVIEWED_POLICY_STATUSES = {
    "payload": {
        "audited payload policy",
        "affine refinement complete",
        "complete; exact grouped control readback",
        "refinement complete; shared constant rule + one bulk runtime readback",
    },
    "outs_authoritative": {"audited outs-authoritative"},
}
DPS_BASE = "Hip_DpsOp"
HIP_OP_BASE = "Hip_Op"
DESTINATION_STYLE_INTERFACE = "DestinationStyleOpInterface"
INVENTORY_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|[^|]*\| ([^|]+) \|$")
CONTROL_DPS_ROW = re.compile(r"^\| `([^`]+)` \| `control_flow_dps` \| ([^|]+) \|$")


def _implements_interface(
    record: dict[str, object],
    records: dict[str, object],
    interface_name: str,
) -> bool:
    if interface_name in record.get("_testInterfaces", []):
        return True
    for trait in record.get("traits", []):
        if not isinstance(trait, dict):
            continue
        trait_record = records.get(trait.get("def"))
        if isinstance(trait_record, dict) and (
            trait_record.get("cppInterfaceName") == interface_name
        ):
            return True
    return False


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
) -> tuple[dict[str, str], dict[str, str], set[str], set[str]]:
    contracts: dict[str, str] = {}
    same_shape_mechanisms: dict[str, str] = {}
    control_dps_ops: set[str] = set()
    non_dps_ops: set[str] = set()
    all_families = set().union(*CONTRACT_FAMILIES.values())

    for name, untyped_record in records.items():
        if not (name.startswith("Hip_") and name.endswith("Op")):
            continue
        if not isinstance(untyped_record, dict):
            continue
        record = untyped_record
        superclasses = set(record.get("!superclasses", []))
        inherits_compute_dps = DPS_BASE in superclasses
        implements_dps = _implements_interface(
            record, records, DESTINATION_STYLE_INTERFACE
        )
        contract = record.get("hipShapeContract")

        if contract is None:
            if inherits_compute_dps:
                raise ValueError(
                    f"{name} ({record['opName']}) is a HIP DPS op without "
                    "shape-contract metadata"
                )
            if HIP_OP_BASE not in superclasses:
                raise ValueError(f"{name} is not derived from the HIP op base")
            if implements_dps:
                control_dps_ops.add(record["opName"])
            else:
                non_dps_ops.add(record["opName"])
            continue

        op_name = record["opName"]
        if not inherits_compute_dps or not implements_dps:
            raise ValueError(
                f"{name} ({op_name}) carries a DPS shape contract without "
                "both Hip_DpsOp and DestinationStyleOpInterface"
            )
        if not record.get("hipHasExplicitShapeContract", 0):
            raise ValueError(
                f"{name} ({op_name}) must inherit an explicit HIP DPS "
                "shape-contract family"
            )
        if contract == "unclassified":
            raise ValueError(f"{name} ({op_name}) has no reviewed shape contract")
        if contract not in CONTRACT_FAMILIES:
            raise ValueError(
                f"{name} ({op_name}) has unknown shape contract {contract!r}"
            )

        inherited_families = sorted(superclasses & all_families)
        if len(inherited_families) != 1:
            raise ValueError(
                f"{name} ({op_name}) must inherit exactly one DPS shape "
                f"behavior family, found {inherited_families}"
            )
        family = inherited_families[0]
        if family not in CONTRACT_FAMILIES[contract]:
            raise ValueError(
                f"{name} ({op_name}) contract metadata {contract!r} conflicts "
                f"with inherited behavior family {family}"
            )
        if family in VERIFIER_FAMILIES and not record.get("hasVerifier", 0):
            raise ValueError(
                f"{name} ({op_name}) behavior family {family} requires "
                "generated verifier wiring"
            )

        same_shape_source = record.get("hipSameShapeSource", "")
        if contract == "same_shape":
            if not same_shape_source:
                raise ValueError(
                    f"{name} ({op_name}) uses a same-shape family without a "
                    "named source accessor"
                )
            same_shape_mechanisms[op_name] = "shared named-source base"
        elif same_shape_source:
            raise ValueError(
                f"{name} ({op_name}) has same-shape source metadata but "
                f"contract {contract!r}"
            )

        if op_name in contracts:
            raise ValueError(f"duplicate HIP op name in TableGen: {op_name}")
        contracts[op_name] = contract

    return contracts, same_shape_mechanisms, control_dps_ops, non_dps_ops


def _load_tablegen(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str], set[str], set[str]]:
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
    return _audit_tablegen(json.loads(result.stdout))


def _load_inventory(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    contracts: dict[str, str] = {}
    statuses: dict[str, str] = {}
    control_dps_ops: set[str] = set()
    for line in path.read_text().splitlines():
        control_match = CONTROL_DPS_ROW.match(line)
        if control_match:
            op_name, _ = control_match.groups()
            if op_name in control_dps_ops:
                raise ValueError(
                    f"duplicate control-flow DPS op in inventory: {op_name}"
                )
            control_dps_ops.add(op_name)
            continue
        match = INVENTORY_ROW.match(line)
        if not match:
            continue
        op_name, contract, status = match.groups()
        if op_name in contracts:
            raise ValueError(f"duplicate HIP op in inventory: {op_name}")
        contracts[op_name] = contract
        statuses[op_name] = status.strip()
    return contracts, statuses, control_dps_ops


def _validate_inventory_statuses(
    contracts: dict[str, str],
    statuses: dict[str, str],
    same_shape_mechanisms: dict[str, str],
) -> None:
    for op_name, expected_status in same_shape_mechanisms.items():
        if statuses[op_name] != expected_status:
            raise ValueError(
                f"{op_name}: same-shape mechanism requires inventory status "
                f"{expected_status!r}, found {statuses[op_name]!r}"
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


def _inventory_differences(
    tablegen: dict[str, str],
    inventory: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    return (
        sorted(set(tablegen) - set(inventory)),
        sorted(set(inventory) - set(tablegen)),
        sorted(
            op for op in set(tablegen) & set(inventory) if tablegen[op] != inventory[op]
        ),
    )


def _fixture_record(
    contract: str,
    *,
    op_name: str = "fixture",
    explicit: bool = True,
    family: str | None = None,
    verifier: bool = True,
) -> dict[str, object]:
    family = family or sorted(CONTRACT_FAMILIES[contract])[0]
    return {
        "opName": op_name,
        "hipShapeContract": contract,
        "hipHasExplicitShapeContract": int(explicit),
        "hipSameShapeSource": "Input" if contract == "same_shape" else "",
        "!superclasses": [HIP_OP_BASE, DPS_BASE, family],
        "_testInterfaces": [DESTINATION_STYLE_INTERFACE],
        "hasVerifier": int(verifier),
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

    bare_dps = {
        "opName": "fixture",
        "hipShapeContract": "unclassified",
        "hipHasExplicitShapeContract": 0,
        "hipSameShapeSource": "",
        "!superclasses": [HIP_OP_BASE, DPS_BASE],
        "_testInterfaces": [DESTINATION_STYLE_INTERFACE],
    }
    expect_failure(bare_dps, "explicit HIP DPS shape-contract family")

    missing_family = _fixture_record("semantic")
    missing_family["!superclasses"] = [HIP_OP_BASE, DPS_BASE]
    expect_failure(missing_family, "exactly one DPS shape behavior family")

    multiple_families = _fixture_record("semantic")
    multiple_families["!superclasses"].append("Hip_DpsOp_Broadcast")
    expect_failure(multiple_families, "exactly one DPS shape behavior family")

    expect_failure(
        _fixture_record("broadcast", family="Hip_DpsOp_Semantic"),
        "conflicts with inherited behavior family",
    )
    expect_failure(
        _fixture_record("semantic", verifier=False),
        "requires generated verifier wiring",
    )
    expect_failure(
        _fixture_record(
            "same_shape",
            family="Hip_DpsOp_SameShapeManualVerify",
            verifier=False,
        ),
        "requires generated verifier wiring",
    )

    payload = _fixture_record("payload", verifier=False)
    contracts, same_shape, control_dps, non_dps = _audit_tablegen(
        {"Hip_FixtureOp": payload}
    )
    assert not same_shape and not control_dps and not non_dps
    try:
        _validate_inventory_statuses(contracts, {"fixture": "no-op stub"}, same_shape)
    except ValueError as exc:
        if "reviewed inventory status" not in str(exc):
            raise
    else:
        raise AssertionError("payload no-op status unexpectedly passed")

    control_and_non_dps = {
        "Hip_ConstantOp": {
            "opName": "constant",
            "!superclasses": [HIP_OP_BASE],
        },
        "Hip_IfOp": {
            "opName": "if",
            "!superclasses": [HIP_OP_BASE],
            "_testInterfaces": [DESTINATION_STYLE_INTERFACE],
        },
    }
    contracts, _, control_dps, non_dps = _audit_tablegen(control_and_non_dps)
    assert not contracts
    assert control_dps == {"if"}
    assert non_dps == {"constant"}

    missing, stale, mismatched = _inventory_differences(
        {"a": "semantic", "b": "payload"},
        {"b": "semantic", "c": "payload"},
    )
    assert missing == ["a"] and stale == ["c"] and mismatched == ["b"]

    print("verified structural shape-contract guard fixtures")
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

        tablegen, same_shape_mechanisms, control_dps_ops, non_dps_ops = _load_tablegen(
            args
        )
        inventory, inventory_statuses, inventory_control_dps = _load_inventory(
            args.inventory
        )
        missing, stale, mismatched = _inventory_differences(tablegen, inventory)
        if missing or stale or mismatched:
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
        if control_dps_ops != inventory_control_dps:
            missing = sorted(control_dps_ops - inventory_control_dps)
            stale = sorted(inventory_control_dps - control_dps_ops)
            if missing:
                print(
                    f"missing control-flow DPS inventory ops: {', '.join(missing)}",
                    file=sys.stderr,
                )
            if stale:
                print(
                    f"stale control-flow DPS inventory ops: {', '.join(stale)}",
                    file=sys.stderr,
                )
            return 1

        _validate_inventory_statuses(
            tablegen, inventory_statuses, same_shape_mechanisms
        )
        print(
            f"verified {len(tablegen)} HIP DPS shape contracts and "
            f"{len(control_dps_ops)} control-flow DPS op and "
            f"{len(non_dps_ops)} structurally non-DPS HIP ops"
        )
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
