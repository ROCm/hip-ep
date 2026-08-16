#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Cross-check HIP op roles and repository-convention DPS contract patterns.

The primary policy checks require exactly one explicit behavior family, reject
bare DPS leaves, and audit semantic destination-shape fallbacks. Family-owned
metadata consistency is intentionally secondary. Handwritten C++ checks
tokenize supported in-tree forms and fail closed on count drift; they are not a
general C++ parser.
"""

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
    "same_shape": (
        "Hip_DpsOp_SameShape",
        "Hip_DpsOp_SameShapeManualVerify",
    ),
    "broadcast": ("Hip_DpsOp_Broadcast",),
    "reduction": ("Hip_DpsOp_Reduction",),
    "semantic": (
        "Hip_DpsOp_Semantic",
        "Hip_DpsOp_SemanticNoInfer",
        "Hip_DpsOp_SemanticAutoReifyInfer",
    ),
    "payload": (
        "Hip_DpsOp_Payload",
        "Hip_DpsOp_PayloadNoInfer",
        "Hip_DpsOp_PayloadAutoReify",
    ),
    "outs_authoritative": ("Hip_DpsOp_OutsAuthoritative",),
}
SEMANTIC_OUTS_REIFY_ALLOWLIST = {
    "gather_nd": {
        "status": "shared infer/reify/verifier; dynamic tuple width uses outs fallback",
        "generated": 0,
        "handwritten": 1,
        "mixed_size": 0,
        "cpp_class": "GatherNDOp",
        "reifier_markers": ("if (failed(dims))",),
        "verifier_markers": ("indicesType.isDynamicDim(indicesType.getRank() - 1)",),
    },
    "gqa": {
        "status": (
            "shared converter extent utility + payload fallback + partial "
            "static verifier"
        ),
        "generated": 0,
        "handwritten": 1,
        "mixed_size": 0,
        "cpp_class": "GqaOp",
        "reifier_markers": (
            "getPositionIds()",
            "getOutputQk()",
            "getQkOutput()",
            "getSoftcap()",
        ),
        "verifier_markers": (
            "getPositionIds()",
            "getOutputQk()",
            "getQkOutput()",
            "getSoftcap()",
        ),
    },
    "size": {
        "status": "shared infer/converter/verifier",
        "generated": 1,
        "handwritten": 0,
        "mixed_size": 0,
        "cpp_class": "SizeOp",
        "reifier_markers": (),
        "verifier_markers": (),
    },
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
DEFAULT_OUTS_REIFY_MARKER = "::mlir::hip::HipDpsOp"
HANDWRITTEN_OUTS_REIFY_MARKER = re.compile(r"\bcast<(?:::mlir::hip::)?HipDpsOp>")
DPS_BASE = "Hip_DpsOp"
HIP_OP_BASE = "Hip_Op"
DESTINATION_STYLE_INTERFACE = "DestinationStyleOpInterface"
INVENTORY_ROW = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|[^|]*\| ([^|]+) \|$")
CONTROL_DPS_ROW = re.compile(r"^\| `([^`]+)` \| `control_flow_dps` \| ([^|]+) \|$")
CPP_TOKEN = re.compile(
    r"""
    //[^\n]* | /\*.*?\*/ |
    "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' |
    :: | -> | == | != | <= | >= | && | \|\| |
    [A-Za-z_]\w* | \d+ | \S
    """,
    re.DOTALL | re.VERBOSE,
)
DESTINATION_GETTERS = {
    "getC",
    "getDpsInit",
    "getDpsInits",
    "getOInit",
    "getOutput",
    "getOutputs",
    "getResult",
    "getResultTensors",
    "getY",
}
MIXED_SIZE_CALLS = {"getMixedSize", "getMixedSizes"}


def _cpp_tokens(text: str) -> list[str]:
    return [
        token
        for token in CPP_TOKEN.findall(text)
        if not token.startswith(("//", "/*", '"', "'"))
    ]


def _contains_destination_source(tokens: list[str], aliases: set[str]) -> bool:
    for index, token in enumerate(tokens):
        if token in aliases:
            return True
        if token in DESTINATION_GETTERS and tokens[index + 1 : index + 2] == ["("]:
            return True
    return False


def _call_arguments(tokens: list[str], open_paren: int) -> list[str]:
    depth = 0
    for index in range(open_paren, len(tokens)):
        if tokens[index] == "(":
            depth += 1
        elif tokens[index] == ")":
            depth -= 1
            if depth == 0:
                return tokens[open_paren + 1 : index]
    raise ValueError("unterminated C++ call expression in handwritten reifier")


def _count_destination_mixed_size_lifts(text: str) -> int:
    tokens = _cpp_tokens(text)
    aliases: set[str] = set()
    statements: list[list[str]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token == ";":
            statements.append(tokens[start:index])
            start = index + 1
    if start < len(tokens):
        statements.append(tokens[start:])

    changed = True
    while changed:
        changed = False
        for statement in statements:
            if "=" not in statement:
                continue
            equals = statement.index("=")
            lhs_identifiers = [
                token
                for token in statement[:equals]
                if re.fullmatch(r"[A-Za-z_]\w*", token)
            ]
            if not lhs_identifiers:
                continue
            alias = lhs_identifiers[-1]
            if alias not in aliases and _contains_destination_source(
                statement[equals + 1 :], aliases
            ):
                aliases.add(alias)
                changed = True

    count = 0
    for statement in statements:
        expression_is_destination_rooted = _contains_destination_source(
            statement, aliases
        )
        for index, token in enumerate(statement):
            if token not in MIXED_SIZE_CALLS or statement[index + 1 : index + 2] != [
                "("
            ]:
                continue
            arguments = _call_arguments(statement, index + 1)
            receiver_is_destination = (
                index >= 2
                and statement[index - 1] in {".", "->"}
                and statement[index - 2] in aliases
            )
            if (
                expression_is_destination_rooted
                or receiver_is_destination
                or _contains_destination_source(arguments, aliases)
            ):
                count += 1
    return count


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


def _method_body(text: str, cpp_class: str, method: str) -> str | None:
    matches = list(re.finditer(rf"\b{re.escape(cpp_class)}::{method}\s*\(", text))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one {cpp_class}::{method} definition")
    open_brace = text.find("{", matches[0].end())
    if open_brace < 0:
        raise ValueError(f"{cpp_class}::{method} has no function body")
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    raise ValueError(f"{cpp_class}::{method} has an unterminated function body")


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
    reifiers_text: str = "",
    verifiers_text: str = "",
    require_complete_allowlist: bool = False,
) -> tuple[dict[str, str], dict[str, str], set[str], set[str], set[str]]:
    contracts: dict[str, str] = {}
    same_shape_mechanisms: dict[str, str] = {}
    outs_reify_exceptions: set[str] = set()
    control_dps_ops: set[str] = set()
    non_dps_ops: set[str] = set()
    for name, untyped_record in records.items():
        if not (name.startswith("Hip_") and name.endswith("Op")):
            continue
        if not isinstance(untyped_record, dict):
            continue
        record = untyped_record
        superclasses = record.get("!superclasses", [])
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
                "shape-contract base"
            )
        if contract == "unclassified":
            raise ValueError(f"{name} ({op_name}) has no reviewed shape contract")
        if contract not in ALLOWED_CONTRACTS:
            raise ValueError(
                f"{name} ({op_name}) has unknown shape contract {contract!r}"
            )

        all_contract_bases = {
            base
            for contract_bases in CONTRACT_BASES.values()
            for base in contract_bases
        }
        inherited_category_bases = sorted(set(superclasses) & all_contract_bases)
        if len(inherited_category_bases) != 1:
            raise ValueError(
                f"{name} ({op_name}) must inherit exactly one DPS shape "
                f"category base, found {inherited_category_bases}"
            )
        expected_bases = CONTRACT_BASES[contract]
        if inherited_category_bases[0] not in expected_bases:
            raise ValueError(
                f"{name} ({op_name}) contract metadata {contract!r} conflicts "
                f"with inherited base {inherited_category_bases[0]}"
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
            generated_count = (record.get("extraClassDefinition") or "").count(
                DEFAULT_OUTS_REIFY_MARKER
            )
            fallback = SEMANTIC_OUTS_REIFY_ALLOWLIST.get(op_name)
            cpp_class = (
                fallback["cpp_class"]
                if fallback
                else record.get("cppClassName", name.removeprefix("Hip_"))
            )
            reifier_body = _method_body(
                reifiers_text, str(cpp_class), "reifyResultShapes"
            )
            handwritten_count = (
                len(HANDWRITTEN_OUTS_REIFY_MARKER.findall(reifier_body))
                if reifier_body
                else 0
            )
            mixed_size_count = (
                _count_destination_mixed_size_lifts(reifier_body) if reifier_body else 0
            )
            if generated_count or handwritten_count or mixed_size_count:
                if fallback is None:
                    raise ValueError(
                        f"{name} ({op_name}) semantic contract uses outs-lift "
                        "reification without a reviewed exception"
                    )
                if (
                    generated_count != fallback["generated"]
                    or (handwritten_count != fallback["handwritten"])
                    or mixed_size_count != fallback["mixed_size"]
                ):
                    raise ValueError(
                        f"{name} ({op_name}) outs-lift count mismatch: "
                        f"generated={generated_count}, "
                        f"handwritten={handwritten_count}, "
                        f"mixed_size={mixed_size_count}"
                    )
                for marker in fallback["reifier_markers"]:
                    if not reifier_body or marker not in reifier_body:
                        raise ValueError(
                            f"{name} ({op_name}) handwritten fallback is "
                            f"missing reifier guard {marker!r}"
                        )
                if handwritten_count:
                    verifier_body = _method_body(
                        verifiers_text, str(cpp_class), "verify"
                    )
                    if verifier_body is None:
                        raise ValueError(
                            f"{name} ({op_name}) handwritten fallback requires "
                            "a handwritten verifier"
                        )
                    for marker in fallback["verifier_markers"]:
                        if marker not in verifier_body:
                            raise ValueError(
                                f"{name} ({op_name}) handwritten fallback is "
                                f"missing verifier guard {marker!r}"
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
    return (
        contracts,
        same_shape_mechanisms,
        outs_reify_exceptions,
        control_dps_ops,
        non_dps_ops,
    )


def _load_tablegen(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str], set[str], set[str], set[str]]:
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
    return _audit_tablegen(
        records,
        reifiers_text=args.reifiers.read_text(),
        verifiers_text=args.verifiers.read_text(),
        require_complete_allowlist=True,
    )


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
    outs_reify_exceptions: set[str],
) -> None:
    for op_name, expected_status in same_shape_mechanisms.items():
        if statuses[op_name] != expected_status:
            raise ValueError(
                f"{op_name}: same-shape mechanism requires inventory status "
                f"{expected_status!r}, found {statuses[op_name]!r}"
            )
    for op_name in outs_reify_exceptions:
        expected_status = SEMANTIC_OUTS_REIFY_ALLOWLIST[op_name]["status"]
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
        "!superclasses": [
            HIP_OP_BASE,
            DPS_BASE,
            base or CONTRACT_BASES[contract][0],
        ],
        "_testInterfaces": [DESTINATION_STYLE_INTERFACE],
        "hasVerifier": int(verifier),
        "extraClassDefinition": DEFAULT_OUTS_REIFY_MARKER if outs_reify else "",
    }


def _run_self_test() -> int:
    def expect_failure(
        record: dict[str, object],
        message: str,
        *,
        reifiers_text: str = "",
        verifiers_text: str = "",
    ) -> None:
        try:
            _audit_tablegen(
                {"Hip_FixtureOp": record},
                reifiers_text=reifiers_text,
                verifiers_text=verifiers_text,
            )
        except ValueError as exc:
            if message not in str(exc):
                raise AssertionError(f"expected {message!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"expected failure containing {message!r}")

    expect_failure(_fixture_record("semantic", explicit=False), "explicit HIP DPS")

    missing_category = _fixture_record("semantic", base=DPS_BASE)
    expect_failure(missing_category, "exactly one DPS shape category base")

    multiple_categories = _fixture_record("semantic")
    multiple_categories["!superclasses"].append(CONTRACT_BASES["broadcast"][0])
    expect_failure(multiple_categories, "exactly one DPS shape category base")

    expect_failure(
        _fixture_record("broadcast", base="Hip_DpsOp_Semantic"),
        "contract metadata 'broadcast' conflicts with inherited base",
    )
    expect_failure(
        _fixture_record("semantic", verifier=False), "requires verifier wiring"
    )
    expect_failure(
        _fixture_record("semantic", outs_reify=True),
        "without a reviewed exception",
    )

    payload = _fixture_record("payload")
    contracts, same_shape, exceptions, control_dps, non_dps = _audit_tablegen(
        {"Hip_FixtureOp": payload}
    )
    assert not control_dps and not non_dps
    try:
        _validate_inventory_statuses(
            contracts, {"fixture": "no-op stub"}, same_shape, exceptions
        )
    except ValueError as exc:
        if "reviewed inventory status" not in str(exc):
            raise
    else:
        raise AssertionError("payload no-op status unexpectedly passed")

    size = _fixture_record("semantic", op_name="size", outs_reify=True)
    contracts, same_shape, exceptions, control_dps, non_dps = _audit_tablegen(
        {"Hip_SizeOp": size}
    )
    assert not control_dps and not non_dps
    _validate_inventory_statuses(
        contracts,
        {"size": SEMANTIC_OUTS_REIFY_ALLOWLIST["size"]["status"]},
        same_shape,
        exceptions,
    )

    missing_contract = _fixture_record("semantic")
    del missing_contract["hipShapeContract"]
    expect_failure(missing_contract, "without shape-contract metadata")

    non_dps_with_fake_contract = _fixture_record("payload")
    non_dps_with_fake_contract["!superclasses"] = [HIP_OP_BASE]
    expect_failure(
        non_dps_with_fake_contract,
        "without both Hip_DpsOp and DestinationStyleOpInterface",
    )

    missing_interface = _fixture_record("semantic")
    missing_interface["_testInterfaces"] = []
    expect_failure(missing_interface, "without both Hip_DpsOp")

    contracts, same_shape, exceptions, control_dps, non_dps = _audit_tablegen(
        {
            "Hip_ConstantOp": {
                "opName": "constant",
                "!superclasses": [HIP_OP_BASE],
            },
            "Hip_ReadbackControlOp": {
                "opName": "readback_control",
                "!superclasses": [HIP_OP_BASE],
            },
            "Hip_LoopOp": {
                "opName": "loop",
                "!superclasses": [HIP_OP_BASE],
            },
            "Hip_IfOp": {
                "opName": "if",
                "!superclasses": [HIP_OP_BASE],
                "_testInterfaces": [DESTINATION_STYLE_INTERFACE],
            },
        }
    )
    assert not contracts and not same_shape and not exceptions
    assert control_dps == {"if"}
    assert non_dps == {"constant", "readback_control", "loop"}

    gqa = _fixture_record("semantic", op_name="gqa")
    gqa_reifier = """
LogicalResult GqaOp::reifyResultShapes() {
  getPositionIds();
  getOutputQk();
  getQkOutput();
  getSoftcap();
  return cast<::mlir::hip::HipDpsOp>(getOperation()).reifyResultShapes();
}
"""
    gqa_verifier = """
LogicalResult GqaOp::verify() {
  getPositionIds();
  getOutputQk();
  getQkOutput();
  getSoftcap();
  return success();
}
"""
    _, _, exceptions, _, _ = _audit_tablegen(
        {"Hip_GqaOp": gqa},
        reifiers_text=gqa_reifier,
        verifiers_text=gqa_verifier,
    )
    assert exceptions == {"gqa"}
    expect_failure(
        gqa,
        "missing verifier guard 'getSoftcap()'",
        reifiers_text=gqa_reifier,
        verifiers_text=gqa_verifier.replace("getSoftcap();", ""),
    )
    expect_failure(
        gqa,
        "outs-lift count mismatch",
        reifiers_text=gqa_reifier.replace(
            "return cast<::mlir::hip::HipDpsOp>",
            "cast<::mlir::hip::HipDpsOp>(getOperation()).reifyResultShapes();\n"
            "  return cast<::mlir::hip::HipDpsOp>",
        ),
        verifiers_text=gqa_verifier,
    )
    expect_failure(
        gqa,
        "mixed_size=1",
        reifiers_text=gqa_reifier.replace(
            "getSoftcap();",
            "getSoftcap();\n"
            "  auto dims = tensor::getMixedSizes(b, getLoc(), getOutput());",
        ),
        verifiers_text=gqa_verifier,
    )

    gather_nd = _fixture_record("semantic", op_name="gather_nd")
    gather_reifier = """
LogicalResult GatherNDOp::reifyResultShapes() {
  if (failed(dims))
    return cast<HipDpsOp>(getOperation()).reifyResultShapes();
  return success();
}
"""
    gather_verifier = """
LogicalResult GatherNDOp::verify() {
  if (indicesType.isDynamicDim(indicesType.getRank() - 1))
    return success();
  return verifyShape();
}
"""
    _, _, exceptions, _, _ = _audit_tablegen(
        {"Hip_GatherNDOp": gather_nd},
        reifiers_text=gather_reifier,
        verifiers_text=gather_verifier,
    )
    assert exceptions == {"gather_nd"}
    expect_failure(
        gather_nd,
        "missing verifier guard",
        reifiers_text=gather_reifier,
        verifiers_text=gather_verifier.replace(
            "indicesType.isDynamicDim(indicesType.getRank() - 1)", "dynamic"
        ),
    )

    unreviewed = _fixture_record("semantic")
    expect_failure(
        unreviewed,
        "without a reviewed exception",
        reifiers_text="""
LogicalResult FixtureOp::reifyResultShapes() {
  return cast<::mlir::hip::HipDpsOp>(getOperation()).reifyResultShapes();
}
""",
    )
    mixed_size_bypasses = (
        """
LogicalResult FixtureOp::reifyResultShapes() {
  auto dims = tensor::getMixedSizes(b, getLoc(), getOutput());
  return success();
}
""",
        """
LogicalResult FixtureOp::reifyResultShapes() {
  auto init = getDpsInits()[0];
  auto alias = init;
  auto dims = mlir :: memref :: getMixedSizes (b, getLoc(), alias);
  return success();
}
""",
        """
LogicalResult FixtureOp::reifyResultShapes() {
  auto init = getOutput();
  auto empty = init.getDefiningOp<tensor::EmptyOp>();
  auto dims = empty.getMixedSizes();
  return success();
}
""",
        """
LogicalResult FixtureOp::reifyResultShapes() {
  auto dims = cast<tensor::EmptyOp>(
      getDpsInit().getDefiningOp()).getMixedSizes();
  return success();
}
""",
    )
    for reifier in mixed_size_bypasses:
        expect_failure(
            unreviewed,
            "without a reviewed exception",
            reifiers_text=reifier,
        )
    print("verified shape-contract guard fixtures")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tblgen", type=Path)
    parser.add_argument("--project-include", type=Path)
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--reifiers", type=Path)
    parser.add_argument("--verifiers", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            return _run_self_test()
        required = (
            "tblgen",
            "project_include",
            "ops",
            "inventory",
            "reifiers",
            "verifiers",
        )
        if missing := [
            f"--{name.replace('_', '-')}"
            for name in required
            if not getattr(args, name)
        ]:
            parser.error(f"the following arguments are required: {', '.join(missing)}")

        (
            tablegen,
            same_shape_mechanisms,
            outs_reify_exceptions,
            control_dps_ops,
            non_dps_ops,
        ) = _load_tablegen(args)
        inventory, inventory_statuses, inventory_control_dps = _load_inventory(
            args.inventory
        )
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
            tablegen,
            inventory_statuses,
            same_shape_mechanisms,
            outs_reify_exceptions,
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
