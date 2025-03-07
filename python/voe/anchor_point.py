##
##  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
import json
import math


class _AnchorPointTag(str):
    pass


def transpose(order: _AnchorPointTag) -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": "transpose", "attribute": {"transposeAttr": {"order": order}}},
            sort_keys=True,
            indent=4,
        )
    )


def fix(type: str, fixpoint: str) -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": type, "attribute": {"fixAttr": {"fixPoint": fixpoint}}},
            sort_keys=True,
            indent=4,
        )
    )


def cast() -> _AnchorPointTag:
    return _AnchorPointTag(
        json.dumps(
            {"opType": "cast"},
            sort_keys=True,
            indent=4,
        )
    )


def const() -> _AnchorPointTag:
    return _AnchorPointTag(json.dumps({"opType": "const"}, sort_keys=True, indent=4))


def FLOAT2FIX(fixpoint: int) -> _AnchorPointTag:
    return fix("float2fix", fixpoint)


def FIX(fixpoint: int) -> _AnchorPointTag:
    return fix("fix", fixpoint)


def FIX2FLOAT(fixpoint: int) -> _AnchorPointTag:
    return fix("fix2float", fixpoint)


def is_anchor_point(obj) -> bool:
    return isinstance(obj, _AnchorPointTag)


def scale_to_fixpoint(scale) -> int:
    return int(math.log(scale, 1 / 2))


NCHW2NHWC = transpose([0, 2, 3, 1])
NCCHW2NHWCC = transpose([0, 3, 4, 1, 2])
CONST = const()
CAST = cast()
