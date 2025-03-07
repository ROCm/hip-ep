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
from voe.pattern import node, wildcard


def pattern():
    pat_input = wildcard()
    pat_score = wildcard()
    pat_max_output = wildcard()
    pat_threshold = wildcard()
    pat_nms = node(
        "NonMaxSuppression", pat_input, pat_score, pat_max_output, pat_threshold
    )
    return pat_nms.build(locals())


print(pattern())
