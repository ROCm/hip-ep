<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Debug-only CPU fallback（ORT CPU EP Invoke）实施计划

本文记录 **开发/调试阶段** 在 `wrap_*` 内按环境变量将指定算子 fallback 到 **ORT CPU EP**（Quark 式 `Ort::Op::Create` + `Ort::Op::Invoke` + CPUGate）的方案与分阶段交付。**不进入 Release 默认产物**；正式推理仍以 **全 GPU** 为准，性能仅在 Debug 路径可忽略。

---

## 1. 目标与非目标

### 目标

- Debug/RelWithDebInfo（由 **CMake 宏** 控制）下，对 **指定算子** 在 `wrap_*` 中可走：
  **stream 同步 → 输入 D2H → EP 侧 CPUGate + `Ort::Op::Invoke`（CPU EP）→ 输出 H2D**，
  用于与 **ORT CPU EP** 对齐、定位精度差异。
- **`model.dll`（runtime bitcode）**：仅 **分支 + 同步 + memcpy + C ABI 回调**；**不**链接 ONNX Runtime、**不**内嵌 Gate。
- **`onnxruntime_morphizen_ep.dll`（或等价 EP 模块）**：通过 **`morphizen::OpInvoker`** 构造单算子小图 + ORT CPU Session 执行（首版 **Gather**）；**`Ort::Session`/`OpInvoker` 按 (axis, dtypes) 缓存**；在 **InferenceState** 生命周期内 **`hipdnn_ep_set_cpu_fallback` 注册/析构清空**。（历史方案曾讨论 Quark 式 CPUGate；当前实现优先复用 MorphiZen 已有 Invoker。）

### 非目标

- Release 默认构建 **不包含** 该路径（宏默认 OFF + 可选 `NDEBUG` 双重保险）。
- 不要求首日 **全算子** 覆盖；不要求 GPU fp16 与 CPU fp32 **bit 一致**（对比阈值单独约定）。

---

## 2. 架构

```
[model.dll — wrap_*]
  if (编译开启 debug fallback && 环境变量命中本算子)
        hipStreamSynchronize(stream)
        D2H → 填纯 C 描述符 → state->cpu_fallback.invoke(user, &desc)
        H2D → return
  else
        现有 GPU 路径（不变）

[EP — MorphiZen]
  inference_init 之后：get_method("hipdnn_ep_set_cpu_fallback") → 构造 Gate Manager → set_cpu_fallback
  invoke 实现：Ort::Value(borrow host) → Ort::Op::Invoke(kernel_ctx_, …)
  Session 销毁：停 Gate 线程、释放 Manager、set_cpu_fallback(null)
```

**环境变量（命名可微调，实现时以代码为准）**

- `HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS`：逗号分隔列表，与 **内部 op id** 或 **约定名字** 一致（首版仅 `Gather`）。
- 可选：`HIPDNN_EP_DEBUG_CPU_FALLBACK=1` 作总开关。

**`model.dll` 内读环境变量**：使用 **`hip_get_env` / `GetEnvironmentVariableA`**（静态 CRT 下勿用 `std::getenv`）。参见 `include/hip/debug_log.h` 既有模式。

---

## 3. 分阶段交付（仅 Phase 编号）

### Phase 0 — 构建与编译

| 项 | 建议 |
|----|------|
| CMake | `HIPDNN_EP_DEBUG_CPU_FALLBACK`，默认 **OFF**；CI/Release 配置保持 OFF。 |
| EP | ON 时链接现有 ORT C++ API；编译 Gate + Manager。 |
| Runtime / bitcode | ON 时在 `lib/Runtime` 相关 `.cpp` 启用分支；OFF 时回调恒 null、无额外符号依赖。 |

### Phase 1 — C ABI 与 `RuntimeState`

1. 纯 C **描述符**：`op_id`、输入/输出个数、各 tensor 的 `elem_type`、`rank`、`shape`、`host_ptr`（由 `wrap_*` 在 D2H 后填写）；attrs 首版可极简或仅支持 Gather 所需字段。
2. `RuntimeState`（`runtime_state_internal.h`）增加：`cpu_fallback_user`、`cpu_fallback_invoke`（及可选 log 指针）。
3. 导出：`hipdnn_ep_set_cpu_fallback(RuntimeState *, const iface *, void *user)`；`initialize_state` 置空。
4. `GenerateInterface.cpp` / 导出列表：与 `hipdnn_ep_runtime_begin_compute` 等一致，保证 **model.dll 导出**。

### Phase 2 — EP 侧 CPUGate + `Ort::Op`

1. 参考 `3rd-party/Quark/.../execution_provider_cpugate.cpp`：`Manager`、Gate `CustomOp`、后台线程 `Session::Run` 阻塞以固定 **`OrtKernelContext*`**。
2. **不要**依赖 `onnxruntime_providers_ryzenai.dll`；Gate 在 **本 EP 工程** 内注册。
3. `Invoke` 全程 **互斥**，与 Quark `InvokeOrtOp` 一致。
4. **`Ort::Op` 缓存**：按 `(domain, op_type, opset, type_constraints, attrs)` 懒创建。
5. **InferenceState**：`inference_init` 成功后 `set_cpu_fallback`；析构时停线程并清空回调。

### Phase 3 — 首个垂直切片（算子已定：**Gather**）

- **仅** `wrap_gather`：在宏 + 环境变量命中时走 `invoke`；否则原 GPU kernel。
- 流程：`hipStreamSynchronize` → pinned host 缓冲（可 grow-only）→ `hipMemcpy` D2H → 填 `desc`（`op_id = Gather`）→ `invoke` → H2D。
- EP 侧：`Ort::Op::Create` 对应 **`Gather`**（domain 空串、opset 与模型一致），attrs 与 ONNX 对齐（如 axis）。
- 验收：同一输入下，与 **Python 单算子 ORT CPU** 或 **小 ONNX Session** 在约定阈值内一致（cosine / max_abs）。

### Phase 4 — 扩展

- 维护 **`wrap_*` → op_id → ONNX 名/domain/opset/attrs** 表；按需增加 `Where`、`LayerNormalization` 等。
- 每增一算子：**小测试**（pytest 或 LIT 旁路）防回归。
- `Invoke` / `OpInvoker::Create` 失败：**`LOG(ERROR)` + 向 runtime 返回非 0**，由 `wrap_*` 回退 GPU；**勿**在回调路径上 `LOG(FATAL)`（避免在外层 ORT `Run` 栈内 abort 整进程）。

### Phase 5 — 文档与规范

- 本文档随实现迭代更新（**算子表、变量名、宏名** 以代码为准）。
- `CLAUDE.md` 保留 **一行索引**：指向本文 + 强调 **仅调试、非 Release 特性**。

---

## 4. PR 建议切分

| PR | 内容 |
|----|------|
| PR1 | CMake 宏 + `RuntimeState` + C ABI + `hipdnn_ep_set_cpu_fallback` + EP 注册 **空回调**（仅日志验证）。 |
| PR2 | EP 内 CPUGate + **无 `wrap_*`** 的 `Ort::Op::Invoke` 冒烟。 |
| PR3 | **`wrap_gather` 端到端** + 验收用最小对比。 |
| PR4 | 文档修订 + 按需扩算子。 |

---

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| ORT 升级导致 `Invoke`/Context 行为变化 | Debug CI 或本地脚本保留 **最小 Invoke 测试**。 |
| Gate 依赖 ORT 实现细节 | 文档写明；仅 Debug 使用；升级 ORT 后重做冒烟。 |
| fp16 GPU vs fp32 CPU | 文档约定 **阈值**；可选后续：fallback 路径 **promote 到 fp32** 再比。 |
| **外层 ORT `Session::Run` 内嵌套内层 `Session::Run`（OpInvoker）** | 不同 `InferenceSession` 实例；若遇锁/线程池异常，fallback 路径 **仅打日志并返回错误**，由 `wrap_*` 回退 GPU。**禁止**在 EP 回调里使用 `CHECK`/`LOG(FATAL)`（会整进程 abort）。 |
| **`HipdnnCpuFbGatherDesc` 与 `sizeof` 校验失败** | 扩展描述符后须 **重编 EP + 重编 model.dll**（删 `%TEMP%\morphizen_mlir_*` 缓存）；旧 DLL 与 EP 混用会 `detail_size` 不匹配 → 跳过 CPU 路径并打日志。 |
| **`morphizen::OpInvoker::Create` + `Ort::OpAttr` 与 MorphiZen ONNX 栈不兼容** | `OpInvoker` 将 `Ort::OpAttr*` 当作 `AttributeProto*` 克隆；与当前 ORT C++ `Ort::OpAttr` 布局不一致时会污染属性并在 MLIR 里触发 **`llvm::cast` 断言**（`Casting.h`）。**不要**在 EP 里用该路径。 |
| **MorphiZen MLIR 后端：`model_proto_serialize_as_string` ≠ ONNX protobuf** | MLIR 实现把内部模型序列化给自有用途；**不能**交给 `Ort::Session` 解析（典型报错：`protobuf parsing failed`）。Gather CPU 路径改为 **离线 `.onnx` + 构建时嵌入**（`custom-op-mlir/onnx/` + `embed_binary_c_array.py`），当前仅 **axis=0、fp16 data**；indices 为 **int32 或 int64**（int32 在 EP 内 **零时扩成 int64** 再喂 ONNX）。其它组合需增资产与 CMake。 |
| **ORT `IoBinding` + 外部 `CreateTensor` 仍报 Gather indices 伪越界** | 与 `GetInputNames` 顺序无关时，改用 **`Session::Run(..., Ort::Value *output_values)`** 预分配输出路径；int64 索引再 **`memcpy` 到独立缓冲** 后建 `Ort::Value`（见 `GatherOnnxCpuSession::invoke` / `invoke_gather_cpu`）。 |
| **ORT `GetInputNames()` 下标顺序 ≠ Gather 语义** | 不再依赖 `GetInputNames()[i]` 与 `input_values[i]` 配对；`Run` 仍使用固定名 `Input_0`/`Input_1`/`Output_0`。 |
| **ORT Gather 输出 rank 与 MLIR memref rank 不一致** | ONNX 输出形状 = `indices.shape` 与 `data.shape[axis+1:]` 拼接，rank = `indices_rank + (data_rank - 1)`；MLIR 可能给出 **同元素数的低 rank**（如 `[1,4096]` vs `[1,1,4096]`）。EP 在 `invoke_gather_cpu` 中 **优先用 ONNX 公式形状** 建 `Ort::Value`，否则 IoBinding 下可出现 **indices 越界类误报**。 |
| **`hip-onnx-runner` 对 INT64 `input_ids` 曾用全 `uint64` 随机** | 索引几乎总超出词表；ORT CPU Gather **硬失败**，GPU `gather_kernel` **静默钳位到 0**，易误判为 EP bug。Runner 现对 `input_ids`/`*input_ids` 使用 **`[0, --token-vocab)`**（默认 32000）；`--token-vocab 0` 恢复旧行为。 |
| **形状未物化（负维度）或 shape 乘积 ≠ 元素个数** | `wrap_gather` 遇负维跳过 D2H；EP 侧校验 `product(shape)==*_num_elements`，不一致则 `LOG(ERROR)` 并返回，避免 `CreateTensor` / 分配越界。 |

---

## 6. 与既有讨论的一致性摘要

- **CPUGate / ORT 不进 `model.dll`**；仅 **`wrap_*` 内 env/宏 + 回调**。
- **正式产品**：全 GPU；本特性 **默认不编进 Release**。
- **Phase 3 首个算子已定为 `Gather`**（`wrap_gather` ↔ ONNX `Gather`）。

---

## 修订历史

| 日期 | 说明 |
|------|------|
| 2026-06-12 | 初稿：根据设计讨论整理成可执行计划。 |
| 2026-06-12 | Gather 回调：描述符增加元素计数字段 + EP 侧形状校验；移除 `CHECK`/`FATAL` 以免调试路径 abort；文档补充嵌套 Run / ABI 说明。 |
| 2026-06-15 | Gather CPU 路径不再使用 `OpInvoker`+`Ort::OpAttr`（与 `AttributeProto` 误 reinterpret 导致 MLIR `cast` 断言）；改为 `NodeAttributesBuilder` 建 `axis` + 本地 `Ort::Session`。 |
| 2026-06-15 | MLIR 下 `model_proto_serialize_as_string` 非 ONNX：Gather CPU 改为 **嵌入离线 ONNX**（`onnx/gather_cpu_fb_axis0_fp16_i64.onnx` + CMake 生成 **`gather_cpu_fb_onnx_data.inc`**，`cpu_fallback_bridge.cpp` 单 TU `#include`，`static` 字节数组，避免单独生成 `.cpp` 未进静态库时的 **`LNK2019`**；`add_custom_target`+`add_dependencies` 保证先跑嵌入脚本）。 |
| 2026-06-15 | Gather EP：`HipdnnCpuFbGatherDesc` 增加 **`indices_element_size_bytes`**；`invoke_gather_cpu` 对 **int32 索引扩 int64**、对输出使用 **ONNX 公式 rank/shape**（修复 ORT `indices out of bounds` 误报 + int32 路径）。 |
| 2026-06-15 | Gather CPU：`GatherOnnxCpuSession::invoke` 用 **`Session::Run`（预分配 `Ort::Value` 输出）** 替代 `IoBinding`；部分 ORT 下 IoBinding + 外部 `CreateTensor` 仍会出现 **伪 indices 越界**（与 `GetInputNames` 顺序无关）。int64 索引进 EP 前 **`memcpy` 到独立 `vector<int64_t>`** 再建张量。 |
| 2026-06-15 | `hip-onnx-runner`：`input_ids` 随机改为 **`[0, --token-vocab)`**（默认 32000），避免 Gather CPU fallback 与 GPU「越界写零」语义不一致导致的假阳性。 |
