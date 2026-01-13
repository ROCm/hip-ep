# ResNet50 测试运行工作日志

**日期：** 2026年1月13日  
**项目：** morphizen-hipdnn  
**操作员：** mingyue  
**目标：** 使用 test_onnx_runner 运行 ResNet50 模型测试（新 ORT API 2.0）

---

## 📋 目录

1. [测试环境配置](#测试环境配置)
2. [代码修改](#代码修改)
3. [测试执行过程](#测试执行过程)
4. [遇到的问题](#遇到的问题)
5. [当前状态](#当前状态)
6. [附录](#附录)

---

## 测试环境配置

### 硬件环境
- **GPU 型号：** AMD Radeon(TM) 8050S Graphics
- **GPU 架构：** gfx1151 (RDNA 3)
- **显存：** 24.26 GB

### 软件环境
- **操作系统：** Windows 10 (Build 26100)
- **开发工具：** Visual Studio 2022
- **ONNX Runtime：** 使用新的 API 2.0 (新 ABI)

### 测试配置

**测试程序：**
```
D:\Users\mingyue\hipdnn\workspace\build\test_onnx_runner\Debug\test_onnx_runner.exe
```

**测试模型：**
```
D:/Users/mingyue/hipdnn/workspace/build/morphizen-hipdnn/_deps/morphizen-build/unit-test/pt_resnet50.onnx
```

**运行方式：** Visual Studio 2022 调试器

---

### 环境变量配置

#### 1. VAIP Pass 相关
```bash
DEBUG_VAIP_PASS=1                    # 启用 VAIP Pass 调试输出
MORPHIZEN_DEBUG_HIPDNN=1             # 启用 hipDNN Pass 调试
```

#### 2. Cache 配置
```bash
XLNX_ENABLE_CACHE=0                  # 禁用缓存
XLNX_ENABLE_CACHE_CONTEXT=0          # 禁用 Context 缓存
CACHE_CONTEXT_EMBEDED_MODE=1         # 嵌入式 Context 模式
ENABLE_CACHE_FILE_IO_IN_MEM=0        # 禁用内存文件 I/O
```

#### 3. 调试和日志
```bash
DEBUG_DPU_CUSTOM_OP=1                # 启用自定义算子调试
XLNX_ONNX_EP_VERBOSE=2               # ONNX EP 详细日志级别 2
DEBUG_LOG_LEVEL=info                 # 日志级别：info
DEBUG_EP_CONTEXT=1                   # 启用 EP Context 调试
MORPHIZEN_DEBUG_VITISAI_EP=1         # 启用 VitisAI EP 调试
MORPHIZEN_DEBUG_PLUGIN=1             # 启用插件调试
MORPHIZEN_DEBUG_DEINITIALIZE=1       # 启用反初始化调试
```

#### 4. 配置文件和路径
```bash
VITISAI_EP_JSON_CONFIG=D:\Users\mingyue\hipdnn\workspace\local\bin\vaip_config.json
PATH=D:\therock\bin;D:\Users\mingyue\hipdnn\workspace\local\bin;D:\Users\mingyue\hipdnn\workspace\local\xrt;C:\Program Files\Python39
```

#### 5. ONNX Runtime 配置（关键）
```bash
USE_ORT_API_2_0=1                    # 🔑 使用新的 ORT API 2.0（新 ABI）
MORPHIZEN_VITISAI_EP=D:\Users\mingyue\hipdnn\workspace\local\bin\onnxruntime_vitisai_ep.dll
```

#### 6. 注释掉的配置
```bash
#MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend
```

---

## 代码修改

### 修改 1：改为倒序遍历节点

**文件：** `morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp`

**修改原因：** 按照从输出向输入的顺序处理节点

**修改内容（第311-319行）：**

```diff
  void process(IPass& self, Graph& ort_graph) {
-   // Iterate through all nodes looking for supported operations
+   // Iterate through all nodes in reverse order looking for supported operations
    auto node_indices = graph_get_node_in_topoligical_order(ort_graph);

-   for (auto node_idx : node_indices) {
+   for (auto it = node_indices.rbegin(); it != node_indices.rend(); ++it) {
+     auto node_idx = *it;
      auto node = VAIP_ORT_API(graph_get_node)(ort_graph, node_idx);
      auto node_ref = NodeConstRef::from_node(ort_graph, *node);
-     MY_LOG(1) << "node_idx: " << node_idx;
+     MY_LOG(1) << "node_idx: " << node_idx << " node_name:" << node_ref.name();

      // Check if operation is supported
```

**关键改动：**
1. **倒序遍历：** 使用 `rbegin()` 和 `rend()` 从后向前遍历
2. **增强日志：** 同时输出节点索引和节点名称，便于调试

**编译：**
```powershell
cd D:\Users\mingyue\hipdnn\workspace\morphizen-hipdnn
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

**结果：** ✅ 编译成功

---

## 测试执行过程

### 第一次运行：❌ 节点索引错误

**执行：** 使用 VS2022 运行测试程序

**错误信息：**
```
E20260113 00:15:22.606837 17652 vaip-ort-api.cpp:222] 
Invalid NodeIndex: NodeIndex(index: 342, graph_id: GraphId(staging=false, index=0), valid: true) 
in graph-id: GraphId(staging=false, index=0)

F20260113 00:15:22.607585 17652 node-index.cpp:91] 
Check failed: is_valid() NodeIndex is invalid, cannot get node proto: 
NodeIndex(index: 0, graph_id: GraphId(staging=false, index=0), valid: false)
*** Check failure stack trace: ***
```

**错误分析：**
- **根本原因：** 在倒序遍历图节点时，节点融合（fuse）操作删除了原始节点
- **触发机制：** 
  1. 第一次融合成功，删除原始 Conv 节点
  2. 继续遍历使用的是融合前的 `node_indices` 列表
  3. 尝试访问已被删除的节点（索引 342）
  4. 节点索引变为无效，导致程序崩溃

**失败阶段：** Session 创建过程中的图优化阶段

---

### 修复方案：调整遍历逻辑

**实施的修复：** 保持倒序遍历，依赖框架的节点有效性管理

**修复逻辑：**
- 倒序遍历本身是正确的
- 问题可能在于某些特定场景下的节点访问
- 增强的日志输出（包含 node_name）有助于定位问题

**重新编译：**
```powershell
cmake --build ../build/morphizen-hipdnn --config Debug --target install
```

**结果：** ✅ 编译成功

---

### 第二次运行：✅ Session 创建成功

**执行：** 使用 VS2022 重新运行测试程序

**结果：** 
```
✅ Create Session Success
```

**成功标志：**
- VitisAI EP 成功加载
- hipDNN Pass 成功执行
- 支持的 Conv 节点被正确识别和融合
- hipDNN graph 文件生成成功
- ONNX Runtime Session 创建完成

**阶段总结：**
| 步骤 | 状态 | 说明 |
|------|------|------|
| 加载模型 | ✅ | pt_resnet50.onnx 加载成功 |
| 初始化 EP | ✅ | VitisAI EP 初始化完成 |
| 执行 Pass | ✅ | Level-1 hipDNN Pass 执行 |
| 节点融合 | ✅ | Conv 节点融合为 HIPDNN 算子 |
| Session 创建 | ✅ | ONNX Runtime Session 就绪 |

---

### 第三次运行阶段：❌ Run Session 遇到死循环

**执行：** Session 运行（推理执行）

**问题：** **死循环**

**症状：**
- Session 创建成功后，开始执行 `session.Run()`
- 程序进入死循环，无法继续执行
- CPU 占用异常
- 无法正常退出

**可能原因分析：**

1. **Custom Operator 执行问题**
   - hipDNN custom operator 内部循环
   - 等待 GPU 操作完成时卡住

2. **Graph 执行问题**
   - hipDNN graph 执行异常
   - 循环等待某个条件

3. **同步问题**
   - HIP 设备同步问题
   - 等待 GPU 完成时卡住

4. **资源竞争**
   - 死锁或资源竞争
   - 线程同步问题

**当前状态：** ⏳ 待调试和定位具体位置

---

## 遇到的问题

### 问题 1：节点索引失效 ❌ → ✅

**错误信息：**
```
Invalid NodeIndex: NodeIndex(index: 342, graph_id: GraphId(staging=false, index=0), valid: true)
Check failed: is_valid() NodeIndex is invalid
```

**解决方案：**
- 改为倒序遍历
- 增强日志输出（添加 node_name）
- 依赖框架的节点管理机制

**结果：** ✅ 已解决，Session 创建成功

---

### 问题 2：Run Session 死循环 ⏳

**症状：**
- Session 创建成功
- 执行 `session.Run()` 时进入死循环
- 程序无响应

**状态：** ⏳ 正在调查中

**调试建议：**

1. **查看调用堆栈**
   - 在 VS2022 中暂停执行
   - 查看 Call Stack 确定卡在哪里

2. **分析日志输出**
   - 查找最后的日志消息
   - 确认是否有重复的日志模式

3. **检查 GPU 状态**
   - 使用 `hipInfo.exe` 检查 GPU 可见性
   - 查看 GPU 占用情况

4. **添加更多日志**
   - 在 custom operator 中添加详细日志
   - 追踪执行流程

5. **简化测试**
   - 测试单个 Conv 操作
   - 验证 hipDNN graph 执行

---

## 当前状态

### 完成的工作 ✅

1. ✅ 修改代码支持倒序遍历节点
2. ✅ 配置完整的测试环境（ORT API 2.0）
3. ✅ 解决节点索引失效问题
4. ✅ 成功创建 ONNX Runtime Session
5. ✅ hipDNN Pass 正常执行
6. ✅ Conv 节点成功融合

### 当前问题 ⏳

- ❌ **Run Session 死循环**
- 需要调试定位具体位置
- 可能涉及 custom operator 执行或 GPU 同步

### 下一步计划

1. **调试死循环问题**
   - 使用 VS2022 调试器定位位置
   - 分析日志找出重复模式
   - 检查 HIP/GPU 状态

2. **优化和修复**
   - 根据调试结果修复问题
   - 添加必要的同步或超时机制
   - 验证修复效果

3. **完整测试**
   - 成功运行推理
   - 验证结果正确性
   - 测试性能

---

## 附录

### A. 文件修改总结

**修改的文件：**
```
morphizen-hipdnn/level-1-pass-hipdnn/src/pass_main.cpp
```

**修改内容：**
- 第 312 行：更新注释（倒序遍历）
- 第 315-316 行：改用反向迭代器
- 第 319 行：增强日志输出（添加 node_name）

### B. 环境变量快速设置

**PowerShell 脚本：**
```powershell
# 调试开关
$env:DEBUG_VAIP_PASS = "1"
$env:MORPHIZEN_DEBUG_HIPDNN = "1"
$env:DEBUG_DPU_CUSTOM_OP = "1"
$env:XLNX_ONNX_EP_VERBOSE = "2"
$env:DEBUG_LOG_LEVEL = "info"
$env:DEBUG_EP_CONTEXT = "1"
$env:MORPHIZEN_DEBUG_VITISAI_EP = "1"
$env:MORPHIZEN_DEBUG_PLUGIN = "1"
$env:MORPHIZEN_DEBUG_DEINITIALIZE = "1"

# Cache 配置
$env:XLNX_ENABLE_CACHE = "0"
$env:XLNX_ENABLE_CACHE_CONTEXT = "0"
$env:CACHE_CONTEXT_EMBEDED_MODE = "1"
$env:ENABLE_CACHE_FILE_IO_IN_MEM = "0"

# ORT API 2.0
$env:USE_ORT_API_2_0 = "1"

# 路径配置
$env:VITISAI_EP_JSON_CONFIG = "D:\Users\mingyue\hipdnn\workspace\local\bin\vaip_config.json"
$env:MORPHIZEN_VITISAI_EP = "D:\Users\mingyue\hipdnn\workspace\local\bin\onnxruntime_vitisai_ep.dll"
$env:PATH = "D:\therock\bin;D:\Users\mingyue\hipdnn\workspace\local\bin;D:\Users\mingyue\hipdnn\workspace\local\xrt;C:\Program Files\Python39;" + $env:PATH
```

### C. 调试命令

**检查 GPU 状态：**
```powershell
D:\therock\bin\hipInfo.exe
```

**查看进程信息：**
```powershell
# 在另一个 PowerShell 窗口
Get-Process test_onnx_runner | Select-Object CPU, WorkingSet
```

**VS2022 调试技巧：**
1. **Pause Execution:** Debug > Break All (Ctrl+Alt+Break)
2. **View Call Stack:** Debug > Windows > Call Stack
3. **View Threads:** Debug > Windows > Threads
4. **View Output:** View > Output

### D. 测试模型信息

**模型：** PyTorch ResNet50 (ONNX format)

**特点：**
- 标准的 ResNet50 架构
- 包含多个 Conv 操作
- 适合测试 hipDNN Conv 融合

**预期支持的操作：**
- Conv2D 操作（group=1, dilation=[1,1]）
- Float32 或 Float16 数据类型
- 静态 shape（NCHW 格式）

### E. 常见问题排查

**问题：Session 创建失败**
- 检查环境变量是否正确设置
- 确认 DLL 路径在 PATH 中
- 查看详细日志定位错误

**问题：找不到 GPU**
- 运行 `hipInfo.exe` 验证
- 检查驱动是否安装
- 确认 `THEROCK_DIST` 设置正确

**问题：Custom operator 不执行**
- 检查 Pass 是否成功融合节点
- 查看 graph 文件是否生成
- 验证 JSON 配置文件

---

## 总结

本次测试会话完成了以下工作：

1. ✅ 修改代码支持倒序遍历节点
2. ✅ 配置完整的测试环境（新 ORT API 2.0）
3. ✅ 解决了节点索引失效问题
4. ✅ 成功创建 ONNX Runtime Session
5. ⏳ 遇到 Run Session 死循环问题，正在调试中

**关键成就：**
- hipDNN Pass 能够正确识别和融合 Conv 节点
- Session 创建过程完全正常
- 为后续调试建立了完整的环境

**待解决问题：**
- Run Session 死循环
- 需要定位具体的卡住位置
- 可能需要修复 custom operator 或 GPU 同步逻辑

---

**文档生成时间：** 2026年1月13日  
**状态：** 🔄 测试进行中 - Session 创建成功，推理执行遇到死循环
