<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Integration of ORT Log/Trace System

## Introduction

MorphiZen has transitioned from `glog` to the native ONNX Runtime (ORT) logging system to address several limitations:

1. **Windows Service Compatibility**: When the user application runs as a Windows service, `stderr`/`stdout` are typically not available, making debugging difficult in production environments.
2. **ETW Integration**: `glog` does not integrate well with Event Tracing for Windows (ETW), limiting diagnostics on Windows platforms.
3. **API Control**: With `glog`, users could not control log levels using standard ORT APIs, requiring separate configuration.

The ORT native logging system solves these issues by providing:
- ETW support on Windows for production debugging
- Unified API for log level control
- Better integration with ORT internals

## How ORT Logging System Works

The ONNX Runtime logging system provides a flexible, cross-platform logging infrastructure:

### Architecture

1. **Logger Interface**: ORT uses a `Logger` interface that execution providers can access via the `Env` object
2. **Log Levels**: Supports standard severity levels (VERBOSE, INFO, WARNING, ERROR, FATAL)
3. **Log Sinks**: Can output to console, files, or platform-specific sinks (ETW on Windows)
4. **Session-Specific Logging**: Each `InferenceSession` can have its own logger with custom settings

### How MorphiZen Uses ORT Logging

MorphiZen execution provider integrates with ORT logging through:

```cpp
// Access the ORT logger
const auto& logger = session.GetLogger();

// Log messages at different levels
LOGS(logger, VERBOSE) << "Detailed debug information";
LOGS(logger, INFO) << "Informational message";
LOGS(logger, WARNING) << "Warning message";
LOGS(logger, ERROR) << "Error occurred";
```

The `LOGS` macro is provided by ORT and automatically handles:
- Filtering based on configured log level
- Formatting with file/line information
- Routing to appropriate sinks (console, ETW, etc.)

## Enabling ETW Tracing (Windows)

Event Tracing for Windows (ETW) allows capturing logs from production Windows services where console output is unavailable.

### Prerequisites

- Windows 10 or later
- Administrator privileges (for capturing ETW traces)
- Windows Performance Toolkit or similar ETW capture tool

### Enabling ETW in ORT

ETW tracing is built into ONNX Runtime on Windows. To enable it:

1. **Enable logging in SessionOptions**:
   ```cpp
   Ort::SessionOptions session_options;
   session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
   ```

2. **Start ETW capture** (from elevated command prompt):
   ```cmd
   REM Start capturing ETW events from ONNX Runtime
   logman create trace OrtTrace -p "Microsoft-ONNX-Runtime" -o ortlog.etl -ets

   REM Run your application
   your_application.exe

   REM Stop capturing
   logman stop OrtTrace -ets
   ```

3. **View captured logs**:
   ```cmd
   REM Convert ETL to human-readable format
   tracerpt ortlog.etl -o ortlog.xml -of XML

   REM Or view directly with Windows Performance Analyzer
   wpa ortlog.etl
   ```

### ETW Provider Information

- **Provider Name**: `Microsoft-ONNX-Runtime`
- **Provider GUID**: Check ORT source or use `logman query providers` to find
- **Keyword Flags**: Various keywords for filtering (inference, graph optimization, execution provider, etc.)

## Log Levels

ORT supports the following log severity levels:

| Level | Value | Description | Use Case |
|-------|-------|-------------|----------|
| VERBOSE | 0 | Detailed diagnostic information | Development debugging |
| INFO | 1 | General informational messages | Production monitoring |
| WARNING | 2 | Warning conditions that should be reviewed | Production alerting |
| ERROR | 3 | Error conditions that may allow continued execution | Production error tracking |
| FATAL | 4 | Critical errors requiring immediate termination | Crash analysis |

### Configuring Log Levels

Set the log level when creating a session:

```cpp
Ort::SessionOptions session_options;

// Set to INFO level (default for production)
session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);

// Set to VERBOSE for detailed debugging
session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

// Set to WARNING to reduce noise in production
session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);

Ort::Session session(env, model_path, session_options);
```

## Usage Examples

### Basic Logging in MorphiZen Code

```cpp
#include "core/common/logging/logging.h"

void MyExecutionProviderFunction(const onnxruntime::GraphViewer& graph,
                                  const onnxruntime::logging::Logger& logger) {
  // Informational logging
  LOGS(logger, INFO) << "Processing graph with " << graph.NumberOfNodes() << " nodes";

  // Warning logging
  if (node_count > 1000) {
    LOGS(logger, WARNING) << "Large graph detected: " << node_count << " nodes";
  }

  // Error logging
  if (!ValidateGraph(graph)) {
    LOGS(logger, ERROR) << "Graph validation failed";
    return;
  }

  // Verbose/debug logging (only appears when log level is VERBOSE)
  LOGS(logger, VERBOSE) << "Node details: " << node.Name();
}
```

### Conditional Logging

```cpp
// Log only if condition is true
LOGS_IF(logger, INFO, model_loaded) << "Model loaded successfully";

// Log with custom category
LOGS_CATEGORY(logger, INFO, "EP.MorphiZen") << "MorphiZen-specific message";
```

## Relationship to glog

MorphiZen previously used Google's glog library but has migrated to ORT native logging:

### Migration Path

| glog API | ORT Equivalent | Notes |
|----------|----------------|-------|
| `LOG(INFO) << ...` | `LOGS(logger, INFO) << ...` | Requires logger parameter |
| `LOG(WARNING) << ...` | `LOGS(logger, WARNING) << ...` | Logger from session/env |
| `LOG(ERROR) << ...` | `LOGS(logger, ERROR) << ...` | Same severity mapping |
| `DLOG(INFO) << ...` | `LOGS(logger, VERBOSE) << ...` | VERBOSE filtered in release |
| `CHECK(condition)` | Use ORT status/exceptions | Different error handling pattern |

### Why Migrate?

1. **Unified logging**: All ORT components use the same logging system
2. **No external dependency**: glog is no longer needed as a dependency
3. **Better ETW support**: Native ETW integration on Windows
4. **User control**: Applications can configure ORT logging via standard APIs

For legacy glog documentation, see [glog-integration.md](glog-integration.md).

## Platform Differences

### Windows
- **ETW Support**: Full ETW tracing support for production diagnostics
- **Console Output**: Logs to `stdout`/`stderr` when available
- **Windows Event Log**: Can integrate with Windows Event Log (requires custom sink)

### Linux
- **Console Output**: Logs to `stdout`/`stderr`
- **syslog Integration**: Can be configured to output to syslog
- **Systemd Journal**: When running as systemd service, logs appear in journal

### macOS
- **Console Output**: Logs to `stdout`/`stderr`
- **Unified Logging**: Can integrate with macOS Unified Logging system (OSLog)

### Configuration Example (Cross-Platform)

```cpp
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MorphiZenApp");

// On Windows, this enables ETW automatically
// On Linux, logs go to stderr (or syslog if configured)
// On macOS, logs go to stderr (or OSLog if configured)

Ort::SessionOptions session_options;
session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
```

## Best Practices

1. **Use INFO level for production**: Provides useful diagnostics without excessive verbosity
2. **Use VERBOSE level for development**: Helps debug integration issues
3. **Avoid excessive logging in hot paths**: Logging has performance overhead
4. **Include context in messages**: Add model name, node name, or other identifying information
5. **Test ETW capture before deployment** (Windows): Ensure ETW works in your production environment

## Related Documentation

- **[glog Integration](glog-integration.md)**: Legacy glog usage (deprecated)
- **[Developer Guide](../developer-guide.md)**: Development environment setup
- **[Architecture](../architecture.md)**: System design and logging architecture
