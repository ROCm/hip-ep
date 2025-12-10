<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Integration of ORT Log/Trace System

## Introduction

`glog` was used for a long time; however, it has some disadvantages:

1. When the user application is a Windows service, usually `stderr`/`stdout` is not available, making it difficult to debug in the production environment.
2. `glog` is not well integrated with the ETW ecosystem.
3. Users cannot control the log level using the standard ORT APIs.

## How ORT logging system works
