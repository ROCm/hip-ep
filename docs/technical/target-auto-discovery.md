<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

EP auto target discovery with the following priorities.

1.# auto target discovery

EP auto target discovery with the following priorities.

1. `provider_options["target"]` set explicitly by users.
    - It is a fatal error if target is not valid, a list of valid target names are printed and abort abnormally, because end users are expected to know the target they want to use.


2. `target` specified in MEP table
    - It is a fatal error if target is not valid, a list of valid target names are printed and abort abnormally, because the config file is inconsistent, no matter it is the cofnig file provided by the user or the built-in config file.

3. auto target discovery
    - when the build-in config file is used, the plugin must return a valid target name, otherwise it is a fatal error, because it means that the source code is not consistent with the built-in config file, and the built-in config file is regarded as the part source code.

4. `target` in the json config file.
    - It is a fatal error if target is not valid, a list of valid target names are printed and abort abnormally, because end users are expected to know the target they want to use.
    - `target` is now considered as a mandatory field in the default morphizen config, and it must be set to a valid target name.
