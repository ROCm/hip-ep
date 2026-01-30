<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# clean up provider options

# sources of provider options.

## explicitly set by end users

```c++
auto session_options = Ort::SessionOptions();
auto provider_options = std::unordered_map<std::string, std::string>{{"cache_key", "a-sample-cache-key"}};
session_options.AppendExecutionProvider_VitisAI(options);
```

it is saved, and now it is supported by PR #209

## set by config file

```c++
auto session_options = Ort::SessionOptions();
auto provider_options = std::unordered_map<std::string, std::string>{{"config", "vaip_config.json"}};
session_options.AppendExecutionProvider_VitisAI(options);
```

the content of `vaip_config.json`

```json
{
    "provider_options" : {
        "cache_dir" : "sample_cache_dir",
        "cache_key" : "sample_cache_key"
    }
}
```

## read from `context.json`

it is removed. see jira:VAI-9685. and [pass_context_imp.cpp#L1177][s1]

## set when MEP table hit

[see here](../morphizen-core/src/morphizen_compile_model.cpp#L502)

it is to be deprecated, see PR #159 #155

* `model_name`
* `model_category`
* `model_variant`
* `is_preemptible`
* `dd_use_lazy_scratch_bo`
* `qos_priority`

# set when TargetProto hit

[see here]()../morphizen-core/src/config.cpp#L307-L308)

It is to be deprecated, see PR #159 #155

* `xlnx_enable_py3_round`
* `xlnx_enable_old_qdq`
* `xclbin`


[s1]: ../morphizen-core/src/pass_context_imp.cpp#L1177
