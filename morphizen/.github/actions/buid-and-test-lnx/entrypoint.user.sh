##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

printenv
cat <<EOF
===================================================
As $HOST_USER_NAME, start running [$@]
===================================================
EOF

eval "$@"
