#!/bin/bash
adduser --disabled-password --uid $HOST_USER_ID --home /github/workspace --shell /bin/bash --no-create-home --gecos "" $HOST_USER_NAME
printenv
chmod +x /entrypoint.user.sh
sudo --preserve-env -u "$HOST_USER_NAME" env HOME=/github/workspace PATH=$PATH:/github/workspace/.local/bin:/workspace/local/bin /bin/bash /entrypoint.user.sh "$@"
