#!/bin/bash
# Credential helper for Bazel to access private GitHub registries.
# Bridges gh auth token to Bazel's credential helper protocol.
#
# Usage: Bazel calls this script with URI as argument, expects output in format:
#   headers=Authorization=Bearer <token>

set -euo pipefail

# Get GitHub token from gh CLI
TOKEN=$(gh auth token 2>/dev/null || echo "")

if [ -z "$TOKEN" ]; then
  >&2 echo "Error: gh auth token failed. Run 'gh auth login' first."
  exit 1
fi

# Output in Bazel credential helper format
echo "headers=Authorization=Bearer $TOKEN"
