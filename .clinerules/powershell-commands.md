<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
## PowerShell

- Do NOT use `&&` - use `;` to chain: `cd build; cmake ..`
- Append `; echo "=== DONE ==="` to detect command completion
- Use `; if ($?) { echo "=== OK ===" } else { echo "=== FAIL ===" }` for status

### If command appears hung:
- Check VS Code terminal directly
- Ctrl+C to cancel, then inform Cline
