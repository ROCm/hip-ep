# Archive

This folder contains historical implementation summaries and notes that provide context about specific changes to the codebase.

These documents are preserved for historical reference but are not part of the main documentation set.

## Contents

- **IMPLEMENTATION_SUMMARY_TIMEOUT.md** - Implementation details for GPU timeout protection feature (2026-01-17)
  - Background on PID 11940 incident
  - WMI termination workaround
  - Complete file-by-file implementation details
  - For current documentation, see `../05_GPU_TIMEOUT_HANDLING.md`

- **SINGLETON_DESIGN_HISTORY.md** - Historical rationale for singleton HipContext design (2026-01-18)
  - Original benefits of singleton pattern
  - Problems with parallel sessions
  - Why the design was replaced with per-session contexts
  - For current documentation, see `../08_ROCM_RESOURCE_MANAGEMENT.md`
