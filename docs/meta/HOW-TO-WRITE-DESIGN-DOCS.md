<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Design Document Writing Rules

**Date:** 2026-06-13
**Document Type:** Guide
**Status:** Draft
**Audience:** Contributors and automated agents writing docs under `docs/`

---

## Core Rules

1. **No source code file paths** - Reference abstractions only (`RuntimeState`, `inference_init`, `@main_graph`), never files (`lib/Foo.cpp:123`)
2. **No duplication** - Before writing, search `docs/` for existing content. Link instead of copy/paste.
3. **Use hyperlinks extensively** - Reference other docs instead of repeating information. Every time you mention a concept documented elsewhere, add a hyperlink. Link to related docs in "Related Documents" section with purpose statement.
4. **Visual diagram if helpful** - Use for >2 components or non-trivial flow
5. **Concrete metrics only** - If you have metrics from code/benchmarks, use them. Never guess numbers.
6. **No fluff** - Delete adjectives and vague claims
7. **No marketing language** - Delete: elegant, powerful, robust, seamless, comprehensive, etc.
8. **No bureaucracy** - Delete: purpose statements, background sections, scope sections if obvious
9. **Minimal speculation** - Focus on current design. Only add "Future Work" if user requested or doc needs it.
10. **No discussion history** - Don't record committee discussions unless user explicitly requests it
11. **Alternatives only if needed** - Don't compare with alternative solutions unless user requests it or decision is non-obvious

---

## Document Types

```
Major architectural change → Architecture
Component/feature design → Design
Pass/algorithm spec → Implementation
Analysis/comparison → Tech Note
```

---

## Required Header

```markdown
**Date:** YYYY-MM-DD
**Document Type:** [Architecture|Design|Implementation|Tech Note]
**Status:** Draft
**Related:** [link1], [link2]
```

---

## Templates

### Architecture
```markdown
## Overview
[Problem - 2-3 sentences]

## Architecture
[Diagram if helpful]
[Component responsibilities]

## Related Documents
- [doc.md](doc.md) - [purpose]
```

### Design
```markdown
## Overview
[Problem - 2-3 sentences]

## Design
[Diagram if helpful]
[Key abstractions]

## Related Documents
- [doc.md](doc.md) - [purpose]
```

### Implementation
```markdown
## Overview
[What this implements - link to design]

## Implementation
[Algorithm, data structures, edge cases]

## Testing
[Strategy, coverage]

## Related Documents
- [design-doc.md](design-doc.md) - Design spec
```

### Tech Note
```markdown
## Overview
[Question/analysis scope]

## Analysis
[Findings, data, tables]

## Conclusion
[Recommendations]

## Related Documents
- [doc.md](doc.md) - [purpose]
```

---

## Anti-Patterns (Delete These)

### Fluff
❌ "This elegant solution..."
❌ "The design provides a flexible framework..."
❌ "To ensure optimal performance..."
❌ "Fast compilation" (unless you have benchmark data)
✅ "RuntimeState stores device handles."
✅ "Compilation takes 0.8s" (if you measured it)

### Marketing Language
❌ powerful, robust, seamless, comprehensive, elegant, sophisticated, cutting-edge
✅ Use technical terms only

### Bureaucracy
❌ "The purpose of this document is to..."
❌ "This section provides background on..."
❌ "Scope: This design covers X, Y, Z..." (unless scope is non-obvious)
✅ Start with the problem in Overview

### Unnecessary Future Work
❌ "In the future we might add caching..." (not planned)
❌ "This could be extended to support X..." (speculation)
✅ "Future Work: Dynamic shapes not yet supported" (known limitation)
✅ "Open Question: Should constants be mutable?" (needs decision)

### Vague Metrics
❌ "better performance", "faster", "more maintainable"
❌ "reduces memory by 40%" (unless you measured it)
✅ "reduces LOC from 500 to 200" (if you counted)
✅ "eliminates 3 redundant classes" (if you verified)
✅ "avoids memory copy" (factual, no numbers claimed)

### Duplication
❌ Copying the constant upload algorithm from [constant-handling-design.md](../design/constant-handling-design.md)
✅ "For constant handling: [constant-handling-design.md](../design/constant-handling-design.md)"
❌ Repeating the memory pooling offsets algorithm
✅ "Memory pooling computes a count + offsets consumed by generated init. See [pool-allocs-memory-planning.md](../design/pool-allocs-memory-planning.md)"

---

## Validation

**Before presenting to user:**
```
[ ] Header present (Date, Type, Status, Related)
[ ] No file paths (only abstractions)
[ ] Diagram if helpful (>2 components or complex flow)
[ ] No vague claims (concrete metrics only if measured)
[ ] Links have purpose statements
[ ] Hyperlinks used extensively (every concept links to its doc)
[ ] TOC if >100 lines
[ ] No marketing words (powerful, elegant, robust, seamless, comprehensive)
[ ] Searched docs/ for duplication - linked instead of copying
[ ] No discussion history unless user requested
[ ] No alternatives comparison unless user requested
```

---

## Execution

### Write new doc:
1. **Search docs/ first** - Check for existing content to link (no duplication)
2. Select type (Architecture/Design/Implementation/Tech Note)
3. Copy template
4. Fill sections (no marketing words, no vague claims)
5. Add diagram if helpful
6. **Delete fluff** - Remove bureaucracy, marketing language
7. **Add hyperlinks** - Replace any copied content with links
8. **Don't add**: discussion history, alternatives comparison (unless user requested)
9. Validate checklist
10. Set Status: Draft

### Review doc:
1. Run validation checklist
2. Check each anti-pattern section - flag violations
3. Search docs/ for duplicated content
4. Suggest fixes using templates

---

## Examples

| Type | Reference | Pattern |
|------|-----------|---------|
| Design | [constant-handling-design.md](../design/constant-handling-design.md) | Compiler-supplied attribute consumed by generated init |
| Design | [output-allocator-design.md](../design/output-allocator-design.md) | Separation of concerns + diagrams |
| Design | [op-state-slots-design.md](../design/op-state-slots-design.md) | Compile-time identity vs runtime construction |
| Design | [pool-allocs-memory-planning.md](../design/pool-allocs-memory-planning.md) | Pass computes count + offsets; init consumes |
| Reference | [pipeline_pass_menu.md](../pipeline_pass_menu.md) | Pass names, anchors, and built-in order |

---

**Rule:** Follow templates exactly. No fluff, no bureaucracy.
