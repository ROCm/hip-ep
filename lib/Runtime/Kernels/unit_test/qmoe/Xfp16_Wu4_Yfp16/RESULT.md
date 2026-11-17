# Focused interface and native UT validation

2026-11-16, xco18 / Radeon 8060S / gfx1151, SDK 7p16 flat layout,
runtime/driver 70260201. Built with the owning leaf flags and production sources.
The unchanged MatMulNBits device object was source/ABI/hash checked and reused.
The host wrapper/CPU reference compile as C++; the small stream gate compiles
as HIP, avoiding a Windows HIP/MSVC std::min mangling collision.

- M=1 H=128 I=128 E=4 k=2 gs=32: complete wrap_qmoe decode PASS, relL2=0.
- M=9 same geometry, both biases + packed ZP + external router weights:
  complete grouped-prefill PASS, relL2=0.000283031.
- Both cases use exact agent-aie adapter output, one positive CSV result,
  adaptive-v3 screen timing and CPU-correct UT_TIMING_BINDING.
- Existing agent-aie CSV reader/config parser consumes both actual rows.
- Shared policy/parser checks and eight existing JSON LUT roundtrips PASS.
- MatMulNBits u4 fp16/fp32 route/ZP interface checks PASS (15 bounded cases,
  including expected input rejections and representative timer presets).

The user requested urgent smoke validation. No full 288-case QMoE coverage,
all-device, LUT refresh sweep, model or Op Profiling system run was performed.
The three coverage tiers and shared timing presets are retained. These are
functional interface checks, not a kernel performance qualification.
